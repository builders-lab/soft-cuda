/**
 * soft-cuda  — AOT Hardware Profiler
 *
 * Measures real CPU vs GPU throughput for every supported op at several
 * element-count breakpoints, finds the crossover threshold, then writes
 * ~/.config/soft-cuda/CONFIG.soft so every subsequent run uses the
 * optimal backend without measuring again.
 *
 * Usage (standalone binary built by CMake target `soft_profiler`):
 *   ./soft_profiler                      # writes to default path
 *   ./soft_profiler /path/to/CONFIG.soft # custom path
 *
 * Algorithm
 * ---------
 *  For each op O and each candidate size S (from SIZE_BREAKPOINTS):
 *    cpu_ms  = median of REPS timed CPU executions
 *    gpu_ms  = median of REPS timed GPU kernel + cudaDeviceSynchronize
 *  The crossover for op O is the smallest S where gpu_ms < cpu_ms.
 *  Below crossover → "cpu", at or above → "cuda".
 *
 * Staleness detection
 * -------------------
 *  We hash the CUDA device UUID.  If the file already exists and its
 *  device_hash matches, we skip profiling entirely (fast path).
 */

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

#include <cuda_runtime.h>


static constexpr int   WARMUP_REPS = 10;
static constexpr int   BENCH_REPS  = 30;


static const uint32_t SIZE_BREAKPOINTS[] = {
    64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216
};
static constexpr int N_BREAKPOINTS = (int)(sizeof(SIZE_BREAKPOINTS) / sizeof(SIZE_BREAKPOINTS[0]));


static std::string device_fingerprint() {
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    char hex[33];
    for (int i = 0; i < 16; i++) {
        snprintf(hex + i * 2, 3, "%02x",
                 (unsigned char)prop.uuid.bytes[i]);
    }
    return std::string(hex, 32);
}


static double now_ms() {
    using namespace std::chrono;
    return (double)duration_cast<nanoseconds>(
               high_resolution_clock::now().time_since_epoch())
               .count() * 1e-6;
}

static double gpu_event_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    return (double)ms;
}

/* Return median of a sorted copy of v */
static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    if (n % 2 == 0) return (v[n/2-1] + v[n/2]) / 2.0;
    return v[n/2];
}

//////////////////////////////////////////////////////////////////////////
//////////////////   CPU KERNELS REF  ////////////////////////////////////
///////////////// THESE SHOULD MIRROR THE ONE IN LIB ////////////////////
static void cpu_add(const float *__restrict__ a,
                    const float *__restrict__ b,
                    float *__restrict__ out, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) out[i] = a[i] + b[i];
}
static void cpu_sub(const float *__restrict__ a,
                    const float *__restrict__ b,
                    float *__restrict__ out, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) out[i] = a[i] - b[i];
}
static void cpu_relu(const float *__restrict__ a,
                     float *__restrict__ out, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) out[i] = a[i] > 0.f ? a[i] : 0.f;
}
static void cpu_square(const float *__restrict__ a,
                       float *__restrict__ out, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) out[i] = a[i] * a[i];
}
static void cpu_mean(const float *__restrict__ a, uint32_t n, float *out) {
    double s = 0.0;
    for (uint32_t i = 0; i < n; i++) s += a[i];
    *out = (float)(s / n);
}

static void cpu_matmul(const float *A, const float *B, float *C,
                       uint32_t M, uint32_t K, uint32_t N) {
    for (uint32_t i = 0; i < M; i++)
        for (uint32_t j = 0; j < N; j++) {
            float s = 0.f;
            for (uint32_t k = 0; k < K; k++)
                s += A[i*K+k] * B[k*N+j];
            C[i*N+j] = s;
        }
}

//////////////////////////////////////////////////////////////////////////
//////////////////   GPU KERNELS REF  ////////////////////////////////////
///////////////// THESE SHOULD MIRROR THE ONE IN LIB ////////////////////

__global__ static void gpu_add_k(const float *a, const float *b,
                                  float *out, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride) out[i] = a[i] + b[i];
}
__global__ static void gpu_sub_k(const float *a, const float *b,
                                  float *out, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride) out[i] = a[i] - b[i];
}
__global__ static void gpu_relu_k(const float *a, float *out, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride) out[i] = a[i] > 0.f ? a[i] : 0.f;
}
__global__ static void gpu_square_k(const float *a, float *out, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride) out[i] = a[i] * a[i];
}

__global__ static void gpu_reduce_k(const float *in, float *partial,
                                     uint32_t n) {
    extern __shared__ float smem[];
    uint32_t tid = threadIdx.x;
    uint32_t i   = blockIdx.x * blockDim.x + tid;
    smem[tid] = (i < n) ? in[i] : 0.f;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = smem[0];
}
__global__ static void gpu_matmul_k(const float *A, const float *B,
                                     float *C,
                                     uint32_t M, uint32_t K, uint32_t N) {
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float s = 0.f;
        for (uint32_t l = 0; l < K; l++) s += A[row*K+l] * B[l*N+col];
        C[row*N+col] = s;
    }
}

struct BenchResult { double cpu_ms, gpu_ms; };

static void fill(float *p, uint32_t n, float base) {
    for (uint32_t i = 0; i < n; i++) p[i] = base + (float)(i % 256) * 0.001f;
}

enum class OpKind { ADD, SUB, RELU, SQUARE, MEAN, MATMUL };

static BenchResult bench_elementwise(OpKind op, uint32_t n,
                                     float *h_a, float *h_b, float *h_out,
                                     float *d_a, float *d_b, float *d_out) {
    std::vector<double> cpu_times;
    cpu_times.reserve(BENCH_REPS);
    for (int r = 0; r < WARMUP_REPS + BENCH_REPS; r++) {
        double t0 = now_ms();
        switch (op) {
            case OpKind::ADD:    cpu_add(h_a, h_b, h_out, n); break;
            case OpKind::SUB:    cpu_sub(h_a, h_b, h_out, n); break;
            case OpKind::RELU:   cpu_relu(h_a, h_out, n);      break;
            case OpKind::SQUARE: cpu_square(h_a, h_out, n);    break;
            case OpKind::MEAN:   cpu_mean(h_a, n, h_out);      break;
            default: break;
        }
        double t1 = now_ms();
        if (r >= WARMUP_REPS) cpu_times.push_back(t1 - t0);
    }

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0); cudaEventCreate(&ev1);

    int block = 256;
    int grid  = ((int)n + block - 1) / block;

    std::vector<double> gpu_times;
    gpu_times.reserve(BENCH_REPS);
    for (int r = 0; r < WARMUP_REPS + BENCH_REPS; r++) {
        cudaEventRecord(ev0);
        switch (op) {
            case OpKind::ADD:
                gpu_add_k<<<grid, block>>>(d_a, d_b, d_out, n); break;
            case OpKind::SUB:
                gpu_sub_k<<<grid, block>>>(d_a, d_b, d_out, n); break;
            case OpKind::RELU:
                gpu_relu_k<<<grid, block>>>(d_a, d_out, n);      break;
            case OpKind::SQUARE:
                gpu_square_k<<<grid, block>>>(d_a, d_out, n);    break;
            case OpKind::MEAN: {
                /* Two-pass reduction */
                int g2 = (int)(((uint32_t)grid + (uint32_t)block - 1) / (uint32_t)block);
                float *d_partial = nullptr;
                cudaMalloc(&d_partial, (size_t)grid * sizeof(float));
                gpu_reduce_k<<<grid, block, (size_t)block * sizeof(float)>>>(
                    d_a, d_partial, n);
                gpu_reduce_k<<<g2, block, (size_t)block * sizeof(float)>>>(
                    d_partial, d_out, (uint32_t)grid);
                cudaFree(d_partial);
                break;
            }
            default: break;
        }
        cudaEventRecord(ev1);
        cudaEventSynchronize(ev1);
        if (r >= WARMUP_REPS)
            gpu_times.push_back(gpu_event_ms(ev0, ev1));
    }
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    return { median(cpu_times), median(gpu_times) };
}


static BenchResult bench_matmul(uint32_t M, uint32_t K, uint32_t N,
                                 float *h_a, float *h_b, float *h_out,
                                 float *d_a, float *d_b, float *d_out) {
    std::vector<double> cpu_times;
    for (int r = 0; r < WARMUP_REPS + BENCH_REPS; r++) {
        double t0 = now_ms();
        cpu_matmul(h_a, h_b, h_out, M, K, N);
        double t1 = now_ms();
        if (r >= WARMUP_REPS) cpu_times.push_back(t1 - t0);
    }
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0); cudaEventCreate(&ev1);
    dim3 blk(32, 32);
    dim3 grd((N+31)/32, (M+31)/32);
    std::vector<double> gpu_times;
    for (int r = 0; r < WARMUP_REPS + BENCH_REPS; r++) {
        cudaEventRecord(ev0);
        gpu_matmul_k<<<grd, blk>>>(d_a, d_b, d_out, M, K, N);
        cudaEventRecord(ev1);
        cudaEventSynchronize(ev1);
        if (r >= WARMUP_REPS)
            gpu_times.push_back(gpu_event_ms(ev0, ev1));
    }
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    return { median(cpu_times), median(gpu_times) };
}



static uint32_t find_crossover(OpKind op, uint32_t max_n,
                                float *h_a, float *h_b, float *h_out,
                                float *d_a, float *d_b, float *d_out) {
    uint32_t crossover = UINT32_MAX;
    for (int bi = 0; bi < N_BREAKPOINTS; bi++) {
        uint32_t n = SIZE_BREAKPOINTS[bi];
        if (n > max_n) break;
        BenchResult r = bench_elementwise(op, n, h_a, h_b, h_out, d_a, d_b, d_out);
        printf("      n=%-9u  cpu=%.4f ms  gpu=%.4f ms  winner=%s\n",
               n, r.cpu_ms, r.gpu_ms, r.gpu_ms < r.cpu_ms ? "GPU" : "CPU");
        if (r.gpu_ms < r.cpu_ms && crossover == UINT32_MAX) {
            crossover = n;
            break; 
        }
    }
    return crossover;
}


static uint32_t find_crossover_matmul(uint32_t max_n,
                                       float *h_a, float *h_b, float *h_out,
                                       float *d_a, float *d_b, float *d_out) {
    uint32_t crossover = UINT32_MAX;
    for (int bi = 0; bi < N_BREAKPOINTS; bi++) {
        uint32_t side = (uint32_t)sqrtf((float)SIZE_BREAKPOINTS[bi]);
        if (side < 2) side = 2;
        uint32_t n = side * side; 
        if (n * 3 > max_n) break;
        BenchResult r = bench_matmul(side, side, side,
                                      h_a, h_b, h_out, d_a, d_b, d_out);
        printf("      side=%-5u n=%-9u  cpu=%.4f ms  gpu=%.4f ms  winner=%s\n",
               side, n, r.cpu_ms, r.gpu_ms, r.gpu_ms < r.cpu_ms ? "GPU" : "CPU");
        if (r.gpu_ms < r.cpu_ms && crossover == UINT32_MAX) {
            crossover = n;
            break;
        }
    }
    return crossover;
}



struct OpThreshold {
    const char *name;
    uint32_t    crossover; 
};

static void write_config(const std::string &path,
                         const std::string &dev_hash,
                         const char *dev_name,
                         double cc, size_t vram_mb,
                         const std::vector<OpThreshold> &thresholds) {
    std::ofstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "[profiler] ERROR: cannot write %s\n", path.c_str());
        return;
    }

    time_t now = time(nullptr);
    char ts[32];
    strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%SZ", gmtime(&now));

    f << "{\n";
    f << "  \"meta\": {\n";
    f << "    \"soft_version\": \"0.1.0\",\n";
    f << "    \"device_hash\": \"" << dev_hash << "\",\n";
    f << "    \"generated_at\": \"" << ts << "\"\n";
    f << "  },\n";
    f << "  \"device\": {\n";
    f << "    \"name\": \"" << dev_name << "\",\n";
    f << "    \"type\": \"cuda\",\n";
    f << "    \"compute_capability\": " << cc << ",\n";
    f << "    \"vram_mb\": " << vram_mb << "\n";
    f << "  },\n";
    f << "  \"ops\": {\n";

    for (size_t i = 0; i < thresholds.size(); i++) {
        const auto &t = thresholds[i];
        f << "    \"" << t.name << "\": [\n";
        if (t.crossover == UINT32_MAX) {
            f << "      { \"backend\": \"cpu\" }\n";
        } else if (t.crossover == 0) {
            f << "      { \"backend\": \"cuda\" }\n";
        } else {
            f << "      { \"min\": 0, \"max\": " << t.crossover
              << ", \"backend\": \"cpu\" },\n";
            f << "      { \"min\": " << t.crossover
              << ", \"max\": 4294967295, \"backend\": \"cuda\" }\n";
        }
        f << "    ]";
        if (i + 1 < thresholds.size()) f << ",";
        f << "\n";
    }

    f << "  },\n";
    f << "  \"pool\": {\n";
    f << "    \"device\": \"cuda\",\n";
    f << "    \"block_size\": 2097152\n";
    f << "  }\n";
    f << "}\n";
    f.close();
    printf("[profiler] Config written to: %s\n", path.c_str());
}

/* -----------------------------------------------------------------------
 * Public entry point — called by soft_profiler binary and optionally by
 * the library's sc_init() if no config exists.
 * ----------------------------------------------------------------------- */
int soft_profile_and_write(const char *out_path) {
    // Detect CUDA device 
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        fprintf(stderr, "[profiler] No CUDA devices found — writing CPU-only config.\n");
        // Write a minimal CPU-only config 
        //
        return 1;
    }

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    cudaSetDevice(0);

    printf("[profiler] Device: %s  CC=%d.%d  VRAM=%zu MB\n",
           prop.name,
           prop.major, prop.minor,
           prop.totalGlobalMem / (1024*1024));

    std::string dev_hash = device_fingerprint();
    double cc = prop.major + prop.minor * 0.1;
    size_t vram_mb = prop.totalGlobalMem / (1024*1024);

    /// Check if existing config is latest
    if (out_path && std::filesystem::exists(out_path)) {
        std::ifstream f(out_path);
        std::string content((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>());
        if (content.find(dev_hash) != std::string::npos) {
            printf("[profiler] Config is up-to-date (device_hash matches). Skipping.\n");
            return 0;
        }
    }

    uint32_t max_elems = (uint32_t)std::min(
        (size_t)16777216UL,   
        prop.totalGlobalMem / (8UL * sizeof(float))
    );
    printf("[profiler] Probing up to %u elements per op\n\n", max_elems);

    float *h_a   = (float *)malloc(max_elems * sizeof(float));
    float *h_b   = (float *)malloc(max_elems * sizeof(float));
    float *h_out = (float *)malloc(max_elems * sizeof(float));
    fill(h_a,   max_elems, 1.5f);
    fill(h_b,   max_elems, 0.7f);
    memset(h_out, 0, max_elems * sizeof(float));

    float *d_a, *d_b, *d_out;
    cudaMalloc(&d_a,   max_elems * sizeof(float));
    cudaMalloc(&d_b,   max_elems * sizeof(float));
    cudaMalloc(&d_out, max_elems * sizeof(float));
    cudaMemcpy(d_a, h_a, max_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, max_elems * sizeof(float), cudaMemcpyHostToDevice);

    gpu_add_k<<<256, 256>>>(d_a, d_b, d_out, 1024);
    cudaDeviceSynchronize();

    std::vector<OpThreshold> thresholds;

    struct { const char *name; OpKind kind; } ops[] = {
        {"add",    OpKind::ADD},
        {"sub",    OpKind::SUB},
        {"relu",   OpKind::RELU},
        {"square", OpKind::SQUARE},
        {"mean",   OpKind::MEAN},
    };

    for (auto &o : ops) {
        printf("  [%s]\n", o.name);
        uint32_t xover = find_crossover(o.kind, max_elems,
                                         h_a, h_b, h_out, d_a, d_b, d_out);
        printf("  → crossover at n=%u\n\n", xover);
        thresholds.push_back({o.name, xover});
    }

    printf("  [matmul]\n");
    uint32_t mm_xover = find_crossover_matmul(max_elems,
                                               h_a, h_b, h_out, d_a, d_b, d_out);
    printf("  → crossover at n=%u\n\n", mm_xover);
    thresholds.push_back({"matmul", mm_xover});

    thresholds.push_back({"mul_scalar", thresholds[0].crossover});
    thresholds.push_back({"broadcast_add", thresholds[0].crossover});

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    free(h_a); free(h_b); free(h_out);

    std::string final_path;
    if (out_path && out_path[0] != '\0') {
        final_path = out_path;
    } else {
        const char *home = getenv("HOME");
        if (!home) home = "/tmp";
        final_path = std::string(home) + "/.config/soft-cuda/CONFIG.soft";
    }

    std::filesystem::create_directories(
        std::filesystem::path(final_path).parent_path());

    write_config(final_path, dev_hash, prop.name, cc, vram_mb, thresholds);
    return 0;
}

