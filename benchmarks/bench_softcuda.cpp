/* Build:  (benchmarks/CMakeLists.txt)
 * Run:    ./bench_softcuda
 */

#include "soft-cuda/tensor/api.h"
#include "soft-cuda/python/soft_cuda_python.h"
#include <cublas_v2.h>

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>

static double now_ms() {
    using namespace std::chrono;
    return (double)duration_cast<nanoseconds>(
               high_resolution_clock::now().time_since_epoch())
               .count() * 1e-6;
}

static void hline() { printf("─────────────────────────────────────────────────────────────────\n"); }
static void header(const char *s) {
    hline();
    printf("  %s\n", s);
    hline();
}
static void result(const char *label, double ms, double ops_billions) {
    if (ops_billions > 0)
        printf("  %-30s  %8.2f ms   %6.2f GFLOPs\n", label, ms, ops_billions / ms * 1e3);
    else
        printf("  %-30s  %8.2f ms\n", label, ms);
}

static void bench_add() {
    header("Benchmark 1: Element-wise ADD  (1M float32)");

    const uint32_t N = 1024 * 1024;
    float *data_a = new float[N];
    float *data_b = new float[N];
    for (uint32_t i = 0; i < N; i++) { data_a[i] = (float)i * 0.001f; data_b[i] = 1.0f; }

    uint32_t dims[] = {N};
    const int REPS = 20;

    {
        sc_pool_t *pool = sc_pool_create((size_t)N * 4 * 10, 0);
        sc_pool_t *meta = sc_pool_create(2*1024*1024, 0);
        sc_pool_t *gpc  = sc_pool_create((size_t)N * 4 * 2, 0);
        sc_pool_t *gpg  = sc_pool_create(2*1024*1024, 0);  /* dummy */
        sc_pool_t *pgpu = sc_pool_create(2*1024*1024, 0);  /* dummy CPU pool */

        sc_tensor_t *a = sc_tensor_create(pool, SC_DTYPE_FLOAT32, 1, dims, data_a, 0);
        sc_tensor_t *b = sc_tensor_create(pool, SC_DTYPE_FLOAT32, 1, dims, data_b, 0);
        sc_tensor_t *c = sc_tensor_add(pool, a, b);
        sc_graph_t *g  = sc_build_graph(meta, pgpu, gpc, gpg, c, SC_BACKEND_CPU);

        double total = 0;
        for (int r = 0; r < REPS; r++) {
            double t0 = now_ms();
            sc_graph_forward(pool, pgpu, g);
            total += now_ms() - t0;
        }
        sc_graph_destroy(g);

        result("ADD [CPU]", total / REPS, (double)N * 1e-9);
        sc_pool_destroy(pool); sc_pool_destroy(meta);
        sc_pool_destroy(gpc); sc_pool_destroy(gpg); sc_pool_destroy(pgpu);
    }

    {
        sc_pool_t *pool = sc_pool_create((size_t)N * 4 * 10, 0);
        sc_pool_t *meta = sc_pool_create(2*1024*1024, 0);
        sc_pool_t *gpc  = sc_pool_create((size_t)N * 4 * 2, 0);
        sc_pool_t *gpg  = sc_pool_create((size_t)N * 4 * 2, 1);
        sc_pool_t *pgpu = sc_pool_create((size_t)N * 4 * 10, 1);

        sc_tensor_t *a = sc_tensor_create(pool, SC_DTYPE_FLOAT32, 1, dims, data_a, 0);
        sc_tensor_t *b = sc_tensor_create(pool, SC_DTYPE_FLOAT32, 1, dims, data_b, 0);
        sc_tensor_t *c = sc_tensor_add(pool, a, b);
        sc_graph_t *g  = sc_build_graph(meta, pgpu, gpc, gpg, c, SC_BACKEND_GPU);

        double total = 0;
        for (int r = 0; r < REPS; r++) {
            double t0 = now_ms();
            sc_graph_forward(pool, pgpu, g);
            total += now_ms() - t0;
        }
        sc_graph_destroy(g);
        result("ADD [GPU]", total / REPS, (double)N * 1e-9);
        sc_pool_destroy(pool); sc_pool_destroy(meta);
        sc_pool_destroy(gpc); sc_pool_destroy(gpg); sc_pool_destroy(pgpu);
    }

    delete[] data_a; delete[] data_b;
    printf("\n");
}

static void bench_matmul() {
    header("Benchmark 2: Matmul 4096×4096");

    const uint32_t M = 4096, K = 4096, N = 4096;
    const uint32_t total_flops = 2 * M * K * N;  
    float *A = new float[M * K];
    float *B = new float[K * N];
    for (uint32_t i = 0; i < M*K; i++) A[i] = (float)i * 0.001f;
    for (uint32_t i = 0; i < K*N; i++) B[i] = (float)i * 0.001f;

    uint32_t dA[] = {M, K}, dB[] = {K, N};
    const int REPS = 10;

    auto run = [&](int backend_flag, const char *label) {
        bool gpu = (backend_flag == SC_BACKEND_GPU);
        size_t pool_sz = (size_t)(M*K + K*N + M*N) * 4 * 4;
        size_t vram_sz = gpu ? pool_sz : 1024;

        sc_pool_t *pool = sc_pool_create(pool_sz, 0);
        sc_pool_t *meta = sc_pool_create(2*1024*1024, 0);
        sc_pool_t *gpc  = sc_pool_create(2*1024*1024, 0);
        sc_pool_t *gpg  = sc_pool_create(2*1024*1024, gpu ? 1 : 0);
        sc_pool_t *pgpu = sc_pool_create(vram_sz + 1024*1024, gpu ? 1 : 0);

        sc_tensor_t *a = sc_tensor_create(pool, SC_DTYPE_FLOAT32, 2, dA, A, 0);
        sc_tensor_t *b = sc_tensor_create(pool, SC_DTYPE_FLOAT32, 2, dB, B, 0);
        sc_tensor_t *c = sc_tensor_mul_naive(pool, a, b);
        sc_graph_t *g  = sc_build_graph(meta, pgpu, gpc, gpg, c, backend_flag);

        double total = 0;
        for (int r = 0; r < REPS; r++) {
            double t0 = now_ms();
            sc_graph_forward(pool, pgpu, g);
            total += now_ms() - t0;
        }
        sc_graph_destroy(g);
        result(label, total / REPS, (double)total_flops * 1e-9);
        sc_pool_destroy(pool); sc_pool_destroy(meta);
        sc_pool_destroy(gpc); sc_pool_destroy(gpg); sc_pool_destroy(pgpu);
    };
    auto run_cublas = [&]() {
        float *d_a, *d_b, *d_c;
        cudaMalloc((void**)&d_a, M*K*sizeof(float));
        cudaMalloc((void**)&d_b, K*N*sizeof(float));
        cudaMalloc((void**)&d_c, M*N*sizeof(float));
        cudaMemcpy(d_a, A, M*K*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, B, K*N*sizeof(float), cudaMemcpyHostToDevice);

        cublasHandle_t handle;
        cublasCreate(&handle);
        float alpha = 1.0f, beta = 0.0f;

        // warmup
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_b, N, d_a, K, &beta, d_c, N);
        cudaDeviceSynchronize();

        double total = 0;
        for (int r = 0; r < REPS; r++) {
            double t0 = now_ms();
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_b, N, d_a, K, &beta, d_c, N);
            cudaDeviceSynchronize();
            total += now_ms() - t0;
        }

        double ms = total / REPS;
        double gflops = (double)total_flops / ms * 1e-6;
        printf("  %-34s  %8.2f ms   %6.2f GFLOPs\n", "Matmul 4096x4096 [cuBLAS]", ms, gflops);

        cublasDestroy(handle);
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    };

    // run(SC_BACKEND_CPU, "Matmul 512x512 [CPU naive]");
    run(SC_BACKEND_GPU, "Matmul 4096x4096 [GPU sgemm]");
    printf("-----------------------------------------RUNING CUBLAS_OP_N--------------------------------------\n");
    run_cublas();

    delete[] A; delete[] B;
    printf("\n");
}




static void bench_xor_training(int backend_flag, const char *label) {
    float X_data[] = {0,0, 0,1, 1,0, 1,1};
    float Y_data[] = {0, 1, 1, 0};
    uint32_t dX[]  = {4, 2}, dY[] = {4, 1};
    uint32_t dW1[] = {2, 4}, db1[] = {1, 4};
    uint32_t dW2[] = {4, 1}, db2[] = {1, 1};
    const int EPOCHS = 10000;

    bool gpu = (backend_flag != SC_BACKEND_CPU);
    sc_pool_t *pool  = sc_pool_create(4*1024*1024, 0);
    sc_pool_t *meta  = sc_pool_create(2*1024*1024, 0);
    sc_pool_t *gpc   = sc_pool_create(1024*1024, 0);
    sc_pool_t *gpg   = sc_pool_create(gpu ? 1024*1024 : 1024, gpu ? 1 : 0);
    sc_pool_t *pgpu  = sc_pool_create(gpu ? 4*1024*1024 : 1024, gpu ? 1 : 0);

    sc_tensor_t *X  = sc_tensor_create(pool, SC_DTYPE_FLOAT32, 2, dX, X_data, 0);
    assert(pool != NULL && meta != NULL && gpc != NULL && gpg != NULL && pgpu != NULL);
    sc_tensor_t *Y  = sc_tensor_create(pool, SC_DTYPE_FLOAT32, 2, dY, Y_data, 0);
    sc_tensor_t *W1 = sc_tensor_create(pool, SC_DTYPE_FLOAT32, 2, dW1, NULL, 1);
    sc_tensor_t *b1 = sc_tensor_create(pool, SC_DTYPE_FLOAT32, 2, db1, NULL, 1);
    sc_tensor_t *W2 = sc_tensor_create(pool, SC_DTYPE_FLOAT32, 2, dW2, NULL, 1);
    sc_tensor_t *b2 = sc_tensor_create(pool, SC_DTYPE_FLOAT32, 2, db2, NULL, 1);
    sc_tensor_fill_random_normal(W1, 0.0f, 0.3f);
    sc_tensor_fill_random_normal(W2, 0.0f, 0.3f);

    sc_tensor_t *H1   = sc_tensor_mul_naive(pool, X, W1);
    sc_tensor_t *H1b  = sc_tensor_add(pool, H1, b1);
    sc_tensor_t *H    = sc_tensor_relu(pool, H1b);
    sc_tensor_t *O    = sc_tensor_mul_naive(pool, H, W2);
    sc_tensor_t *Yp   = sc_tensor_add(pool, O, b2);
    sc_tensor_t *diff = sc_tensor_sub(pool, Yp, Y);
    sc_tensor_t *sq   = sc_tensor_square(pool, diff);
    sc_tensor_t *loss = sc_tensor_mean(pool, sq);

    sc_graph_t *g = sc_build_graph(meta, pgpu, gpc, gpg, loss, backend_flag);
    assert(g != NULL);

    double t0 = now_ms();
    for (int e = 0; e < EPOCHS; e++)
        sc_graph_step(pool, pgpu, g, 0.05f);
    double elapsed = now_ms() - t0;

    float final_loss = sc_graph_get_loss(g);
    printf("  %-32s  %8.2f ms   loss=%.6f  (%d epochs)\n",
           label, elapsed, final_loss, EPOCHS);

    sc_graph_destroy(g);
    sc_pool_destroy(pool); sc_pool_destroy(meta);
    sc_pool_destroy(gpc); sc_pool_destroy(gpg); sc_pool_destroy(pgpu);
}

static void bench_before_after_profiling() {
    header("Benchmark 4: Before/After Profiling Benefit");

    printf("  [BEFORE profiling -> uses hardcoded default thresholds]\n");
    {
        system("mv ~/.config/soft-cuda/CONFIG.soft "
               "~/.config/soft-cuda/CONFIG.soft.bak 2>/dev/null || true");
        bench_xor_training(SC_BACKEND_HYBRID, "XOR HYBRID [before profiling]");
        system("mv ~/.config/soft-cuda/CONFIG.soft.bak "
               "~/.config/soft-cuda/CONFIG.soft 2>/dev/null || true");
    }

    printf("\n  [AFTER profiling → real measured thresholds]\n");
    bench_xor_training(SC_BACKEND_HYBRID, "XOR HYBRID [after profiling]");
    printf("\n");
}

static void bench_hybrid_network() {
    header("Benchmark 6: Complex Hybrid Network (Forward Pass)");

    const int B = 128; // Batch size
    const int D_IN = 256, D_HID = 256, D_SMALL = 16, D_OUT = 1;

    uint32_t dX[] = {B, D_IN};
    uint32_t dW1[] = {D_IN, D_HID}, db1[] = {1, D_HID};
    uint32_t dW2[] = {D_HID, D_SMALL}, db2[] = {1, D_SMALL};
    uint32_t dW3[] = {D_SMALL, D_OUT}, db3[] = {1, D_OUT};

    sc_pool_t *pool = sc_pool_create(16*1024*1024, 0);
    sc_pool_t *meta = sc_pool_create(4*1024*1024, 0);
    sc_pool_t *gpc  = sc_pool_create(4*1024*1024, 0);
    sc_pool_t *gpg  = sc_pool_create(4*1024*1024, 1);
    sc_pool_t *pgpu = sc_pool_create(16*1024*1024, 1);

    sc_tensor_t *X  = sc_tensor_create(pool, SC_DTYPE_FLOAT32, 2, dX, NULL, 0);
    sc_tensor_t *W1 = sc_tensor_create(pool, SC_DTYPE_FLOAT32, 2, dW1, NULL, 0);
    sc_tensor_t *b1 = sc_tensor_create(pool, SC_DTYPE_FLOAT32, 2, db1, NULL, 0);
    sc_tensor_t *W2 = sc_tensor_create(pool, SC_DTYPE_FLOAT32, 2, dW2, NULL, 0);
    sc_tensor_t *b2 = sc_tensor_create(pool, SC_DTYPE_FLOAT32, 2, db2, NULL, 0);
    sc_tensor_t *W3 = sc_tensor_create(pool, SC_DTYPE_FLOAT32, 2, dW3, NULL, 0);
    sc_tensor_t *b3 = sc_tensor_create(pool, SC_DTYPE_FLOAT32, 2, db3, NULL, 0);

    // Layer 1: GPU (32k elements)
    sc_tensor_t *L1_mat = sc_tensor_mul_naive(pool, X, W1);
    sc_tensor_t *L1_add = sc_tensor_add(pool, L1_mat, b1);
    sc_tensor_t *L1_out = sc_tensor_relu(pool, L1_add);

    // Layer 2: Mixed (2k elements -> Matmul on GPU, Add/ReLU on CPU)
    sc_tensor_t *L2_mat = sc_tensor_mul_naive(pool, L1_out, W2);
    sc_tensor_t *L2_add = sc_tensor_add(pool, L2_mat, b2);
    sc_tensor_t *L2_out = sc_tensor_relu(pool, L2_add);

    // Layer 3: CPU (128 elements)
    sc_tensor_t *L3_mat = sc_tensor_mul_naive(pool, L2_out, W3);
    sc_tensor_t *Yp     = sc_tensor_add(pool, L3_mat, b3);

    sc_graph_t *g = sc_build_graph(meta, pgpu, gpc, gpg, Yp, SC_BACKEND_HYBRID);
    
    double t0 = now_ms();
    sc_graph_forward(pool, pgpu, g);
    double elapsed = now_ms() - t0;

    result("Hybrid Complex Network", elapsed, 0);

    sc_graph_destroy(g);
    sc_pool_destroy(pool); sc_pool_destroy(meta);
    sc_pool_destroy(gpc); sc_pool_destroy(gpg); sc_pool_destroy(pgpu);
    printf("\n");
}

// Bench: Large pipeline (add+relu+mean on 4M elements)=================
static void bench_large_pipeline() {
    header("Benchmark 3: XOR Training  (10000 epochs)");
    bench_xor_training(SC_BACKEND_CPU, "XOR [CPU backend]");
    bench_xor_training(SC_BACKEND_GPU, "XOR [GPU backend]");
    printf("\n");
}


// BIG test
static void bench_billion() {
    header("Benchmark 5: 100-Million Element Reduction (Mean)");

    // 100 Million elements = 400 MB of RAM/VRAM
    const uint32_t N = 100 * 1000 * 1000;
    uint32_t d[] = {N};
    
    printf("  [Allocating 400MB Host Memory...]\n");
    fflush(stdout);
    float *data = new float[N];
    for (uint32_t i = 0; i < N; i++) data[i] = 1.0f; // Expected mean is 1.0

    auto run = [&](int backend_flag, const char *label) {
        bool gpu = (backend_flag == SC_BACKEND_GPU);
        size_t vram_sz = gpu ? (size_t)N * 4 + 4*1024*1024 : 1024;
        
        sc_pool_t *pool = sc_pool_create((size_t)N * 4 + 4*1024*1024, 0);
        sc_pool_t *meta = sc_pool_create(4*1024*1024, 0);
        sc_pool_t *gpc  = sc_pool_create(4*1024*1024, 0);
        sc_pool_t *gpg  = sc_pool_create(gpu ? 4*1024*1024 : 1024, gpu ? 1 : 0);
        sc_pool_t *pgpu = sc_pool_create(vram_sz, gpu ? 1 : 0);

        if (!pool || !meta || !gpc || !gpg || !pgpu) {
            printf("  [ERROR] Memory allocation failed! pool=%p, meta=%p, pgpu=%p (vram_sz=%zu)\n", pool, meta, pgpu, vram_sz);
            if (pool) sc_pool_destroy(pool);
            if (meta) sc_pool_destroy(meta);
            if (gpc) sc_pool_destroy(gpc);
            if (gpg) sc_pool_destroy(gpg);
            if (pgpu) sc_pool_destroy(pgpu);
            return;
        }

        sc_tensor_t *A = sc_tensor_create(pool, SC_DTYPE_FLOAT32, 1, d, data, 0);
        sc_tensor_t *M = sc_tensor_mean(pool, A);

        sc_graph_t *g = sc_build_graph(meta, pgpu, gpc, gpg, M, backend_flag);
        assert(g != NULL && "Graph build failed");

        double t0 = now_ms();
        sc_graph_forward(pool, pgpu, g);
        double elapsed = now_ms() - t0;
        
        float result_val = sc_graph_get_loss(g);
        
        printf("  %-32s  %8.2f ms   (Mean=%.1f)\n", label, elapsed, result_val);
        
        sc_graph_destroy(g);
        sc_pool_destroy(pool); sc_pool_destroy(meta);
        sc_pool_destroy(gpc); sc_pool_destroy(gpg); sc_pool_destroy(pgpu);
    };

    run(SC_BACKEND_CPU, "Mean 1B [CPU]");
    run(SC_BACKEND_GPU, "Mean 1B [GPU]");

    delete[] data;
    printf("\n");
}

int main(void) {
    printf("\n");
    hline();
    printf("||           soft-cuda  vs  reference  —  Benchmark Suite         ||\n");
    printf("||     (Run bench_pytorch.py for PyTorch comparison numbers)      ||\n");
    hline();
    bench_add();
    bench_matmul();
    bench_large_pipeline();
    bench_before_after_profiling();
    bench_hybrid_network();
    bench_billion();

    hline();
    printf("  Done.\n");
    hline();
    return 0;
}
