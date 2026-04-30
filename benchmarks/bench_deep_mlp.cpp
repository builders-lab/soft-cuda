/**
 * bench_deep_mlp.cpp
 *
 * Implements a 4-layer Deep MLP to stress-test the Hybrid Dispatcher.
 * Network:  Input (784) -> Hidden1 (512) -> Hidden2 (256) -> Hidden3 (128) -> Output (10)
 *
 * This benchmark demonstrates:
 *   1. Correct routing of small vs large layers.
 *   2. Persistence of GPU memory across 1000 iterations.
 *   3. Hybrid execution benefit (dispatching compute-heavy layers to GPU,
 *      and memory-bound/small layers to CPU).
 */

#include "soft-cuda/tensor/api.h"
#include "soft-cuda/python/soft_cuda_python.h"

#include <chrono>
#include <cstdio>
#include <vector>
#include <cmath>
#include <cassert>

static double now_ms() {
    using namespace std::chrono;
    return (double)duration_cast<nanoseconds>(
               high_resolution_clock::now().time_since_epoch())
               .count() * 1e-6;
}

struct Layer {
    sc_tensor_t *W;
    sc_tensor_t *b;
};

static Layer create_layer(sc_pool_t *pool, uint32_t in_dim, uint32_t out_dim) {
    uint32_t dW[] = {in_dim, out_dim};
    uint32_t db[] = {1, out_dim};
    sc_tensor_t *W = sc_tensor_create(pool, SC_DTYPE_FLOAT32, 2, dW, NULL, 1);
    sc_tensor_t *b = sc_tensor_create(pool, SC_DTYPE_FLOAT32, 2, db, NULL, 1);
    sc_tensor_fill_random_normal(W, 0.0f, sqrtf(2.0f / (float)in_dim)); // He init
    sc_tensor_fill_random_normal(b, 0.0f, 0.01f);
    return {W, b};
}

static sc_tensor_t* forward_layer(sc_pool_t *pool, sc_tensor_t *X, Layer &L, bool use_relu = true) {
    sc_tensor_t *mat = sc_tensor_mul_naive(pool, X, L.W);
    sc_tensor_t *add = sc_tensor_add(pool, mat, L.b);
    if (use_relu) return sc_tensor_relu(pool, add);
    return add;
}

void run_mlp_bench(int backend_mode, const char *label, int epochs = 100) {
    printf("--- Benchmarking %s ---\n", label);

    const uint32_t BATCH = 64;
    const uint32_t D_IN = 784, D1 = 512, D2 = 256, D3 = 128, D_OUT = 10;

    sc_pool_t *pool = sc_pool_create(128 * 1024 * 1024, 0); // 128MB CPU
    sc_pool_t *meta = sc_pool_create(8 * 1024 * 1024, 0);   // 8MB Meta
    sc_pool_t *gpc  = sc_pool_create(32 * 1024 * 1024, 0);  // 32MB Grad CPU
    
    bool use_gpu = (backend_mode != SC_BACKEND_CPU);
    sc_pool_t *gpg  = sc_pool_create(use_gpu ? 32 * 1024 * 1024 : 1024, use_gpu ? 1 : 0);
    sc_pool_t *pgpu = sc_pool_create(use_gpu ? 128 * 1024 * 1024 : 1024, use_gpu ? 1 : 0);

    if (!pool || !meta || !gpc || !gpg || !pgpu) {
        printf("FAILED to allocate pools (backend_mode=%d)\n", backend_mode);
        if (pool) sc_pool_destroy(pool);
        if (meta) sc_pool_destroy(meta);
        if (gpc) sc_pool_destroy(gpc);
        if (gpg) sc_pool_destroy(gpg);
        if (pgpu) sc_pool_destroy(pgpu);
        return;
    }

    uint32_t dX[] = {BATCH, D_IN};
    uint32_t dY[] = {BATCH, D_OUT};
    sc_tensor_t *X = sc_tensor_create(pool, SC_DTYPE_FLOAT32, 2, dX, NULL, 0);
    sc_tensor_t *Y = sc_tensor_create(pool, SC_DTYPE_FLOAT32, 2, dY, NULL, 0);
    sc_tensor_fill_random_normal(X, 0.5f, 0.2f);
    sc_tensor_fill_random_normal(Y, 0.1f, 0.05f);

    Layer L1 = create_layer(pool, D_IN, D1);
    Layer L2 = create_layer(pool, D1, D2);
    Layer L3 = create_layer(pool, D2, D3);
    Layer L4 = create_layer(pool, D3, D_OUT);

    // Build Graph
    sc_tensor_t *H1 = forward_layer(pool, X, L1);
    sc_tensor_t *H2 = forward_layer(pool, H1, L2);
    sc_tensor_t *H3 = forward_layer(pool, H2, L3);
    sc_tensor_t *Yp = forward_layer(pool, H3, L4, false);
    
    sc_tensor_t *diff = sc_tensor_sub(pool, Yp, Y);
    sc_tensor_t *sq   = sc_tensor_square(pool, diff);
    sc_tensor_t *loss = sc_tensor_mean(pool, sq);

    sc_graph_t *g = sc_build_graph(meta, pgpu, gpc, gpg, loss, backend_mode);
    if (!g) {
        printf("FAILED to build graph\n");
        return;
    }

    // Warmup
    sc_graph_step(pool, pgpu, g, 0.01f);

    double t0 = now_ms();
    for (int i = 0; i < epochs; i++) {
        sc_graph_step(pool, pgpu, g, 0.01f);
        if ((i+1) % (epochs/5) == 0) {
             printf("  Epoch %d/%d | Loss: %.6f\n", i+1, epochs, sc_graph_get_loss(g));
        }
    }
    double elapsed = now_ms() - t0;

    printf("  [RESULT] Total: %.2f ms | Avg: %.2f ms/step\n\n", elapsed, elapsed / epochs);

    sc_graph_destroy(g);
    sc_pool_destroy(pool); sc_pool_destroy(meta);
    sc_pool_destroy(gpc); sc_pool_destroy(gpg); sc_pool_destroy(pgpu);
}

int main() {
    printf("\n--- Deep MLP Benchmark (4-Layer, Hybrid Dispatch) ---\n\n");

    // CPU Reference
    run_mlp_bench(SC_BACKEND_CPU, "CPU-Only Backend (Baseline)");

    //  GPU Reference
    run_mlp_bench(SC_BACKEND_GPU, "GPU-Only Backend (Full Acceleration)");

    //  Hybrid Dispatch (AOT-Profiled)
    // This will use thresholds from CONFIG.soft
    run_mlp_bench(SC_BACKEND_HYBRID, "HYBRID Backend (AOT-Optimized)");

    return 0;
}
