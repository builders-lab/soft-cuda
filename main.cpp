/* ═══════════════════════════════════════════════════════════════════════════
 *  Written by  : Antigravity (AI Coding Assistant)
 *  Date        : 2026-04-21  02:12 IST
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 *  soft-cuda  —  XOR demo using the flat-C Python bridge API
 *
 *  This file demonstrates how to build and train a tiny neural network
 *  (2 → 4 → 1, XOR problem) using only the sc_* bridge functions.
 *
 *  The exact same function calls work from Python via ctypes/cffi
 *  because every sc_* symbol has extern "C" linkage and uses only
 *  C-safe primitives  (no std::vector, no enum class, no default args).
 *
 *  Network architecture:
 *
 *        X  [4×2]
 *         │
 *    ┌────┴────┐
 *    │ matmul  │ W1 [2×4]
 *    └────┬────┘
 *         │
 *    ┌────┴────┐
 *    │  + b1   │ b1 [1×4]
 *    └────┬────┘
 *         │
 *    ┌────┴────┐
 *    │  ReLU   │
 *    └────┬────┘
 *         │
 *    ┌────┴────┐
 *    │ matmul  │ W2 [4×1]
 *    └────┬────┘
 *         │
 *    ┌────┴────┐
 *    │  + b2   │ b2 [1×1]
 *    └────┬────┘
 *         │
 *       Y_pred  [4×1]
 *         │
 *    ┌────┴────┐
 *    │ MSE     │─── Y (ground truth)
 *    └────┬────┘
 *         │
 *       loss (scalar)
 */

#include "soft-cuda/python/soft_cuda_python.h"     /* the ONLY header you need */
#include <stdio.h>
#include <assert.h>

/* ═══════════════════════════════════════════════════════════════════════════
 *  Helpers  — nice printing without pulling in <iostream>
 * ═══════════════════════════════════════════════════════════════════════════ */

static void banner(const char *msg) {
    printf("\n");
    printf("╔══════════════════════════════════════════════╗\n");
    printf("║  %-44s║\n", msg);
    printf("╚══════════════════════════════════════════════╝\n");
}

static void separator(void) {
    printf("──────────────────────────────────────────────\n");
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  main
 * ═══════════════════════════════════════════════════════════════════════════ */

int main(void) {

    /* ─────────────────────────────────────────────
     *  1 )  CREATE MEMORY POOLS
     *
     *  soft-cuda uses arena-style allocation.
     *  You need separate pools for:
     *    • data tensors   (pool)
     *    • graph metadata (pool_meta)
     *    • GPU VRAM       (pool_gpu)           ← on_device = 1
     *    • CPU gradients  (pool_grad_cpu)
     *    • GPU gradients  (pool_grad_gpu)      ← on_device = 1
     * ───────────────────────────────────────────── */

    banner("1. Creating memory pools");

    sc_pool_t *pool          = sc_pool_create(1024 * 1024, 0);  /* 1 MB CPU  */
    sc_pool_t *pool_meta     = sc_pool_create(1024 * 1024, 0);  /* 1 MB CPU  */
    sc_pool_t *pool_grad_cpu = sc_pool_create(1024 * 1024, 0);  /* 1 MB CPU  */
    sc_pool_t *pool_gpu      = sc_pool_create(1024 * 1024, 1);  /* 1 MB VRAM */
    sc_pool_t *pool_grad_gpu = sc_pool_create(1024 * 1024, 1);  /* 1 MB VRAM */

    assert(pool          != NULL);
    assert(pool_meta     != NULL);
    assert(pool_grad_cpu != NULL);
    assert(pool_gpu      != NULL);
    assert(pool_grad_gpu != NULL);

    printf("  pool      : %zu bytes total, %zu used\n",
           sc_pool_size(pool), sc_pool_used(pool));

    /* ─────────────────────────────────────────────
     *  2 )  CREATE INPUT / TARGET TENSORS
     *
     *  sc_tensor_create(pool, dtype, ndims, dims, data, grad)
     *
     *  • dtype: SC_DTYPE_FLOAT32 (= 4)
     *  • grad = 0  for data tensors  (X, Y)
     *  • grad = 1  for trainable weights  (W, b)
     * ───────────────────────────────────────────── */

    banner("2. Creating data tensors");

    /* XOR truth table */
    float val_X[] = { 0,0,  0,1,  1,0,  1,1 };
    float val_Y[] = { 0,     1,    1,    0  };

    uint32_t dims_X[]  = {4, 2};
    uint32_t dims_Y[]  = {4, 1};

    sc_tensor_t *X = sc_tensor_create(pool, SC_DTYPE_FLOAT32, 2, dims_X, val_X, 0);
    sc_tensor_t *Y = sc_tensor_create(pool, SC_DTYPE_FLOAT32, 2, dims_Y, val_Y, 0);

    assert(X != NULL);
    assert(Y != NULL);

    printf("  X tensor id = %u   (ndims=%u,  dims=[%u, %u])\n",
           sc_tensor_id(X),
           sc_tensor_get_ndims(X),
           sc_tensor_get_dims(X)[0],
           sc_tensor_get_dims(X)[1]);

    printf("  Y tensor id = %u   (ndims=%u,  dims=[%u, %u])\n",
           sc_tensor_id(Y),
           sc_tensor_get_ndims(Y),
           sc_tensor_get_dims(Y)[0],
           sc_tensor_get_dims(Y)[1]);

    /* ─────────────────────────────────────────────
     *  3 )  CREATE TRAINABLE WEIGHT TENSORS
     *
     *  Pass NULL for data → initialized to zero.
     *  Then fill with random normals for gradient-friendly init.
     * ───────────────────────────────────────────── */

    banner("3. Creating trainable weights");

    uint32_t dims_W1[] = {2, 4};     /* input → hidden  */
    uint32_t dims_b1[] = {1, 4};
    uint32_t dims_W2[] = {4, 1};     /* hidden → output */
    uint32_t dims_b2[] = {1, 1};

    sc_tensor_t *W1 = sc_tensor_create(pool, SC_DTYPE_FLOAT32, 2, dims_W1, NULL, 1);
    sc_tensor_t *b1 = sc_tensor_create(pool, SC_DTYPE_FLOAT32, 2, dims_b1, NULL, 1);
    sc_tensor_t *W2 = sc_tensor_create(pool, SC_DTYPE_FLOAT32, 2, dims_W2, NULL, 1);
    sc_tensor_t *b2 = sc_tensor_create(pool, SC_DTYPE_FLOAT32, 2, dims_b2, NULL, 1);

    /* Xavier-ish init */
    sc_tensor_fill_random_normal(W1, 0.5f, 0.2f);
    sc_tensor_fill_random_normal(W2, 0.5f, 0.2f);
    sc_tensor_fill_random_normal(b1, 0.0f, 0.1f);
    sc_tensor_fill_random_normal(b2, 0.0f, 0.1f);

    printf("  W1 [%u×%u]  grad=ON   id=%u\n",
           sc_tensor_get_dims(W1)[0], sc_tensor_get_dims(W1)[1], sc_tensor_id(W1));
    printf("  b1 [%u×%u]  grad=ON   id=%u\n",
           sc_tensor_get_dims(b1)[0], sc_tensor_get_dims(b1)[1], sc_tensor_id(b1));
    printf("  W2 [%u×%u]  grad=ON   id=%u\n",
           sc_tensor_get_dims(W2)[0], sc_tensor_get_dims(W2)[1], sc_tensor_id(W2));
    printf("  b2 [%u×%u]  grad=ON   id=%u\n",
           sc_tensor_get_dims(b2)[0], sc_tensor_get_dims(b2)[1], sc_tensor_id(b2));

    /* ─────────────────────────────────────────────
     *  4 )  DEFINE THE COMPUTATION GRAPH (LAZY)
     *
     *  These calls do NOT compute anything yet —
     *  they construct op-nodes that record the
     *  operation and its operand pointers.
     *
     *  The graph is:
     *    H      = relu( X·W1 + b1 )
     *    Y_pred = H·W2 + b2
     *    loss   = mean( (Y_pred − Y)² )
     * ───────────────────────────────────────────── */

    banner("4. Defining computation graph (lazy)");

    sc_tensor_t *H_pre  = sc_tensor_mul_naive(pool, X, W1);
    sc_tensor_t *H_bias = sc_tensor_add(pool, H_pre, b1);
    sc_tensor_t *H      = sc_tensor_relu(pool, H_bias);

    sc_tensor_t *O_pre  = sc_tensor_mul_naive(pool, H, W2);
    sc_tensor_t *Y_pred = sc_tensor_add(pool, O_pre, b2);

    sc_tensor_t *diff   = sc_tensor_sub(pool, Y_pred, Y);
    sc_tensor_t *sq     = sc_tensor_square(pool, diff);
    sc_tensor_t *loss   = sc_tensor_mean(pool, sq);

    printf("  %d op-nodes defined (X→…→loss)\n", 8);
    printf("  No computation has happened yet.\n");

    /* ─────────────────────────────────────────────
     *  5 )  BUILD THE GRAPH  (Layer-2 one-liner)
     *
     *  sc_build_graph does everything at once:
     *    1. verifyIfDAG  → topological sort
     *    2. assignBackendGraph  → CPU / GPU / HYBRID dispatch
     *    3. assignGradMemory  → allocate gradient buffers
     *
     *  Returns a ready sc_graph_t* handle.
     * ───────────────────────────────────────────── */

    banner("5. Building execution graph");

    sc_graph_t *graph = sc_build_graph(
        pool_meta,
        pool_gpu,
        pool_grad_cpu,
        pool_grad_gpu,
        loss,
        SC_BACKEND_GPU          /* use SC_BACKEND_GPU or SC_BACKEND_HYBRID if CUDA is available */
    );
    assert(graph != NULL);

    printf("  Graph nodes : %zu\n", sc_graph_size(graph));
    printf("  Backend     : CPU\n");
    printf("  Pool usage  : data  %zu / %zu bytes\n",
           sc_pool_used(pool), sc_pool_size(pool));
    printf("                meta  %zu / %zu bytes\n",
           sc_pool_used(pool_meta), sc_pool_size(pool_meta));

    /* ─────────────────────────────────────────────
     *  6 )  TRAINING LOOP
     *
     *  Option A — manual loop (Layer-1 calls):
     *      sc_graph_forward(...)
     *      sc_autograd_gpu_transfer(...)
     *      sc_grad_initializer(...)
     *      sc_backward(...)
     *      sc_sgd(...)
     *
     *  Option B — one-liner (Layer-2):
     *      sc_graph_step(pool, pool_gpu, graph, lr)
     *
     *  We'll use Option B for brevity.
     * ───────────────────────────────────────────── */

    banner("6. Training (10000 epochs, lr=0.05)");

    int   epochs = 10000;
    float lr     = 0.05f;

    for (int i = 0; i <= epochs; i++) {

        /* ── single training step ────────────── */
        sc_graph_step(pool, pool_gpu, graph, lr);

        /* ── log every 1000 epochs ───────────── */
        if (i % 1000 == 0) {
            float l = sc_graph_get_loss(graph);
            printf("  epoch %5d   loss = %.8f\n", i, l);
        }
    }

    /* ─────────────────────────────────────────────
     *  7 )  INFERENCE  — run one more forward pass
     *       and read back the predictions
     * ───────────────────────────────────────────── */

    banner("7. Final predictions");

    sc_graph_forward(pool, pool_gpu, graph);
    // Was cause of error
    sc_autograd_gpu_transfer(graph);  

    float *inputs      = (float *)sc_tensor_get_data(X);
    float *targets     = (float *)sc_tensor_get_data(Y);
    float *predictions = (float *)sc_tensor_get_data(Y_pred);

    separator();
    printf("   X1    X2    │  Target  │  Predicted\n");
    separator();

    for (int i = 0; i < 4; i++) {
        float x1 = inputs[i * 2 + 0];
        float x2 = inputs[i * 2 + 1];

        printf("  %4.1f  %4.1f   │   %4.1f   │   %7.4f\n",
               x1, x2, targets[i], predictions[i]);
    }

    separator();
    printf("  Final loss : %.8f\n", sc_graph_get_loss(graph));

    /* ─────────────────────────────────────────────
     *  8 )  SAVE & LOAD  (round-trip demo)
     *
     *  sc_save_model / sc_load_model persist raw
     *  float data — no shape metadata is stored,
     *  so you must recreate tensors of the correct
     *  shapes before loading.
     * ───────────────────────────────────────────── */

    banner("8. Save / Load model weights");

    sc_tensor_t *weights[] = { W1, b1, W2, b2 };
    int saved = sc_save_model("xor_weights.bin", weights, 4);
    printf("  save_model → %s\n", saved ? "OK" : "FAILED");

    /* Zero out W1 to prove load really works */
    sc_tensor_fill_random_normal(W1, 0.0f, 0.0001f);

    int loaded = sc_load_model("xor_weights.bin", weights, 4);
    printf("  load_model → %s\n", loaded ? "OK" : "FAILED");

    /* Forward pass after reloading to confirm predictions match */
    sc_graph_forward(pool, pool_gpu, graph);
    sc_autograd_gpu_transfer(graph);

    printf("  Post-reload loss : %.8f\n", sc_graph_get_loss(graph));

    /* ─────────────────────────────────────────────
     *  9 )  CLEANUP
     *
     *  Always destroy pools when done.
     *  sc_graph_destroy frees the graph handle
     *  but NOT the pools themselves.
     * ───────────────────────────────────────────── */

    banner("9. Cleanup");

    sc_graph_destroy(graph);

    sc_pool_destroy(pool);
    sc_pool_destroy(pool_meta);
    sc_pool_destroy(pool_grad_cpu);
    sc_pool_destroy(pool_grad_gpu);
    sc_pool_destroy(pool_gpu);

    printf("  All resources freed.\n\n");

    return 0;
}
