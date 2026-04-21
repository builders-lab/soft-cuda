#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include "soft-cuda/tensor/api.h"
#include "graph/DAGbuild.h"

// ============================================================
//  HELPERS
// ============================================================

static tensor_pool_t *make_pool(size_t size = 1024 * 512) {
    return tensor_pool_create(size);
}

static tensor_t *make_matrix(tensor_pool_t *pool, uint32_t rows, uint32_t cols, float *data,
                              bool grad = true) {
    uint32_t dims[3] = {rows, cols, 0};
    return tensor_create(pool, tensor_dtype_t::FLOAT32_T, 2, dims, data, grad);
}

static tensor_t *make_1d(tensor_pool_t *pool, uint32_t n, float *data, bool grad = true) {
    uint32_t dims[2] = {n, 0};
    return tensor_create(pool, tensor_dtype_t::FLOAT32_T, 1, dims, data, grad);
}

// Forward pass + backward pass ek saath karo
// Returns true agar dono succeed kare
static bool run_forward_backward(tensor_pool_t *pool, tensor_t *output,
                                  std::vector<execution_node_t *> &nodes) {
    tensor_pool_t *pool_gpu = tensor_pool_create(1024);
    if (!verifyIfDAG(pool, output, nodes)) return false;
    assignBackendGraph(pool_gpu, nodes, backend_mode::CPU);
    assignGradMemory(pool, pool_gpu, nodes);
    if (!tensor_graph_forward_evaluate(pool, pool_gpu, nodes)) return false;
    gradInitializer(nodes);
    bool ok = tensor_graph_backward(nodes);
    tensor_pool_destroy(pool_gpu);
    return ok;
}

// ============================================================
//  ADD BACKWARD TESTS
//  dL/dA = dL/dOut, dL/dB = dL/dOut (gradient flows equally)
// ============================================================

// [1 2] + [3 4] — gradient of A and B both == 1 (since dOut/dA = 1)
TEST(BackpropTest, AddGradientFlowsToInputs) {
    tensor_pool_t *pool = make_pool();
    ASSERT_NE(pool, nullptr);

    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[] = {3.0f, 4.0f, 5.0f, 6.0f};
    tensor_t *a = make_matrix(pool, 2, 2, a_data);
    tensor_t *b = make_matrix(pool, 2, 2, b_data);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);

    tensor_t *result = tensor_add(pool, a, b);
    ASSERT_NE(result, nullptr);

    std::vector<execution_node_t *> nodes;
    ASSERT_TRUE(run_forward_backward(pool, result, nodes));

    // dL/dA == dL/dOut == 1 (gradInitializer sets root grad to 1)
    float *g_a = (float *)tensor_get_data(a->grad);
    float *g_b = (float *)tensor_get_data(b->grad);
    for (int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(g_a[i], 1.0f) << "g_a mismatch at " << i;
        EXPECT_FLOAT_EQ(g_b[i], 1.0f) << "g_b mismatch at " << i;
    }

    tensor_pool_destroy(pool);
}

// ============================================================
//  SUB BACKWARD TESTS
//  dL/dA = +dL/dOut, dL/dB = -dL/dOut
// ============================================================

TEST(BackpropTest, SubGradientSigns) {
    tensor_pool_t *pool = make_pool();
    ASSERT_NE(pool, nullptr);

    float a_data[] = {5.0f, 6.0f, 7.0f, 8.0f};
    float b_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    tensor_t *a = make_matrix(pool, 2, 2, a_data);
    tensor_t *b = make_matrix(pool, 2, 2, b_data);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);

    tensor_t *result = tensor_sub(pool, a, b);
    ASSERT_NE(result, nullptr);

    std::vector<execution_node_t *> nodes;
    ASSERT_TRUE(run_forward_backward(pool, result, nodes));

    float *g_a = (float *)tensor_get_data(a->grad);
    float *g_b = (float *)tensor_get_data(b->grad);

    for (int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(g_a[i],  1.0f) << "g_a[" << i << "] sahi nahi";
        EXPECT_FLOAT_EQ(g_b[i], -1.0f) << "g_b[" << i << "] sahi nahi";
    }

    tensor_pool_destroy(pool);
}

// ============================================================
//  RELU BACKWARD TESTS
//  dL/dA = dL/dOut if A > 0, else 0
// ============================================================

TEST(BackpropTest, ReluGradientMasked) {
    tensor_pool_t *pool = make_pool();
    ASSERT_NE(pool, nullptr);

    // positive → grad pass hoga, negative → grad 0 hoga
    float data[] = {2.0f, -1.0f, 3.0f, -4.0f};
    tensor_t *a = make_matrix(pool, 2, 2, data);
    ASSERT_NE(a, nullptr);

    tensor_t *result = tensor_relu(pool, a);
    ASSERT_NE(result, nullptr);

    std::vector<execution_node_t *> nodes;
    ASSERT_TRUE(run_forward_backward(pool, result, nodes));

    float *g_a = (float *)tensor_get_data(a->grad);
    EXPECT_FLOAT_EQ(g_a[0], 1.0f);  //  2 > 0 → grad = 1
    EXPECT_FLOAT_EQ(g_a[1], 0.0f);  // -1 < 0 → grad = 0
    EXPECT_FLOAT_EQ(g_a[2], 1.0f);  //  3 > 0 → grad = 1
    EXPECT_FLOAT_EQ(g_a[3], 0.0f);  // -4 < 0 → grad = 0

    tensor_pool_destroy(pool);
}

// ============================================================
//  SQUARE BACKWARD TESTS
//  dL/dX = 2 * X * dL/dOut
// ============================================================

// d(x^2)/dx = 2x
// input = [1 2 3] → grad = [2 4 6]
TEST(BackpropTest, SquareGradientIsDoubleInput) {
    tensor_pool_t *pool = make_pool();
    ASSERT_NE(pool, nullptr);

    float data[] = {1.0f, 2.0f, 3.0f};
    tensor_t *a = make_1d(pool, 3, data);
    ASSERT_NE(a, nullptr);

    tensor_t *result = tensor_square(pool, a);
    ASSERT_NE(result, nullptr);

    std::vector<execution_node_t *> nodes;
    ASSERT_TRUE(run_forward_backward(pool, result, nodes));

    float *g_a = (float *)tensor_get_data(a->grad);
    // grad = 2 * input * upstream(1)
    EXPECT_NEAR(g_a[0], 2.0f, 1e-5f);
    EXPECT_NEAR(g_a[1], 4.0f, 1e-5f);
    EXPECT_NEAR(g_a[2], 6.0f, 1e-5f);

    tensor_pool_destroy(pool);
}

// ============================================================
//  NAIVE MATMUL BACKWARD TESTS
//  dL/dA = dL/dOut * B^T
//  dL/dB = A^T * dL/dOut
// ============================================================

// A = [[1,2],[3,4]], B = [[5,6],[7,8]]
// Out = A*B = [[19,22],[43,50]]
// dL/dA = dL/dOut * B^T, dL/dB = A^T * dL/dOut
// With upstream grad = ones:
// dL/dA = [[1,1],[1,1]] * [[5,7],[6,8]] = [[11,15],[11,15]]
// dL/dB = [[1,3],[2,4]] * [[1,1],[1,1]] = [[4,4],[6,6]]
TEST(BackpropTest, NaiveMatMulGradients) {
    tensor_pool_t *pool = make_pool();
    ASSERT_NE(pool, nullptr);

    float a_data[] = {1, 2, 3, 4};
    float b_data[] = {5, 6, 7, 8};
    tensor_t *a = make_matrix(pool, 2, 2, a_data);
    tensor_t *b = make_matrix(pool, 2, 2, b_data);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);

    tensor_t *result = tensor_mul_naive(pool, a, b);
    ASSERT_NE(result, nullptr);

    std::vector<execution_node_t *> nodes;
    ASSERT_TRUE(run_forward_backward(pool, result, nodes));

    float *g_a = (float *)tensor_get_data(a->grad);
    float *g_b = (float *)tensor_get_data(b->grad);

    // dL/dA
    EXPECT_NEAR(g_a[0], 11.0f, 1e-4f);
    EXPECT_NEAR(g_a[1], 15.0f, 1e-4f);
    EXPECT_NEAR(g_a[2], 11.0f, 1e-4f);
    EXPECT_NEAR(g_a[3], 15.0f, 1e-4f);

    // dL/dB
    EXPECT_NEAR(g_b[0], 4.0f, 1e-4f);
    EXPECT_NEAR(g_b[1], 4.0f, 1e-4f);
    EXPECT_NEAR(g_b[2], 6.0f, 1e-4f);
    EXPECT_NEAR(g_b[3], 6.0f, 1e-4f);

    tensor_pool_destroy(pool);
}

// ============================================================
//  CHAINED BACKPROP — MatMul → ReLU
//  Gradient pehle ReLU se pass hoga, phir MatMul tak
// ============================================================

TEST(BackpropTest, ChainedMatMulReluGradient) {
    tensor_pool_t *pool = make_pool();
    ASSERT_NE(pool, nullptr);

    // Identity matrix taaki output = input
    float x_data[] = {2.0f, -1.0f, 3.0f, -2.0f};
    float w_data[] = {1.0f,  0.0f, 0.0f,  1.0f};
    tensor_t *x = make_matrix(pool, 2, 2, x_data);
    tensor_t *w = make_matrix(pool, 2, 2, w_data);
    ASSERT_NE(x, nullptr);
    ASSERT_NE(w, nullptr);

    tensor_t *matmul = tensor_mul_naive(pool, x, w);
    ASSERT_NE(matmul, nullptr);

    tensor_t *result = tensor_relu(pool, matmul);
    ASSERT_NE(result, nullptr);

    std::vector<execution_node_t *> nodes;
    ASSERT_TRUE(run_forward_backward(pool, result, nodes));

    // x = [2, -1, 3, -2]
    // matmul = x*I = [2, -1, 3, -2]  (identity matrix)
    // relu = [2, 0, 3, 0]
    // relu grad mask = [1, 0, 1, 0]
    // grad flows only where relu was active
    float *g_x = (float *)tensor_get_data(x->grad);
    EXPECT_NEAR(g_x[0], 1.0f, 1e-4f);  // 2 > 0 → grad flows
    EXPECT_NEAR(g_x[1], 0.0f, 1e-4f);  // -1 < 0 → blocked
    EXPECT_NEAR(g_x[2], 1.0f, 1e-4f);  // 3 > 0 → grad flows
    EXPECT_NEAR(g_x[3], 0.0f, 1e-4f);  // -2 < 0 → blocked

    tensor_pool_destroy(pool);
}
