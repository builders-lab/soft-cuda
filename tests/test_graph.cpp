#include <gtest/gtest.h>
#include <vector>
#include "soft-cuda/tensor/api.h"

// ============================================================
//  HELPERS
// ============================================================

static tensor_pool_t *make_pool(size_t size = 1024 * 256) {
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

// ============================================================
//  DAG BUILD TESTS  (verifyIfDAG)
// ============================================================

// Single tensor — NONE op — DAG banana chahiye, 1 node hona chahiye
TEST(DAGTest, SingleTensorBuilds) {
    tensor_pool_t *pool = make_pool();
    ASSERT_NE(pool, nullptr);

    float data[] = {1, 2, 3, 4};
    tensor_t *a = make_matrix(pool, 2, 2, data);
    ASSERT_NE(a, nullptr);

    std::vector<execution_node_t *> nodes;
    bool ok = verifyIfDAG(pool, a, nodes);

    EXPECT_TRUE(ok);
    EXPECT_EQ(nodes.size(), 1u);

    tensor_pool_destroy(pool);
}

// A + B — 3 nodes hone chahiye: A, B, result
TEST(DAGTest, AddOpBuildsThreeNodes) {
    tensor_pool_t *pool = make_pool();
    ASSERT_NE(pool, nullptr);

    float a_data[] = {1, 2, 3, 4};
    float b_data[] = {5, 6, 7, 8};
    tensor_t *a = make_matrix(pool, 2, 2, a_data);
    tensor_t *b = make_matrix(pool, 2, 2, b_data);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);

    tensor_t *result = tensor_add(pool, a, b);
    ASSERT_NE(result, nullptr);

    std::vector<execution_node_t *> nodes;
    bool ok = verifyIfDAG(pool, result, nodes);

    EXPECT_TRUE(ok);
    EXPECT_EQ(nodes.size(), 3u);  // a, b, result

    tensor_pool_destroy(pool);
}

// Topological order — inputs pehle, output baad mein hona chahiye
TEST(DAGTest, TopologicalOrderCorrect) {
    tensor_pool_t *pool = make_pool();
    ASSERT_NE(pool, nullptr);

    float a_data[] = {1, 2, 3, 4};
    float b_data[] = {1, 0, 0, 1};
    tensor_t *a = make_matrix(pool, 2, 2, a_data);
    tensor_t *b = make_matrix(pool, 2, 2, b_data);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);

    tensor_t *result = tensor_add(pool, a, b);
    ASSERT_NE(result, nullptr);

    std::vector<execution_node_t *> nodes;
    ASSERT_TRUE(verifyIfDAG(pool, result, nodes));

    // Result node last mein hona chahiye
    EXPECT_EQ(nodes.back()->t, result);

    // Inputs result se pehle hone chahiye
    int32_t result_pos = getPosOfNode(nodes.back());
    for (size_t i = 0; i < nodes.size() - 1; i++) {
        EXPECT_LT(getPosOfNode(nodes[i]), result_pos);
    }

    tensor_pool_destroy(pool);
}

// Shared input — A use kiya dono jagah — duplicate nodes nahi hone chahiye
// C = A + A — sirf 2 nodes: A, C
TEST(DAGTest, SharedInputNoDuplicateNodes) {
    tensor_pool_t *pool = make_pool();
    ASSERT_NE(pool, nullptr);

    float data[] = {1, 2, 3, 4};
    tensor_t *a = make_matrix(pool, 2, 2, data);
    ASSERT_NE(a, nullptr);

    tensor_t *result = tensor_add(pool, a, a);  // same tensor dono jagah
    ASSERT_NE(result, nullptr);

    std::vector<execution_node_t *> nodes;
    bool ok = verifyIfDAG(pool, result, nodes);

    EXPECT_TRUE(ok);
    EXPECT_EQ(nodes.size(), 2u);  // sirf A aur result — duplicate nahi

    tensor_pool_destroy(pool);
}

// ============================================================
//  FORWARD EVALUATE TESTS  (tensor_graph_forward_evaluate)
// ============================================================

// Simple graph — A + B forward evaluate karo
// [1 2] + [5 6] = [6  8 ]
// [3 4]   [7 8]   [10 12]
TEST(GraphForwardTest, AddForwardCorrect) {
    tensor_pool_t *pool     = make_pool();
    tensor_pool_t *pool_gpu = make_pool();  // CPU mode mein GPU pool khali rahega
    ASSERT_NE(pool, nullptr);
    ASSERT_NE(pool_gpu, nullptr);

    float a_data[] = {1, 2, 3, 4};
    float b_data[] = {5, 6, 7, 8};
    tensor_t *a = make_matrix(pool, 2, 2, a_data);
    tensor_t *b = make_matrix(pool, 2, 2, b_data);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);

    tensor_t *result = tensor_add(pool, a, b);
    ASSERT_NE(result, nullptr);

    std::vector<execution_node_t *> nodes;
    ASSERT_TRUE(verifyIfDAG(pool, result, nodes));

    assignBackendGraph(pool_gpu, nodes, backend_mode::CPU);

    bool ok = tensor_graph_forward_evaluate(pool, pool_gpu, nodes);
    EXPECT_TRUE(ok);

    float *out = (float *)tensor_get_data(result);
    EXPECT_FLOAT_EQ(out[0], 6.0f);
    EXPECT_FLOAT_EQ(out[1], 8.0f);
    EXPECT_FLOAT_EQ(out[2], 10.0f);
    EXPECT_FLOAT_EQ(out[3], 12.0f);

    tensor_pool_destroy(pool);
    tensor_pool_destroy(pool_gpu);
}

// Chained ops — MatMul → ReLU
// W = [[1 0],[0 1]] (identity), X = [[2 -3],[4 -5]]
// MatMul = [[2 -3],[4 -5]], ReLU = [[2 0],[4 0]]
TEST(GraphForwardTest, MatMulThenReluCorrect) {
    tensor_pool_t *pool     = make_pool();
    tensor_pool_t *pool_gpu = make_pool();
    ASSERT_NE(pool, nullptr);
    ASSERT_NE(pool_gpu, nullptr);

    float x_data[] = {2, -3, 4, -5};
    float w_data[] = {1,  0, 0,  1};  // identity
    tensor_t *x = make_matrix(pool, 2, 2, x_data);
    tensor_t *w = make_matrix(pool, 2, 2, w_data);
    ASSERT_NE(x, nullptr);
    ASSERT_NE(w, nullptr);

    // W transpose karo — tensor_mul expects transposed B
    tensor_t *wt = tensor_transpose(pool, w);
    ASSERT_NE(wt, nullptr);

    tensor_t *matmul_result = tensor_mul(pool, x, wt);
    ASSERT_NE(matmul_result, nullptr);

    tensor_t *relu_result = tensor_relu(pool, matmul_result);
    ASSERT_NE(relu_result, nullptr);

    std::vector<execution_node_t *> nodes;
    ASSERT_TRUE(verifyIfDAG(pool, relu_result, nodes));

    assignBackendGraph(pool_gpu, nodes, backend_mode::CPU);

    bool ok = tensor_graph_forward_evaluate(pool, pool_gpu, nodes);
    EXPECT_TRUE(ok);

    float *out = (float *)tensor_get_data(relu_result);
    EXPECT_FLOAT_EQ(out[0], 2.0f);
    EXPECT_FLOAT_EQ(out[1], 0.0f);  // -3 relu → 0
    EXPECT_FLOAT_EQ(out[2], 4.0f);
    EXPECT_FLOAT_EQ(out[3], 0.0f);  // -5 relu → 0

    tensor_pool_destroy(pool);
    tensor_pool_destroy(pool_gpu);
}

// Graph dobara evaluate karne par same result aana chahiye
TEST(GraphForwardTest, EvaluateTwiceGivesSameResult) {
    tensor_pool_t *pool     = make_pool();
    tensor_pool_t *pool_gpu = make_pool();
    ASSERT_NE(pool, nullptr);
    ASSERT_NE(pool_gpu, nullptr);

    float a_data[] = {3, 1, 4, 1};
    float b_data[] = {5, 9, 2, 6};
    tensor_t *a = make_matrix(pool, 2, 2, a_data);
    tensor_t *b = make_matrix(pool, 2, 2, b_data);
    tensor_t *result = tensor_add(pool, a, b);
    ASSERT_NE(result, nullptr);

    std::vector<execution_node_t *> nodes;
    ASSERT_TRUE(verifyIfDAG(pool, result, nodes));
    assignBackendGraph(pool_gpu, nodes, backend_mode::CPU);

    // Pehli baar
    ASSERT_TRUE(tensor_graph_forward_evaluate(pool, pool_gpu, nodes));
    float out1[4];
    float *ptr = (float *)tensor_get_data(result);
    for (int i = 0; i < 4; i++) out1[i] = ptr[i];

    // Doosri baar
    ASSERT_TRUE(tensor_graph_forward_evaluate(pool, pool_gpu, nodes));
    float *ptr2 = (float *)tensor_get_data(result);
    for (int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(ptr2[i], out1[i]) << "Mismatch at index " << i;
    }

    tensor_pool_destroy(pool);
    tensor_pool_destroy(pool_gpu);
}
