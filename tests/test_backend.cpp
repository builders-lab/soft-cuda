#include <gtest/gtest.h>
#include <vector>
#include "soft-cuda/tensor/api.h"
#include "graph/DAGbuild.h"

// ============================================================
//  HELPERS
// ============================================================

static tensor_pool_t *make_pool(size_t size = 1024 * 256) {
    return tensor_pool_create(size);
}

static tensor_t *make_matrix(tensor_pool_t *pool, uint32_t rows, uint32_t cols, float *data) {
    uint32_t dims[3] = {rows, cols, 0};
    return tensor_create(pool, tensor_dtype_t::FLOAT32_T, 2, dims, data, true);
}

// Build DAG aur CPU backend assign karo
static std::vector<execution_node_t *> build_cpu_graph(tensor_pool_t *pool,
                                                        tensor_pool_t *pool_gpu,
                                                        tensor_t *output) {
    std::vector<execution_node_t *> nodes;
    verifyIfDAG(pool, output, nodes);
    assignBackendGraph(pool_gpu, nodes, backend_mode::CPU);
    return nodes;
}

// ============================================================
//  CPU BACKEND ASSIGNMENT TESTS
// ============================================================

// CPU mode mein saare nodes CPU backend pe hone chahiye
TEST(BackendAssignTest, AllNodesCPUInCPUMode) {
    tensor_pool_t *pool     = make_pool();
    tensor_pool_t *pool_gpu = make_pool();
    ASSERT_NE(pool, nullptr);
    ASSERT_NE(pool_gpu, nullptr);

    float a_data[] = {1, 2, 3, 4};
    float b_data[] = {5, 6, 7, 8};
    tensor_t *a = make_matrix(pool, 2, 2, a_data);
    tensor_t *b = make_matrix(pool, 2, 2, b_data);
    tensor_t *result = tensor_add(pool, a, b);
    ASSERT_NE(result, nullptr);

    auto nodes = build_cpu_graph(pool, pool_gpu, result);

    // CPU mode mein koi bhi node GPU pe nahi hona chahiye
    for (auto node : nodes) {
        EXPECT_EQ(node->backend_fn, (decltype(node->backend_fn))tensor_evaluate)
            << "Node pos=" << node->pos << " CPU pe nahi hai";
        EXPECT_EQ(node->device_ptr, nullptr)
            << "CPU node ka device_ptr NULL hona chahiye";
    }

    tensor_pool_destroy(pool);
    tensor_pool_destroy(pool_gpu);
}

// CPU mode mein parent_pos sahi set hona chahiye
TEST(BackendAssignTest, ParentPosSetCorrectly) {
    tensor_pool_t *pool     = make_pool();
    tensor_pool_t *pool_gpu = make_pool();
    ASSERT_NE(pool, nullptr);
    ASSERT_NE(pool_gpu, nullptr);

    float a_data[] = {1, 2, 3, 4};
    float b_data[] = {5, 6, 7, 8};
    tensor_t *a = make_matrix(pool, 2, 2, a_data);
    tensor_t *b = make_matrix(pool, 2, 2, b_data);
    tensor_t *result = tensor_add(pool, a, b);
    ASSERT_NE(result, nullptr);

    auto nodes = build_cpu_graph(pool, pool_gpu, result);
    ASSERT_EQ(nodes.size(), 3u);

    // Last node (result) ke parents A aur B hone chahiye
    execution_node_t *result_node = nodes.back();
    EXPECT_NE(result_node->parent_pos[0], -1);
    EXPECT_NE(result_node->parent_pos[1], -1);

    // Leaf nodes ke parents -1 hone chahiye
    execution_node_t *leaf_a = nodes[0];
    EXPECT_EQ(leaf_a->parent_pos[0], -1);
    EXPECT_EQ(leaf_a->parent_pos[1], -1);

    tensor_pool_destroy(pool);
    tensor_pool_destroy(pool_gpu);
}

// Chained ops mein bhi CPU assignment sahi hona chahiye
// A → transpose → matmul → relu
TEST(BackendAssignTest, ChainedOpsCPUAssignment) {
    tensor_pool_t *pool     = make_pool();
    tensor_pool_t *pool_gpu = make_pool();
    ASSERT_NE(pool, nullptr);
    ASSERT_NE(pool_gpu, nullptr);

    float x_data[] = {1, 2, 3, 4};
    float w_data[] = {1, 0, 0, 1};
    tensor_t *x  = make_matrix(pool, 2, 2, x_data);
    tensor_t *w  = make_matrix(pool, 2, 2, w_data);
    tensor_t *wt = tensor_transpose(pool, w);
    ASSERT_NE(wt, nullptr);

    tensor_t *mm  = tensor_mul(pool, x, wt);
    ASSERT_NE(mm, nullptr);
    tensor_t *out = tensor_relu(pool, mm);
    ASSERT_NE(out, nullptr);

    auto nodes = build_cpu_graph(pool, pool_gpu, out);

    // Saare nodes CPU pe
    for (auto node : nodes) {
        EXPECT_EQ(node->backend_fn, (decltype(node->backend_fn))tensor_evaluate)
            << "Node pos=" << node->pos << " CPU pe nahi";
    }

    // Correct node count — x, w, wt, mm, out = 5
    EXPECT_EQ(nodes.size(), 5u);

    tensor_pool_destroy(pool);
    tensor_pool_destroy(pool_gpu);
}

// ============================================================
//  GRAD MEMORY ASSIGNMENT TESTS
// ============================================================

// assignGradMemory ke baad har grad_compute=true node ka grad != NULL hona chahiye
TEST(BackendAssignTest, GradMemoryAssigned) {
    tensor_pool_t *pool          = make_pool();
    tensor_pool_t *pool_gpu      = make_pool();
    tensor_pool_t *pool_grad_cpu = make_pool();
    tensor_pool_t *pool_grad_gpu = make_pool();
    ASSERT_NE(pool, nullptr);

    float a_data[] = {1, 2, 3, 4};
    float b_data[] = {5, 6, 7, 8};
    tensor_t *a = make_matrix(pool, 2, 2, a_data);
    tensor_t *b = make_matrix(pool, 2, 2, b_data);
    tensor_t *result = tensor_add(pool, a, b);
    ASSERT_NE(result, nullptr);

    std::vector<execution_node_t *> nodes;
    verifyIfDAG(pool, result, nodes);
    assignBackendGraph(pool_gpu, nodes, backend_mode::CPU);
    assignGradMemory(pool_grad_cpu, pool_grad_gpu, nodes);

    // grad_compute=true wale nodes ka grad NULL nahi hona chahiye
    for (auto node : nodes) {
        if (node->t->grad_compute) {
            EXPECT_NE(node->t->grad, nullptr)
                << "Node pos=" << node->pos << " ka grad NULL hai";
        }
    }

    tensor_pool_destroy(pool);
    tensor_pool_destroy(pool_gpu);
    tensor_pool_destroy(pool_grad_cpu);
    tensor_pool_destroy(pool_grad_gpu);
}

// CPU mode mein device_ptr_grad NULL hona chahiye
TEST(BackendAssignTest, CPUNodeGradDevicePtrIsNull) {
    tensor_pool_t *pool          = make_pool();
    tensor_pool_t *pool_gpu      = make_pool();
    tensor_pool_t *pool_grad_cpu = make_pool();
    tensor_pool_t *pool_grad_gpu = make_pool();
    ASSERT_NE(pool, nullptr);

    float a_data[] = {1, 2, 3, 4};
    float b_data[] = {5, 6, 7, 8};
    tensor_t *a = make_matrix(pool, 2, 2, a_data);
    tensor_t *b = make_matrix(pool, 2, 2, b_data);
    tensor_t *result = tensor_add(pool, a, b);
    ASSERT_NE(result, nullptr);

    std::vector<execution_node_t *> nodes;
    verifyIfDAG(pool, result, nodes);
    assignBackendGraph(pool_gpu, nodes, backend_mode::CPU);
    assignGradMemory(pool_grad_cpu, pool_grad_gpu, nodes);

    // CPU nodes ka device_ptr_grad NULL hona chahiye
    for (auto node : nodes) {
        EXPECT_EQ(node->device_ptr_grad, nullptr)
            << "CPU node pos=" << node->pos << " ka device_ptr_grad NULL nahi";
    }

    tensor_pool_destroy(pool);
    tensor_pool_destroy(pool_gpu);
    tensor_pool_destroy(pool_grad_cpu);
    tensor_pool_destroy(pool_grad_gpu);
}

// ============================================================
//  GPU MODE TESTS — sirf GPU available hone par chalenge
// ============================================================

#ifdef HAS_CUDA
TEST(BackendAssignTest, GPUModeAssignsGPUBackend) {
    tensor_pool_t *pool     = make_pool();
    tensor_pool_t *pool_gpu = tensor_pool_create(1024 * 256, true);  // device pool
    ASSERT_NE(pool, nullptr);
    ASSERT_NE(pool_gpu, nullptr);

    float a_data[] = {1, 2, 3, 4};
    float b_data[] = {5, 6, 7, 8};
    tensor_t *a = make_matrix(pool, 2, 2, a_data);
    tensor_t *b = make_matrix(pool, 2, 2, b_data);
    tensor_t *result = tensor_add(pool, a, b);
    ASSERT_NE(result, nullptr);

    std::vector<execution_node_t *> nodes;
    verifyIfDAG(pool, result, nodes);
    assignBackendGraph(pool_gpu, nodes, backend_mode::GPU);

    // GPU mode mein op wale nodes GPU backend pe hone chahiye
    for (auto node : nodes) {
        if (node->t->op != tensor_op_t::NONE) {
            EXPECT_EQ(node->backend_fn, (decltype(node->backend_fn))tensor_evaluate_GPU)
                << "Node pos=" << node->pos << " GPU pe nahi";
        }
    }

    tensor_pool_destroy(pool);
    tensor_pool_destroy(pool_gpu);
}
#endif  // HAS_CUDA
