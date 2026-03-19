#include <gtest/gtest.h>
#include "soft-cuda/tensor/api.h"

// ============================================================
//  POOL TESTS
//  pool.cpp ka har function yahan cover hai
// ============================================================

TEST(PoolTest, CreateAndSize) {
    tensor_pool_t *pool = tensor_pool_create(1024);
    ASSERT_NE(pool, nullptr);
    EXPECT_EQ(tensor_pool_size(pool), 1024);
    EXPECT_EQ(tensor_pool_used(pool), 0);
    tensor_pool_destroy(pool);
}

TEST(PoolTest, CreateZeroSizeDies) {
    EXPECT_DEATH(tensor_pool_create(0), "");
}

TEST(PoolTest, AllocIncreasesUsed) {
    tensor_pool_t *pool = tensor_pool_create(1024);
    ASSERT_NE(pool, nullptr);

    size_t before = tensor_pool_used(pool);
    void *ptr = tensor_pool_alloc(pool, 64, nullptr);

    EXPECT_NE(ptr, nullptr);
    EXPECT_GE(tensor_pool_used(pool), before + 64);
    tensor_pool_destroy(pool);
}

TEST(PoolTest, AllocExhaustReturnsNull) {
    tensor_pool_t *pool = tensor_pool_create(64);
    ASSERT_NE(pool, nullptr);

    void *ptr1 = tensor_pool_alloc(pool, 64, nullptr);
    EXPECT_NE(ptr1, nullptr);

    void *ptr2 = tensor_pool_alloc(pool, 1, nullptr);
    EXPECT_EQ(ptr2, nullptr);

    tensor_pool_destroy(pool);
}

TEST(PoolTest, AllocIdIncrements) {
    tensor_pool_t *pool = tensor_pool_create(1024);
    ASSERT_NE(pool, nullptr);

    uint32_t id1 = 0, id2 = 0;
    tensor_pool_alloc(pool, 8, &id1);
    tensor_pool_alloc(pool, 8, &id2);

    EXPECT_EQ(id1, 1);
    EXPECT_EQ(id2, 2);
    tensor_pool_destroy(pool);
}

TEST(PoolTest, ZeroResetsPool) {
    tensor_pool_t *pool = tensor_pool_create(1024);
    ASSERT_NE(pool, nullptr);

    tensor_pool_alloc(pool, 128, nullptr);
    EXPECT_GT(tensor_pool_used(pool), 0);

    tensor_pool_zero(pool);
    EXPECT_EQ(tensor_pool_used(pool), 0);

    void *ptr = tensor_pool_alloc(pool, 128, nullptr);
    EXPECT_NE(ptr, nullptr);

    tensor_pool_destroy(pool);
}

TEST(PoolTest, DestroyNullIsSafe) {
    // This tells GTest: "I expect this function call to crash the program."
    // If it crashes, the test PASSES. If it survives, the test FAILS.
    EXPECT_DEATH(tensor_pool_destroy(nullptr), ".*");
}

// ============================================================
//  TENSOR CREATE TESTS
//  tensor.cpp → tensor_create / tensor_dtype_create
// ============================================================

TEST(TensorCreateTest, Float32TensorCreates) {
    tensor_pool_t *pool = tensor_pool_create(4096);
    ASSERT_NE(pool, nullptr);

    uint32_t dims[] = {3, 3, 0}; 
    tensor_t *t = tensor_create(pool, tensor_dtype_t::FLOAT32_T, 2, dims, nullptr);

    ASSERT_NE(t, nullptr);
    EXPECT_GT(tensor_id(t), 0u); 

    tensor_pool_destroy(pool);
}

TEST(TensorCreateTest, NullElemsZeroInitializes) {
    tensor_pool_t *pool = tensor_pool_create(4096);
    ASSERT_NE(pool, nullptr);

    uint32_t dims[] = {4, 0}; 
    tensor_t *t = tensor_create(pool, tensor_dtype_t::FLOAT32_T, 1, dims, nullptr);
    ASSERT_NE(t, nullptr);

    float *data = (float *)tensor_get_data(t);
    for (int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(data[i], 0.0f);
    }

    tensor_pool_destroy(pool);
}

TEST(TensorCreateTest, ElemsCopiedCorrectly) {
    tensor_pool_t *pool = tensor_pool_create(4096);
    ASSERT_NE(pool, nullptr);

    float input[] = {1.0f, 2.0f, 3.0f, 4.0f};
    uint32_t dims[] = {4, 0};
    tensor_t *t = tensor_create(pool, tensor_dtype_t::FLOAT32_T, 1, dims, input);
    ASSERT_NE(t, nullptr);

    float *data = (float *)tensor_get_data(t);
    EXPECT_FLOAT_EQ(data[0], 1.0f);
    EXPECT_FLOAT_EQ(data[1], 2.0f);
    EXPECT_FLOAT_EQ(data[2], 3.0f);
    EXPECT_FLOAT_EQ(data[3], 4.0f);

    tensor_pool_destroy(pool);
}

TEST(TensorCreateTest, Int32TensorCreates) {
    tensor_pool_t *pool = tensor_pool_create(4096);
    ASSERT_NE(pool, nullptr);

    int32_t input[] = {10, 20, 30};
    uint32_t dims[] = {3, 0};
    tensor_t *t = tensor_create(pool, tensor_dtype_t::INT32_T, 1, dims, input);
    ASSERT_NE(t, nullptr);

    int32_t *data = (int32_t *)tensor_get_data(t);
    EXPECT_EQ(data[0], 10);
    EXPECT_EQ(data[1], 20);
    EXPECT_EQ(data[2], 30);

    tensor_pool_destroy(pool);
}

TEST(TensorCreateTest, TensorIdsAreUnique) {
    tensor_pool_t *pool = tensor_pool_create(4096);
    ASSERT_NE(pool, nullptr);

    uint32_t dims[] = {2, 0};
    tensor_t *t1 = tensor_create(pool, tensor_dtype_t::FLOAT32_T, 1, dims, nullptr);
    tensor_t *t2 = tensor_create(pool, tensor_dtype_t::FLOAT32_T, 1, dims, nullptr);
    
    ASSERT_NE(t1, nullptr);
    ASSERT_NE(t2, nullptr);
    EXPECT_NE(tensor_id(t1), tensor_id(t2));

    tensor_pool_destroy(pool);
}

TEST(TensorCreateTest, ReturnsNullWhenPoolExhausted) {
    tensor_pool_t *pool = tensor_pool_create(8); 
    ASSERT_NE(pool, nullptr);

    uint32_t dims[] = {100, 0}; 
    tensor_t *t = tensor_create(pool, tensor_dtype_t::FLOAT32_T, 1, dims, nullptr);
    
    EXPECT_EQ(t, nullptr);
    tensor_pool_destroy(pool);
}

TEST(TensorCreateTest, NullPoolDies) {
    uint32_t dims[] = {2, 0};
    EXPECT_DEATH(tensor_create(nullptr, tensor_dtype_t::FLOAT32_T, 1, dims, nullptr), "");
}
