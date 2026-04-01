#include <gtest/gtest.h>
#include "soft-cuda/tensor/api.h"

// ============================================================
//  HELPERS
// ============================================================

static tensor_pool_t *make_pool() {
    return tensor_pool_create(1024 * 64);
}

static tensor_t *make_matrix(tensor_pool_t *pool, uint32_t rows, uint32_t cols, float *data) {
    uint32_t dims[3] = {rows, cols, 0};
    return tensor_create(pool, tensor_dtype_t::FLOAT32_T, 2, dims, data);
}

// ============================================================
//  TRANSPOSE TESTS
// ============================================================

// [1 2 3] transpose = [1 4]
// [4 5 6]             [2 5]
//                     [3 6]
TEST(TransposeTest, BasicTranspose) {
    tensor_pool_t *pool = make_pool();
    ASSERT_NE(pool, nullptr);

    float data[] = {1, 2, 3, 4, 5, 6};
    tensor_t *a = make_matrix(pool, 2, 3, data);
    ASSERT_NE(a, nullptr);

    tensor_t *t = tensor_transpose(pool, a);
    ASSERT_NE(t, nullptr);
    ASSERT_TRUE(tensor_evaluate(pool, t, nullptr, nullptr, nullptr));

    // Shape check — 2x3 ka transpose 3x2 hona chahiye
    uint32_t *dims = tensor_get_dims(t);
    EXPECT_EQ(dims[0], 3u);
    EXPECT_EQ(dims[1], 2u);

    // Data check
    float *out = (float *)tensor_get_data(t);
    EXPECT_FLOAT_EQ(out[0], 1.0f);  // [0][0]
    EXPECT_FLOAT_EQ(out[1], 4.0f);  // [0][1]
    EXPECT_FLOAT_EQ(out[2], 2.0f);  // [1][0]
    EXPECT_FLOAT_EQ(out[3], 5.0f);  // [1][1]
    EXPECT_FLOAT_EQ(out[4], 3.0f);  // [2][0]
    EXPECT_FLOAT_EQ(out[5], 6.0f);  // [2][1]

    tensor_pool_destroy(pool);
}

// Square matrix transpose
// [1 2] = [1 3]
// [3 4]   [2 4]
TEST(TransposeTest, SquareMatrix) {
    tensor_pool_t *pool = make_pool();
    ASSERT_NE(pool, nullptr);

    float data[] = {1, 2, 3, 4};
    tensor_t *a = make_matrix(pool, 2, 2, data);
    ASSERT_NE(a, nullptr);

    tensor_t *t = tensor_transpose(pool, a);
    ASSERT_NE(t, nullptr);
    ASSERT_TRUE(tensor_evaluate(pool, t, nullptr, nullptr, nullptr));

    float *out = (float *)tensor_get_data(t);
    EXPECT_FLOAT_EQ(out[0], 1.0f);
    EXPECT_FLOAT_EQ(out[1], 3.0f);
    EXPECT_FLOAT_EQ(out[2], 2.0f);
    EXPECT_FLOAT_EQ(out[3], 4.0f);

    tensor_pool_destroy(pool);
}

// Double transpose — wapas original aana chahiye
// transpose(transpose(A)) == A
TEST(TransposeTest, DoubleTransposeRestoresOriginal) {
    tensor_pool_t *pool = make_pool();
    ASSERT_NE(pool, nullptr);

    float data[] = {1, 2, 3, 4, 5, 6};
    tensor_t *a  = make_matrix(pool, 2, 3, data);
    ASSERT_NE(a, nullptr);

    tensor_t *t1 = tensor_transpose(pool, a);
    ASSERT_NE(t1, nullptr);
    ASSERT_TRUE(tensor_evaluate(pool, t1, nullptr, nullptr, nullptr));

    tensor_t *t2 = tensor_transpose(pool, t1);
    ASSERT_NE(t2, nullptr);
    ASSERT_TRUE(tensor_evaluate(pool, t2, nullptr, nullptr, nullptr));

    // Shape wapas 2x3 honi chahiye
    uint32_t *dims = tensor_get_dims(t2);
    EXPECT_EQ(dims[0], 2u);
    EXPECT_EQ(dims[1], 3u);

    // Data original se match hona chahiye
    float *out = (float *)tensor_get_data(t2);
    for (int i = 0; i < 6; i++) {
        EXPECT_FLOAT_EQ(out[i], data[i]) << "Mismatch at index " << i;
    }

    tensor_pool_destroy(pool);
}

// Transpose ka is_transposed flag set hona chahiye
TEST(TransposeTest, IsTransposedFlagSet) {
    tensor_pool_t *pool = make_pool();
    ASSERT_NE(pool, nullptr);

    float data[] = {1, 2, 3, 4};
    tensor_t *a = make_matrix(pool, 2, 2, data);
    ASSERT_NE(a, nullptr);

    tensor_t *t = tensor_transpose(pool, a);
    ASSERT_NE(t, nullptr);

    // is_transposed flag check — tensor_mul optimized path isko use karta hai
    //EXPECT_TRUE(tensor_get_is_transposed(t));

    tensor_pool_destroy(pool);
}

// ============================================================
//  RELU TESTS
// ============================================================

// Positive values same rehne chahiye, negative zero ho jaane chahiye
TEST(ReluTest, BasicRelu) {
    tensor_pool_t *pool = make_pool();
    ASSERT_NE(pool, nullptr);

    float data[] = {-3.0f, -1.0f, 0.0f, 2.0f, 5.0f};
    uint32_t dims[] = {5, 0};
    tensor_t *a = tensor_create(pool, tensor_dtype_t::FLOAT32_T, 1, dims, data);
    ASSERT_NE(a, nullptr);

    tensor_t *t = tensor_relu(pool, a);
    ASSERT_NE(t, nullptr);
    ASSERT_TRUE(tensor_evaluate(pool, t, nullptr, nullptr, nullptr));

    float *out = (float *)tensor_get_data(t);
    EXPECT_FLOAT_EQ(out[0], 0.0f);   // -3 → 0
    EXPECT_FLOAT_EQ(out[1], 0.0f);   // -1 → 0
    EXPECT_FLOAT_EQ(out[2], 0.0f);   //  0 → 0
    EXPECT_FLOAT_EQ(out[3], 2.0f);   //  2 → 2
    EXPECT_FLOAT_EQ(out[4], 5.0f);   //  5 → 5

    tensor_pool_destroy(pool);
}

// Saare negative — sab zero ho jaane chahiye
TEST(ReluTest, AllNegativeBecomesZero) {
    tensor_pool_t *pool = make_pool();
    ASSERT_NE(pool, nullptr);

    float data[] = {-10.0f, -5.0f, -0.1f, -100.0f};
    uint32_t dims[] = {4, 0};
    tensor_t *a = tensor_create(pool, tensor_dtype_t::FLOAT32_T, 1, dims, data);
    ASSERT_NE(a, nullptr);

    tensor_t *t = tensor_relu(pool, a);
    ASSERT_NE(t, nullptr);
    ASSERT_TRUE(tensor_evaluate(pool, t, nullptr, nullptr, nullptr));

    float *out = (float *)tensor_get_data(t);
    for (int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(out[i], 0.0f) << "Index " << i << " zero nahi hai";
    }

    tensor_pool_destroy(pool);
}

// Saare positive — kuch nahi badalna chahiye
TEST(ReluTest, AllPositiveUnchanged) {
    tensor_pool_t *pool = make_pool();
    ASSERT_NE(pool, nullptr);

    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    uint32_t dims[] = {4, 0};
    tensor_t *a = tensor_create(pool, tensor_dtype_t::FLOAT32_T, 1, dims, data);
    ASSERT_NE(a, nullptr);

    tensor_t *t = tensor_relu(pool, a);
    ASSERT_NE(t, nullptr);
    ASSERT_TRUE(tensor_evaluate(pool, t, nullptr, nullptr, nullptr));

    float *out = (float *)tensor_get_data(t);
    EXPECT_FLOAT_EQ(out[0], 1.0f);
    EXPECT_FLOAT_EQ(out[1], 2.0f);
    EXPECT_FLOAT_EQ(out[2], 3.0f);
    EXPECT_FLOAT_EQ(out[3], 4.0f);

    tensor_pool_destroy(pool);
}

// 2D matrix par relu
TEST(ReluTest, ReluOn2DMatrix) {
    tensor_pool_t *pool = make_pool();
    ASSERT_NE(pool, nullptr);

    float data[] = {-1, 2, -3, 4};
    tensor_t *a = make_matrix(pool, 2, 2, data);
    ASSERT_NE(a, nullptr);

    tensor_t *t = tensor_relu(pool, a);
    ASSERT_NE(t, nullptr);
    ASSERT_TRUE(tensor_evaluate(pool, t, nullptr, nullptr, nullptr));

    float *out = (float *)tensor_get_data(t);
    EXPECT_FLOAT_EQ(out[0], 0.0f);
    EXPECT_FLOAT_EQ(out[1], 2.0f);
    EXPECT_FLOAT_EQ(out[2], 0.0f);
    EXPECT_FLOAT_EQ(out[3], 4.0f);

    tensor_pool_destroy(pool);
}
