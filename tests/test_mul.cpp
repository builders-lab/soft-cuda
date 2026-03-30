#include <gtest/gtest.h>
#include "soft-cuda/tensor/api.h"

// ============================================================
//  HELPER FUNCTIONS
// ============================================================

static tensor_pool_t *make_pool() {
    return tensor_pool_create(1024 * 64);
}

static tensor_t *make_matrix(tensor_pool_t *pool, uint32_t rows, uint32_t cols, float *data) {
    uint32_t dims[3] = {rows, cols, 0};
    return tensor_create(pool, tensor_dtype_t::FLOAT32_T, 2, dims, data);
}

static tensor_t *make_scalar(tensor_pool_t *pool, float val) {
    return tensor_create(pool, tensor_dtype_t::FLOAT32_T, 0, nullptr, &val);
}

// ============================================================
//  NAIVE MATRIX MUL TESTS
// ============================================================

// [1 2] * [5 6] = [19 22]
// [3 4]   [7 8]   [43 50]
TEST(NaiveMulTest, TwoByTwo) {
    tensor_pool_t *pool = make_pool();
    ASSERT_NE(pool, nullptr);

    float a_data[] = {1, 2, 3, 4};
    float b_data[] = {5, 6, 7, 8};
    tensor_t *a = make_matrix(pool, 2, 2, a_data);
    tensor_t *b = make_matrix(pool, 2, 2, b_data);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);

    tensor_t *t = tensor_mul_naive(pool, a, b);
    ASSERT_NE(t, nullptr);
    ASSERT_TRUE(tensor_evaluate(pool, t, nullptr, nullptr, nullptr));

    float *out = (float *)tensor_get_data(t);
    EXPECT_FLOAT_EQ(out[0], 19.0f);
    EXPECT_FLOAT_EQ(out[1], 22.0f);
    EXPECT_FLOAT_EQ(out[2], 43.0f);
    EXPECT_FLOAT_EQ(out[3], 50.0f);

    tensor_pool_destroy(pool);
}

// [1 2 3] * [7  8 ] = [58  64 ]
// [4 5 6]   [9  10]   [139 154]
//           [11 12]
TEST(NaiveMulTest, RectangularMatrix) {
    tensor_pool_t *pool = make_pool();
    ASSERT_NE(pool, nullptr);

    float a_data[] = {1, 2, 3, 4, 5, 6};
    float b_data[] = {7, 8, 9, 10, 11, 12};
    tensor_t *a = make_matrix(pool, 2, 3, a_data);
    tensor_t *b = make_matrix(pool, 3, 2, b_data);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);

    tensor_t *t = tensor_mul_naive(pool, a, b);
    ASSERT_NE(t, nullptr);
    ASSERT_TRUE(tensor_evaluate(pool, t, nullptr, nullptr, nullptr));

    float *out = (float *)tensor_get_data(t);
    EXPECT_FLOAT_EQ(out[0], 58.0f);
    EXPECT_FLOAT_EQ(out[1], 64.0f);
    EXPECT_FLOAT_EQ(out[2], 139.0f);
    EXPECT_FLOAT_EQ(out[3], 154.0f);

    tensor_pool_destroy(pool);
}

// [1 0] * [3 4] = [3 4]
// [0 1]   [5 6]   [5 6]
TEST(NaiveMulTest, IdentityMatrix) {
    tensor_pool_t *pool = make_pool();
    ASSERT_NE(pool, nullptr);

    float identity[] = {1, 0, 0, 1};
    float b_data[]   = {3, 4, 5, 6};
    tensor_t *a = make_matrix(pool, 2, 2, identity);
    tensor_t *b = make_matrix(pool, 2, 2, b_data);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);

    tensor_t *t = tensor_mul_naive(pool, a, b);
    ASSERT_NE(t, nullptr);
    ASSERT_TRUE(tensor_evaluate(pool, t, nullptr, nullptr, nullptr));

    float *out = (float *)tensor_get_data(t);
    EXPECT_FLOAT_EQ(out[0], 3.0f);
    EXPECT_FLOAT_EQ(out[1], 4.0f);
    EXPECT_FLOAT_EQ(out[2], 5.0f);
    EXPECT_FLOAT_EQ(out[3], 6.0f);

    tensor_pool_destroy(pool);
}

// A * 0 = 0
TEST(NaiveMulTest, ZeroMatrix) {
    tensor_pool_t *pool = make_pool();
    ASSERT_NE(pool, nullptr);

    float a_data[] = {1, 2, 3, 4};
    float zeros[]  = {0, 0, 0, 0};
    tensor_t *a = make_matrix(pool, 2, 2, a_data);
    tensor_t *b = make_matrix(pool, 2, 2, zeros);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);

    tensor_t *t = tensor_mul_naive(pool, a, b);
    ASSERT_NE(t, nullptr);
    ASSERT_TRUE(tensor_evaluate(pool, t, nullptr, nullptr, nullptr));

    float *out = (float *)tensor_get_data(t);
    for (int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(out[i], 0.0f) << "Index " << i << " zero nahi hai";
    }

    tensor_pool_destroy(pool);
}

// 2x3 * 3x4 = 2x4 — shape check
TEST(NaiveMulTest, OutputShapeCorrect) {
    tensor_pool_t *pool = make_pool();
    ASSERT_NE(pool, nullptr);

    float a_data[6]  = {1, 2, 3, 4, 5, 6};
    float b_data[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    tensor_t *a = make_matrix(pool, 2, 3, a_data);
    tensor_t *b = make_matrix(pool, 3, 4, b_data);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);

    tensor_t *t = tensor_mul_naive(pool, a, b);
    ASSERT_NE(t, nullptr);

    uint32_t *dims = tensor_get_dims(t);
    EXPECT_EQ(dims[0], 2u);
    EXPECT_EQ(dims[1], 4u);

    tensor_pool_destroy(pool);
}

// Dim mismatch — 2x3 * 2x3 invalid, assert fire hoga
TEST(NaiveMulTest, DimMismatchReturnsNull) {
    tensor_pool_t *pool = make_pool();
    ASSERT_NE(pool, nullptr);

    float data[] = {1, 2, 3, 4, 5, 6};
    tensor_t *a = make_matrix(pool, 2, 3, data);
    tensor_t *b = make_matrix(pool, 2, 3, data);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);

    EXPECT_DEATH(tensor_mul_naive(pool, a, b), "");

    tensor_pool_destroy(pool);
}

// ============================================================
//  OPTIMIZED MUL TEST — naive se result same hona chahiye
// ============================================================

TEST(OptimizedMulTest, MatchesNaiveTwoByTwo) {
    tensor_pool_t *pool = make_pool();
    ASSERT_NE(pool, nullptr);

    float a_data[] = {1, 2, 3, 4};
    float b_data[] = {5, 6, 7, 8};
    tensor_t *a = make_matrix(pool, 2, 2, a_data);
    tensor_t *b = make_matrix(pool, 2, 2, b_data);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);

    // Naive path
    tensor_t *naive = tensor_mul_naive(pool, a, b);
    ASSERT_NE(naive, nullptr);
    ASSERT_TRUE(tensor_evaluate(pool, naive, nullptr, nullptr, nullptr));

    // Optimized path — b ko transpose karo pehle
    tensor_t *bt = tensor_transpose(pool, b);
    ASSERT_NE(bt, nullptr);
    ASSERT_TRUE(tensor_evaluate(pool, bt, nullptr, nullptr, nullptr));

    tensor_t *opt = tensor_mul(pool, a, bt);
    ASSERT_NE(opt, nullptr);
    ASSERT_TRUE(tensor_evaluate(pool, opt, nullptr, nullptr, nullptr));

    float *n = (float *)tensor_get_data(naive);
    float *o = (float *)tensor_get_data(opt);
    for (int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(o[i], n[i]) << "Mismatch at index " << i;
    }

    tensor_pool_destroy(pool);
}

// ============================================================
//  SCALAR MUL TESTS
// ============================================================

// [2 4 6] * 3.0 = [6 12 18]
TEST(ScalarMulTest, BasicScalarMul) {
    tensor_pool_t *pool = make_pool();
    ASSERT_NE(pool, nullptr);

    float data[] = {2.0f, 4.0f, 6.0f};
    uint32_t dims[] = {3, 0};
    tensor_t *a = tensor_create(pool, tensor_dtype_t::FLOAT32_T, 1, dims, data);
    tensor_t *s = make_scalar(pool, 3.0f);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(s, nullptr);

    tensor_t *t = tensor_mul(pool, a, s);
    ASSERT_NE(t, nullptr);
    ASSERT_TRUE(tensor_evaluate(pool, t, nullptr, nullptr, nullptr));

    float *out = (float *)tensor_get_data(t);
    EXPECT_FLOAT_EQ(out[0], 6.0f);
    EXPECT_FLOAT_EQ(out[1], 12.0f);
    EXPECT_FLOAT_EQ(out[2], 18.0f);

    tensor_pool_destroy(pool);
}

// A * 1 = A
TEST(ScalarMulTest, MultiplyByOne) {
    tensor_pool_t *pool = make_pool();
    ASSERT_NE(pool, nullptr);

    float data[] = {5.0f, 10.0f, 15.0f};
    uint32_t dims[] = {3, 0};
    tensor_t *a = tensor_create(pool, tensor_dtype_t::FLOAT32_T, 1, dims, data);
    tensor_t *s = make_scalar(pool, 1.0f);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(s, nullptr);

    tensor_t *t = tensor_mul(pool, a, s);
    ASSERT_NE(t, nullptr);
    ASSERT_TRUE(tensor_evaluate(pool, t, nullptr, nullptr, nullptr));

    float *out = (float *)tensor_get_data(t);
    EXPECT_FLOAT_EQ(out[0], 5.0f);
    EXPECT_FLOAT_EQ(out[1], 10.0f);
    EXPECT_FLOAT_EQ(out[2], 15.0f);

    tensor_pool_destroy(pool);
}

// A * 0 = 0
TEST(ScalarMulTest, MultiplyByZero) {
    tensor_pool_t *pool = make_pool();
    ASSERT_NE(pool, nullptr);

    float data[] = {5.0f, 10.0f, 15.0f};
    uint32_t dims[] = {3, 0};
    tensor_t *a = tensor_create(pool, tensor_dtype_t::FLOAT32_T, 1, dims, data);
    tensor_t *s = make_scalar(pool, 0.0f);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(s, nullptr);

    tensor_t *t = tensor_mul(pool, a, s);
    ASSERT_NE(t, nullptr);
    ASSERT_TRUE(tensor_evaluate(pool, t, nullptr, nullptr, nullptr));

    float *out = (float *)tensor_get_data(t);
    for (int i = 0; i < 3; i++) {
        EXPECT_FLOAT_EQ(out[i], 0.0f);
    }

    tensor_pool_destroy(pool);
}

// ============================================================
//  TENSOR ADD TESTS
// ============================================================

// [1 2] + [5 6] = [6  8 ]
// [3 4]   [7 8]   [10 12]
TEST(AddTest, BasicMatrixAdd) {
    tensor_pool_t *pool = make_pool();
    ASSERT_NE(pool, nullptr);

    float a_data[] = {1, 2, 3, 4};
    float b_data[] = {5, 6, 7, 8};
    tensor_t *a = make_matrix(pool, 2, 2, a_data);
    tensor_t *b = make_matrix(pool, 2, 2, b_data);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);

    tensor_t *t = tensor_add(pool, a, b);
    ASSERT_NE(t, nullptr);
    ASSERT_TRUE(tensor_evaluate(pool, t, nullptr, nullptr, nullptr));

    float *out = (float *)tensor_get_data(t);
    EXPECT_FLOAT_EQ(out[0], 6.0f);
    EXPECT_FLOAT_EQ(out[1], 8.0f);
    EXPECT_FLOAT_EQ(out[2], 10.0f);
    EXPECT_FLOAT_EQ(out[3], 12.0f);

    tensor_pool_destroy(pool);
}

// A + 0 = A
TEST(AddTest, AddZeroMatrix) {
    tensor_pool_t *pool = make_pool();
    ASSERT_NE(pool, nullptr);

    float a_data[] = {3, 1, 4, 1};
    float zeros[]  = {0, 0, 0, 0};
    tensor_t *a = make_matrix(pool, 2, 2, a_data);
    tensor_t *b = make_matrix(pool, 2, 2, zeros);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);

    tensor_t *t = tensor_add(pool, a, b);
    ASSERT_NE(t, nullptr);
    ASSERT_TRUE(tensor_evaluate(pool, t, nullptr, nullptr, nullptr));

    float *out = (float *)tensor_get_data(t);
    EXPECT_FLOAT_EQ(out[0], 3.0f);
    EXPECT_FLOAT_EQ(out[1], 1.0f);
    EXPECT_FLOAT_EQ(out[2], 4.0f);
    EXPECT_FLOAT_EQ(out[3], 1.0f);

    tensor_pool_destroy(pool);
}

// Broadcast — bias row [1 2] ko har row mein add karo
// [1 2] + [1 2] = [2 4]
// [3 4]           [4 6]
TEST(AddTest, BroadcastBiasAdd) {
    tensor_pool_t *pool = make_pool();
    ASSERT_NE(pool, nullptr);

    float a_data[] = {1, 2, 3, 4};
    float bias[]   = {1, 2};
    tensor_t *a = make_matrix(pool, 2, 2, a_data);
    tensor_t *b = make_matrix(pool, 1, 2, bias);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);

    tensor_t *t = tensor_add(pool, a, b);
    ASSERT_NE(t, nullptr);
    ASSERT_TRUE(tensor_evaluate(pool, t, nullptr, nullptr, nullptr));

    float *out = (float *)tensor_get_data(t);
    EXPECT_FLOAT_EQ(out[0], 2.0f);
    EXPECT_FLOAT_EQ(out[1], 4.0f);
    EXPECT_FLOAT_EQ(out[2], 4.0f);
    EXPECT_FLOAT_EQ(out[3], 6.0f);

    tensor_pool_destroy(pool);
}

// Output shape same honi chahiye
TEST(AddTest, OutputShapeCorrect) {
    tensor_pool_t *pool = make_pool();
    ASSERT_NE(pool, nullptr);

    float data[] = {1, 2, 3, 4, 5, 6};
    tensor_t *a = make_matrix(pool, 2, 3, data);
    tensor_t *b = make_matrix(pool, 2, 3, data);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);

    tensor_t *t = tensor_add(pool, a, b);
    ASSERT_NE(t, nullptr);

    uint32_t *dims = tensor_get_dims(t);
    EXPECT_EQ(dims[0], 2u);
    EXPECT_EQ(dims[1], 3u);

    tensor_pool_destroy(pool);
}
