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
    return tensor_create(pool, tensor_dtype_t::FLOAT32_T, 2, dims, data, true);
}

static tensor_t *make_1d(tensor_pool_t *pool, uint32_t n, float *data) {
    uint32_t dims[2] = {n, 0};
    return tensor_create(pool, tensor_dtype_t::FLOAT32_T, 1, dims, data, true);
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

    uint32_t *dims = tensor_get_dims(t);
    EXPECT_EQ(dims[0], 3u);
    EXPECT_EQ(dims[1], 2u);

    float *out = (float *)tensor_get_data(t);
    EXPECT_FLOAT_EQ(out[0], 1.0f);
    EXPECT_FLOAT_EQ(out[1], 4.0f);
    EXPECT_FLOAT_EQ(out[2], 2.0f);
    EXPECT_FLOAT_EQ(out[3], 5.0f);
    EXPECT_FLOAT_EQ(out[4], 3.0f);
    EXPECT_FLOAT_EQ(out[5], 6.0f);

    tensor_pool_destroy(pool);
}

// [1 2] transpose = [1 3]
// [3 4]             [2 4]
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

    uint32_t *dims = tensor_get_dims(t2);
    EXPECT_EQ(dims[0], 2u);
    EXPECT_EQ(dims[1], 3u);

    float *out = (float *)tensor_get_data(t2);
    for (int i = 0; i < 6; i++) {
        EXPECT_FLOAT_EQ(out[i], data[i]) << "Mismatch at index " << i;
    }

    tensor_pool_destroy(pool);
}

TEST(TransposeTest, DISABLED_IsTransposedFlagSet) {
    tensor_pool_t *pool = make_pool();
    ASSERT_NE(pool, nullptr);

    float data[] = {1, 2, 3, 4};
    tensor_t *a = make_matrix(pool, 2, 2, data);
    ASSERT_NE(a, nullptr);

    tensor_t *t = tensor_transpose(pool, a);
    ASSERT_NE(t, nullptr);

    

    tensor_pool_destroy(pool);
}

// ============================================================
//  RELU TESTS
// ============================================================

TEST(ReluTest, BasicRelu) {
    tensor_pool_t *pool = make_pool();
    ASSERT_NE(pool, nullptr);

    float data[] = {-3.0f, -1.0f, 0.0f, 2.0f, 5.0f};
    tensor_t *a = make_1d(pool, 5, data);
    ASSERT_NE(a, nullptr);

    tensor_t *t = tensor_relu(pool, a);
    ASSERT_NE(t, nullptr);
    ASSERT_TRUE(tensor_evaluate(pool, t, nullptr, nullptr, nullptr));

    float *out = (float *)tensor_get_data(t);
    EXPECT_FLOAT_EQ(out[0], 0.0f);
    EXPECT_FLOAT_EQ(out[1], 0.0f);
    EXPECT_FLOAT_EQ(out[2], 0.0f);
    EXPECT_FLOAT_EQ(out[3], 2.0f);
    EXPECT_FLOAT_EQ(out[4], 5.0f);

    tensor_pool_destroy(pool);
}

TEST(ReluTest, AllNegativeBecomesZero) {
    tensor_pool_t *pool = make_pool();
    ASSERT_NE(pool, nullptr);

    float data[] = {-10.0f, -5.0f, -0.1f, -100.0f};
    tensor_t *a = make_1d(pool, 4, data);
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

TEST(ReluTest, AllPositiveUnchanged) {
    tensor_pool_t *pool = make_pool();
    ASSERT_NE(pool, nullptr);

    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    tensor_t *a = make_1d(pool, 4, data);
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

// ============================================================
//  SUB TESTS
// ============================================================

// [5 6] - [1 2] = [4 4]
// [7 8]   [3 4]   [4 4]
TEST(SubTest, BasicSub) {
    tensor_pool_t *pool = make_pool();
    ASSERT_NE(pool, nullptr);

    float a_data[] = {5, 6, 7, 8};
    float b_data[] = {1, 2, 3, 4};
    tensor_t *a = make_matrix(pool, 2, 2, a_data);
    tensor_t *b = make_matrix(pool, 2, 2, b_data);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);

    tensor_t *t = tensor_sub(pool, a, b);
    ASSERT_NE(t, nullptr);
    ASSERT_TRUE(tensor_evaluate(pool, t, nullptr, nullptr, nullptr));

    float *out = (float *)tensor_get_data(t);
    EXPECT_FLOAT_EQ(out[0], 4.0f);
    EXPECT_FLOAT_EQ(out[1], 4.0f);
    EXPECT_FLOAT_EQ(out[2], 4.0f);
    EXPECT_FLOAT_EQ(out[3], 4.0f);

    tensor_pool_destroy(pool);
}

// A - A = 0
TEST(SubTest, SelfSubIsZero) {
    tensor_pool_t *pool = make_pool();
    ASSERT_NE(pool, nullptr);

    float data[] = {3, 1, 4, 1};
    tensor_t *a = make_matrix(pool, 2, 2, data);
    ASSERT_NE(a, nullptr);

    tensor_t *t = tensor_sub(pool, a, a);
    ASSERT_NE(t, nullptr);
    ASSERT_TRUE(tensor_evaluate(pool, t, nullptr, nullptr, nullptr));

    float *out = (float *)tensor_get_data(t);
    for (int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(out[i], 0.0f) << "Index " << i;
    }

    tensor_pool_destroy(pool);
}

// A - 0 = A
TEST(SubTest, SubZeroIsIdentity) {
    tensor_pool_t *pool = make_pool();
    ASSERT_NE(pool, nullptr);

    float a_data[] = {1, 2, 3, 4};
    float zeros[]  = {0, 0, 0, 0};
    tensor_t *a = make_matrix(pool, 2, 2, a_data);
    tensor_t *b = make_matrix(pool, 2, 2, zeros);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);

    tensor_t *t = tensor_sub(pool, a, b);
    ASSERT_NE(t, nullptr);
    ASSERT_TRUE(tensor_evaluate(pool, t, nullptr, nullptr, nullptr));

    float *out = (float *)tensor_get_data(t);
    EXPECT_FLOAT_EQ(out[0], 1.0f);
    EXPECT_FLOAT_EQ(out[1], 2.0f);
    EXPECT_FLOAT_EQ(out[2], 3.0f);
    EXPECT_FLOAT_EQ(out[3], 4.0f);

    tensor_pool_destroy(pool);
}

// ============================================================
//  SQUARE TESTS
// ============================================================

// [1 2 3 4] ^ 2 = [1 4 9 16]
TEST(SquareTest, BasicSquare) {
    tensor_pool_t *pool = make_pool();
    ASSERT_NE(pool, nullptr);

    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    tensor_t *a = make_1d(pool, 4, data);
    ASSERT_NE(a, nullptr);

    tensor_t *t = tensor_square(pool, a);
    ASSERT_NE(t, nullptr);
    ASSERT_TRUE(tensor_evaluate(pool, t, nullptr, nullptr, nullptr));

    float *out = (float *)tensor_get_data(t);
    EXPECT_FLOAT_EQ(out[0], 1.0f);
    EXPECT_FLOAT_EQ(out[1], 4.0f);
    EXPECT_FLOAT_EQ(out[2], 9.0f);
    EXPECT_FLOAT_EQ(out[3], 16.0f);

    tensor_pool_destroy(pool);
}

// Negative bhi square hone par positive hona chahiye
// [-2 -3] ^ 2 = [4 9]
TEST(SquareTest, NegativeSquaredIsPositive) {
    tensor_pool_t *pool = make_pool();
    ASSERT_NE(pool, nullptr);

    float data[] = {-2.0f, -3.0f};
    tensor_t *a = make_1d(pool, 2, data);
    ASSERT_NE(a, nullptr);

    tensor_t *t = tensor_square(pool, a);
    ASSERT_NE(t, nullptr);
    ASSERT_TRUE(tensor_evaluate(pool, t, nullptr, nullptr, nullptr));

    float *out = (float *)tensor_get_data(t);
    EXPECT_FLOAT_EQ(out[0], 4.0f);
    EXPECT_FLOAT_EQ(out[1], 9.0f);

    tensor_pool_destroy(pool);
}

// Zero square = zero
TEST(SquareTest, ZeroSquaredIsZero) {
    tensor_pool_t *pool = make_pool();
    ASSERT_NE(pool, nullptr);

    float data[] = {0.0f, 0.0f, 0.0f};
    tensor_t *a = make_1d(pool, 3, data);
    ASSERT_NE(a, nullptr);

    tensor_t *t = tensor_square(pool, a);
    ASSERT_NE(t, nullptr);
    ASSERT_TRUE(tensor_evaluate(pool, t, nullptr, nullptr, nullptr));

    float *out = (float *)tensor_get_data(t);
    for (int i = 0; i < 3; i++) {
        EXPECT_FLOAT_EQ(out[i], 0.0f);
    }

    tensor_pool_destroy(pool);
}

// ============================================================
//  MEAN TESTS
//  Note: tensor_op_mean t->grad->data access karta hai
//        grad_status=true zaroori hai
// ============================================================

// mean([2 4 6]) = 4.0
TEST(MeanTest, DISABLED_BasicMean) {
    tensor_pool_t *pool = make_pool();
    ASSERT_NE(pool, nullptr);

    float data[] = {2.0f, 4.0f, 6.0f};
    tensor_t *a = make_1d(pool, 3, data);
    ASSERT_NE(a, nullptr);

    tensor_t *t = tensor_mean(pool, a);
    ASSERT_NE(t, nullptr);
    ASSERT_TRUE(tensor_evaluate(pool, t, nullptr, nullptr, nullptr));

    float *out = (float *)tensor_get_data(t);
    EXPECT_FLOAT_EQ(out[0], 4.0f);

    tensor_pool_destroy(pool);
}

// mean([1 1 1 1]) = 1.0
TEST(MeanTest, DISABLED_UniformMean) {
    tensor_pool_t *pool = make_pool();
    ASSERT_NE(pool, nullptr);

    float data[] = {1.0f, 1.0f, 1.0f, 1.0f};
    tensor_t *a = make_1d(pool, 4, data);
    ASSERT_NE(a, nullptr);

    tensor_t *t = tensor_mean(pool, a);
    ASSERT_NE(t, nullptr);
    ASSERT_TRUE(tensor_evaluate(pool, t, nullptr, nullptr, nullptr));

    float *out = (float *)tensor_get_data(t);
    EXPECT_FLOAT_EQ(out[0], 1.0f);

    tensor_pool_destroy(pool);
}

// mean([0 10]) = 5.0
TEST(MeanTest, DISABLED_TwoElementMean) {
    tensor_pool_t *pool = make_pool();
    ASSERT_NE(pool, nullptr);

    float data[] = {0.0f, 10.0f};
    tensor_t *a = make_1d(pool, 2, data);
    ASSERT_NE(a, nullptr);

    tensor_t *t = tensor_mean(pool, a);
    ASSERT_NE(t, nullptr);
    ASSERT_TRUE(tensor_evaluate(pool, t, nullptr, nullptr, nullptr));

    float *out = (float *)tensor_get_data(t);
    EXPECT_FLOAT_EQ(out[0], 5.0f);

    tensor_pool_destroy(pool);
}

// Mean output scalar hona chahiye — ndims == 0
TEST(MeanTest, OutputIsScalar) {
    tensor_pool_t *pool = make_pool();
    ASSERT_NE(pool, nullptr);

    float data[] = {1.0f, 2.0f, 3.0f};
    tensor_t *a = make_1d(pool, 3, data);
    ASSERT_NE(a, nullptr);

    tensor_t *t = tensor_mean(pool, a);
    ASSERT_NE(t, nullptr);

    EXPECT_EQ(tensor_get_ndims(t), 0);

    tensor_pool_destroy(pool);
}
