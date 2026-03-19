#include <gtest/gtest.h>
#include "soft-cuda/tensor/api.h"
#include "tensor/tensor.h"

// ============================================================
//  HELPER — ek chhota pool banana aur 2D tensor banana
// ============================================================

static tensor_pool_t *make_pool() {
    return tensor_pool_create(1024 * 64);  // 64KB — sabke liye kaafi hai
}

// 2D float32 tensor banao with zero-terminated dims
static tensor_t *make_matrix(tensor_pool_t *pool, uint32_t rows, uint32_t cols, float *data) {
    uint32_t dims[3] = {rows, cols, 0};
    return tensor_create(pool, tensor_dtype_t::FLOAT32_T, 2, dims, data);
}

// Scalar float32 tensor banao
static tensor_t *make_scalar(tensor_pool_t *pool, float val) {
    return tensor_create(pool, tensor_dtype_t::FLOAT32_T, 0, nullptr, &val);
}

// ============================================================
//  NAIVE MATRIX MUL TESTS  (tensor_mul_naive + tensor_evaluate)
//
//  tensor_mul_naive lazy hai — sirf op set karta hai
//  tensor_evaluate tab actual compute karta hai
// ============================================================

// 2x2 * 2x2 — sabse basic case
// [1 2] * [5 6] = [1*5+2*7  1*6+2*8] = [19 22]
// [3 4]   [7 8]   [3*5+4*7  3*6+4*8]   [43 50]
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

    bool ok = tensor_evaluate(pool, t);
    ASSERT_TRUE(ok);

    float *out = (float *)t->data;
    EXPECT_FLOAT_EQ(out[0], 19.0f);
    EXPECT_FLOAT_EQ(out[1], 22.0f);
    EXPECT_FLOAT_EQ(out[2], 43.0f);
    EXPECT_FLOAT_EQ(out[3], 50.0f);

    tensor_pool_destroy(pool);
}

// 2x3 * 3x2 — rectangular matrices
// [1 2 3] * [7  8 ] = [1*7+2*9+3*11   1*8+2*10+3*12] = [58  64]
// [4 5 6]   [9  10]   [4*7+5*9+6*11   4*8+5*10+6*12]   [139 154]
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

    bool ok = tensor_evaluate(pool, t);
    ASSERT_TRUE(ok);

    float *out = (float *)t->data;
    EXPECT_FLOAT_EQ(out[0], 58.0f);
    EXPECT_FLOAT_EQ(out[1], 64.0f);
    EXPECT_FLOAT_EQ(out[2], 139.0f);
    EXPECT_FLOAT_EQ(out[3], 154.0f);

    tensor_pool_destroy(pool);
}

// Identity matrix se multiply karne par original matrix milni chahiye
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
    ASSERT_TRUE(tensor_evaluate(pool, t));

    float *out = (float *)t->data;
    EXPECT_FLOAT_EQ(out[0], 3.0f);
    EXPECT_FLOAT_EQ(out[1], 4.0f);
    EXPECT_FLOAT_EQ(out[2], 5.0f);
    EXPECT_FLOAT_EQ(out[3], 6.0f);

    tensor_pool_destroy(pool);
}

// Zero matrix se multiply karne par sab zero aana chahiye
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
    ASSERT_TRUE(tensor_evaluate(pool, t));

    float *out = (float *)t->data;
    for (int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(out[i], 0.0f) << "Index " << i << " zero nahi hai";
    }

    tensor_pool_destroy(pool);
}

// Output tensor ka shape sahi hona chahiye
// 2x3 * 3x4 = 2x4
TEST(NaiveMulTest, OutputShapeCorrect) {
    tensor_pool_t *pool = make_pool();
    ASSERT_NE(pool, nullptr);

    float a_data[6] = {1,2,3,4,5,6};
    float b_data[12] = {1,2,3,4,5,6,7,8,9,10,11,12};
    tensor_t *a = make_matrix(pool, 2, 3, a_data);
    tensor_t *b = make_matrix(pool, 3, 4, b_data);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);

    tensor_t *t = tensor_mul_naive(pool, a, b);
    ASSERT_NE(t, nullptr);

    // Shape check — evaluate se pehle bhi dims set hone chahiye
    EXPECT_EQ(t->dims[0], 2u);
    EXPECT_EQ(t->dims[1], 4u);

    tensor_pool_destroy(pool);
}

// Dimension mismatch par NULL milna chahiye (2x3 * 2x3 — invalid)
TEST(NaiveMulTest, DimMismatchReturnsNull) {
    tensor_pool_t *pool = make_pool();
    ASSERT_NE(pool, nullptr);

    float data[] = {1, 2, 3, 4, 5, 6};
    tensor_t *a = make_matrix(pool, 2, 3, data);
    tensor_t *b = make_matrix(pool, 2, 3, data);  // cols != rows — invalid
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);

    // assert() fire hoga — death test
    EXPECT_DEATH(tensor_mul_naive(pool, a, b), "");

    tensor_pool_destroy(pool);
}

// ============================================================
//  CACHE-OPTIMIZED MUL TESTS  (tensor_mul — transposed path)
//
//  tensor_mul internally transposes b for cache efficiency
//  Result naive se same hona chahiye
// ============================================================

// Optimized aur naive ka result same hona chahiye — 2x2
TEST(OptimizedMulTest, MatchesNaiveTwoByTwo) {
    tensor_pool_t *pool = make_pool();
    ASSERT_NE(pool, nullptr);

    float a_data[] = {1, 2, 3, 4};
    float b_data[] = {5, 6, 7, 8};
    tensor_t *a  = make_matrix(pool, 2, 2, a_data);
    tensor_t *b  = make_matrix(pool, 2, 2, b_data);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);

    // Naive path
    tensor_t *naive = tensor_mul_naive(pool, a, b);
    ASSERT_NE(naive, nullptr);
    ASSERT_TRUE(tensor_evaluate(pool, naive));

    // Optimized path — tensor_mul uses transposed b internally
    tensor_t *bt = tensor_transpose(pool, b);
    ASSERT_NE(bt, nullptr);
    ASSERT_TRUE(tensor_evaluate(pool, bt));

    tensor_t *opt = tensor_mul(pool, a, bt);
    ASSERT_NE(opt, nullptr);
    ASSERT_TRUE(tensor_evaluate(pool, opt));

    float *n = (float *)naive->data;
    float *o = (float *)opt->data;
    for (int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(o[i], n[i]) << "Mismatch at index " << i;
    }

    tensor_pool_destroy(pool);
}

// ============================================================
//  SCALAR MUL TESTS
// ============================================================

// Matrix ko scalar se multiply karo
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

    tensor_t *t = tensor_mul(pool, a, s);  // scalar detect hoga automatically
    ASSERT_NE(t, nullptr);
    ASSERT_TRUE(tensor_evaluate(pool, t));

    float *out = (float *)t->data;
    EXPECT_FLOAT_EQ(out[0], 6.0f);
    EXPECT_FLOAT_EQ(out[1], 12.0f);
    EXPECT_FLOAT_EQ(out[2], 18.0f);

    tensor_pool_destroy(pool);
}

// Scalar 1.0 se multiply karne par kuch nahi badalna chahiye
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
    ASSERT_TRUE(tensor_evaluate(pool, t));

    float *out = (float *)t->data;
    EXPECT_FLOAT_EQ(out[0], 5.0f);
    EXPECT_FLOAT_EQ(out[1], 10.0f);
    EXPECT_FLOAT_EQ(out[2], 15.0f);

    tensor_pool_destroy(pool);
}

// Scalar 0.0 se multiply karne par sab zero hona chahiye
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
    ASSERT_TRUE(tensor_evaluate(pool, t));

    float *out = (float *)t->data;
    for (int i = 0; i < 3; i++) {
        EXPECT_FLOAT_EQ(out[i], 0.0f);
    }

    tensor_pool_destroy(pool);
}
