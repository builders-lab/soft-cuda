#include "internal_header.h"

int main() {
    // Create pool
    tensor_pool_t *pool = tensor_pool_create(1024 * 1024);
    assert(pool != NULL);

    // Create two float32 scalars
    float val_a = 3.0f;
    float val_b = 4.0f;
    tensor_t *a = tensor_dtype_create(pool, tensor_dtype_t::FLOAT32_T, NULL, &val_a);
    tensor_t *b = tensor_dtype_create(pool, tensor_dtype_t::FLOAT32_T, NULL, &val_b);
    assert(a != NULL);
    assert(b != NULL);

    // Multiply them
    tensor_t *c = tensor_mul(pool, a, b);
    assert(c != NULL);

    // Evaluate
    bool ok = tensor_evaluate(pool, c);
    assert(ok);

    // Check result
    float result = *((float *)c->data);
    assert(result == 12.0f);
    debug("result = %f\n", result);

    tensor_pool_destroy(pool);
    return 0;
}
