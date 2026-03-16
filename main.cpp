#include "soft-cuda/tensor/api.h"
#include "soft-cuda/tensor/debug_api.h"

int main() {
    // Create pool
    tensor_pool_t *pool = tensor_pool_create(1024 * 1024);
    assert(pool != NULL);

    // Create two float32 scalars
    float val_a[] = {3.0f, 6.0f};
    float val_b = 4.0f;

    uint32_t dims[] = {2};
    tensor_t *a = tensor_create(pool, tensor_dtype_t::FLOAT32_T, 1, dims, val_a);
    tensor_t *b = tensor_create(pool, tensor_dtype_t::FLOAT32_T, 0, NULL, &val_b);
    assert(a != NULL);
    assert(b != NULL);
    
    tensor_t *c = tensor_mul(pool, a, b);
    assert(c != NULL);
    

    bool ok = tensor_evaluate(pool, c);
    assert(ok);
    uint32_t dims_a[] = {1, 2};
    tensor_t *ac = tensor_create(pool, tensor_dtype_t::FLOAT32_T, 2, dims_a, val_a);
    tensor_t *d =tensor_transpose(pool, ac);
    bool oki = tensor_evaluate(pool, d);
    assert(oki);
    
    tensor_print_data(d);
    float *resulti = (float *)tensor_get_data(d);
    
    float *result = (float *)tensor_get_data(c);
    assert(result[0] == 12.0f);
    assert(result[1] == 24.0f);
    debug("result[0] = %f\n", result[0]);
    debug("result[1] = %f\n", result[1]);
    tensor_pool_destroy(pool);
    return 0;
}
