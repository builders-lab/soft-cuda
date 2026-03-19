#include "soft-cuda/tensor/api.h"
#include "soft-cuda/tensor/debug_api.h"
#include "iostream"

using namespace std;


int main() {
    // Create pool
    tensor_pool_t *pool = tensor_pool_create(1024 * 1024);
    assert(pool != NULL);

    // Create two float32 scalars
    //
    // // Create two float32 scalars
    // float val_a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    // float val_b[] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    //
    // uint32_t dims_a[] = {3, 2};
    // uint32_t dims_b[] = {2, 3};
    // tensor_t *a = tensor_create(pool, tensor_dtype_t::FLOAT32_T, 2, dims_a, val_a);
    // tensor_t *b = tensor_create(pool, tensor_dtype_t::FLOAT32_T, 2, dims_b, val_b);
    // tensor_t *c =tensor_transpose(pool, b);
    // tensor_t *d =tensor_mul(pool, a,b);
    // cout << tensor_id(a) << "\n";
    // cout << tensor_id(b) << "\n";
    // cout << tensor_id(c) << "\n";
    // cout << tensor_id(d) << "\n";
    // bool ok = tensor_evaluate(pool, c);
    // assert(ok);
    // bool oki = tensor_evaluate(pool, d);
    // assert(oki);
    //
    // tensor_print_data(d);
    //
    cout << "=========================================== \n";
    cout << "=============TESTING ADDITION============== \n";
    float val_a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float val_b[] = {7.0f, 8.0f};

    uint32_t dims_a[] = {3, 2};
    uint32_t dims_b[] = {1, 2};

    tensor_t *a = tensor_create(pool, tensor_dtype_t::FLOAT32_T, 2, dims_a, val_a);
    tensor_t *b = tensor_create(pool, tensor_dtype_t::FLOAT32_T, 2, dims_b, val_b);
    
    tensor_t *c =tensor_add(pool, a,b);
    cout << "=========================================== \n";
    cout << "==============LAZY EVAL DONE=============== \n";
    bool ok = tensor_evaluate(pool, c);
    assert(ok);
    tensor_print_data(c);
 
    
    // debug("result[0] = %f\n", result[0]);
    // debug("result[1] = %f\n", result[1]);
    tensor_pool_destroy(pool);
    return 0;
}
