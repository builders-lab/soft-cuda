#include "soft-cuda/tensor/api.h"
#include "soft-cuda/tensor/debug_api.h"
#include <iostream>
#include <vector>

using namespace std;

int main() {
    tensor_pool_t *pool = tensor_pool_create(1024 * 1024);
    assert(pool != NULL);

    cout << "=========================================== \n";
    cout << "=============TESTING SUBTRACTION============== \n";
    float val_a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float val_b[] = {1.0f, 2.0f, 3.0f, 4.0f};

    uint32_t dims_a[] = {2, 2};
    uint32_t dims_b[] = {2, 2};

    tensor_t *a = tensor_create(pool, tensor_dtype_t::FLOAT32_T, 2, dims_a, val_a);
    tensor_t *b = tensor_create(pool, tensor_dtype_t::FLOAT32_T, 2, dims_b, val_b);
    
    tensor_t *c = tensor_mean(pool, a);
    cout << "=========================================== \n";
    cout << "==============LAZY EVAL DONE=============== \n";
    bool ok = tensor_evaluate(pool, c);
    assert(ok);
    std::cout << "============================================= \n";
    std::cout << "===============ORIGINAL TENSOR=============== \n";
    tensor_print_data(a);
    std::cout << "============================================= \n";
    std::cout << "===============SQUARED TENSOR=============== \n";
    // tensor_print_data(c);
    float* data = (float*)tensor_get_data(c);
    std::cout << data[0] << "\n";

    std::cout << "============================================= \n";
    // debug("result[0] = %f\n", result[0]);
    // debug("result[1] = %f\n", result[1]);
    tensor_pool_destroy(pool);
    return 0;
}
