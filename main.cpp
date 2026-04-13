#include "soft-cuda/tensor/api.h"
// #include "soft-cuda/tensor/debug_api.h"
#include <iostream>
#include <vector>

using namespace std;
int main() {
    // Create pools
    // CPU ALLOCATION
    tensor_pool_t *pool = tensor_pool_create(1024 * 1024);
    tensor_pool_t *pool_2 = tensor_pool_create(1024 * 1024);
    tensor_pool_t *pool_grad_cpu = tensor_pool_create(1024 * 1024);
    
    // GPU ALLOCATION
    tensor_pool_t *pool_gpu = tensor_pool_create(1024 * 1024, true);
    tensor_pool_t *pool_grad_gpu = tensor_pool_create(1024 * 1024, true);
    assert(pool != NULL);
    assert(pool_gpu != NULL);

    cout << "=========================================== \n";
    cout << "=============TESTING DAG VERIFICATION============== \n";
    
    float val_a[900]{};
    float val_b[900]{};
    float val_c[900]{};
    float val_d[900]{};
    
    // TACTICAL FIX: Make 'e' the exact same size to avoid a GPU segfault 
    // before we implement a dedicated GPU broadcasting kernel!
    float val_e[900]{};

    uint32_t dims_a[] = {30, 30};
    uint32_t dims_b[] = {30, 30};
    uint32_t dims_c[] = {30, 30};
    uint32_t dims_d[] = {30, 30};
    uint32_t dims_e[] = {30, 30}; // Matched dimensions!

    tensor_t *a = tensor_create(pool, tensor_dtype_t::FLOAT32_T, 2, dims_a, val_a, false);
    tensor_t *b = tensor_create(pool, tensor_dtype_t::FLOAT32_T, 2, dims_b, val_b, false);
    tensor_t *c = tensor_create(pool, tensor_dtype_t::FLOAT32_T, 2, dims_c, val_c, false);
    tensor_t *d = tensor_create(pool, tensor_dtype_t::FLOAT32_T, 2, dims_d, val_d, false);
    tensor_t *e = tensor_create(pool, tensor_dtype_t::FLOAT32_T, 2, dims_e, val_e, false);
    tensor_fill_random_normal(a, 10.1, 5.7);
    tensor_fill_random_normal(b, 10.1, 5.7);
    tensor_fill_random_normal(c, 10.1, 5.7);
    tensor_fill_random_normal(d, 10.1, 5.7);
    tensor_fill_random_normal(e, 10.1, 5.7);
    tensor_t *f = tensor_mul(pool, a, b);
    tensor_t *g = tensor_mul(pool, b, c);
    tensor_t *h = tensor_mul(pool, e, f);
    tensor_t *i = tensor_mul(pool, h, a);
    tensor_t *j = tensor_mul(pool, i, g);

    cout << "=========================================== \n";
    cout << "==============LAZY EVAL DONE=============== \n";
    
    std::vector<execution_node_t*> seq;
    bool oki = verifyIfDAG(pool_2, j, seq);
    
    // Ensure assignDevice is returning device_type::GPU under the hood!
    assignBackendGraph(pool_gpu, seq);

    assignGradMemory(pool_grad_cpu,pool_grad_gpu, seq);
    tensor_graph_forward_evaluate(pool, pool_gpu, seq);
    
    if (oki) {
        cout << "=========================================== \n";
        cout << "========= VRAM TRACE ============= \n";
        
            execution_node_t *node = seq.back();
            
                // Pull it back to the CPU
                execution_node_to_host(node);
                printExecutionNode(node);
                cout << "\n";
    }

    tensor_pool_destroy(pool);
    tensor_pool_destroy(pool_2);
    tensor_pool_destroy(pool_grad_cpu);
    tensor_pool_destroy(pool_grad_gpu);
    tensor_pool_destroy(pool_gpu);
    return 0;
}
