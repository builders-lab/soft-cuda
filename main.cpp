#include "soft-cuda/tensor/api.h"
#include "soft-cuda/tensor/debug_api.h"
#include "iostream"
#include "vector"

using namespace std;

int main() {
    // Create pool
    tensor_pool_t *pool = tensor_pool_create(1024 * 1024);
    tensor_pool_t *pool_2 = tensor_pool_create(1024 * 1024);
    tensor_pool_t *pool_gpu = tensor_pool_create(1024 * 1024, true);
    assert(pool != NULL);
    assert(pool_gpu != NULL);

    cout << "=========================================== \n";
    cout << "=============TESTING DAG VERIFICATION============== \n";
    float val_a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float val_b[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float val_c[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float val_d[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float val_e[] = {7.0f, 8.0f};

    uint32_t dims_a[] = {3, 2};
    uint32_t dims_b[] = {3, 2};
    uint32_t dims_c[] = {3, 2};
    uint32_t dims_d[] = {3, 2};
    uint32_t dims_e[] = {1, 2};

    tensor_t *a = tensor_create(pool, tensor_dtype_t::FLOAT32_T, 2, dims_a, val_a);
    tensor_t *b = tensor_create(pool, tensor_dtype_t::FLOAT32_T, 2, dims_b, val_b);
    tensor_t *c = tensor_create(pool, tensor_dtype_t::FLOAT32_T, 2, dims_c, val_c);
    tensor_t *d = tensor_create(pool, tensor_dtype_t::FLOAT32_T, 2, dims_d, val_d);
    tensor_t *e = tensor_create(pool, tensor_dtype_t::FLOAT32_T, 2, dims_e, val_e);
    
    tensor_t *f =tensor_add(pool, a,b);
    tensor_t *g =tensor_add(pool, b,c);
    tensor_t *h =tensor_add(pool, e,f);
    tensor_t *i =tensor_add(pool, h,a);
    tensor_t *j =tensor_add(pool, i,g);

    cout << "=========================================== \n";
    cout << "==============LAZY EVAL DONE=============== \n";
    
    std::vector<execution_node_t*> seq;
    bool oki = verifyIfDAG(pool_2,j, seq);
    assignBackendGraph(pool_gpu, seq);
    tensor_graph_forward_evaluate(pool, pool_gpu, seq);
    cout << oki << endl;
    // tensor_print_data(c);
    if (oki) {
        for(int i = 0; i < seq.size(); i++) {
             printExecutionNode(seq[i]);
        }
    }

    tensor_pool_destroy(pool);
    tensor_pool_destroy(pool_2);
    return 0;
}
