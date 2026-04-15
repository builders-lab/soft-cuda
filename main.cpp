#include "soft-cuda/tensor/api.h"
#include "soft-cuda/tensor/debug_api.h"
#include <iostream>
#include <vector>
#include <cassert>

using namespace std;

int main() {
    // CPU ALLOCATION
    tensor_pool_t *pool = tensor_pool_create(1024 * 1024);
    tensor_pool_t *pool_meta = tensor_pool_create(1024 * 1024);
    tensor_pool_t *pool_grad_cpu = tensor_pool_create(1024 * 1024);


    // GPU ALLOCATION
    tensor_pool_t *pool_gpu = tensor_pool_create(1024 * 1024, true);
    tensor_pool_t *pool_grad_gpu = tensor_pool_create(1024 * 1024, true);
    
    assert(pool != NULL);
    assert(pool_gpu != NULL);
    
    cout << "=========================================== \n";
    cout << "============= XOR IMPLEMENTATION  ============== \n";
    
    float val_X[8]{0,0,0,1,1,0,1,1};
    float val_Y[4]{0,1,1,0};
    float val_W1[4]{};
    float val_W2[2]{};
    float val_b1[2]{};
    float val_b2[1]{};
    uint32_t dims_X[]  = {4,2};
    uint32_t dims_Y[]  = {4,1};
    uint32_t dims_W1[] = {2, 2};
    uint32_t dims_b1[] = {1, 2};
    uint32_t dims_b2[] = {1, 1};
    uint32_t dims_W2[] = {2, 1};
    
    tensor_t *X  = tensor_create(pool, tensor_dtype_t::FLOAT32_T, 2, dims_X,  val_X);
    tensor_t *Y  = tensor_create(pool, tensor_dtype_t::FLOAT32_T, 2, dims_Y,  val_Y);
    tensor_t *W1 = tensor_create(pool, tensor_dtype_t::FLOAT32_T, 2, dims_W1, val_W1);
    tensor_t *W2 = tensor_create(pool, tensor_dtype_t::FLOAT32_T, 2, dims_W2, val_W2);
    tensor_t *b1 = tensor_create(pool, tensor_dtype_t::FLOAT32_T, 2, dims_b1, val_b1);
    tensor_t *b2 = tensor_create(pool, tensor_dtype_t::FLOAT32_T, 2, dims_b2, val_b2);

    tensor_fill_random_normal(W1, 0, 0.01);
    tensor_fill_random_normal(W2, 0.1, 0.01);
    tensor_fill_random_normal(b1, 0.1, 0.01);
    tensor_fill_random_normal(b2, 0.11, 0.01);
    
    cout << "=========================================== \n";
    cout << "============== DEFINING OPS =============== \n";
    
    tensor_t *H_pred_bef = tensor_mul_naive(pool, X, W1);
    tensor_t *H_pred = tensor_add(pool, H_pred_bef, b1);
    tensor_t *H = tensor_relu(pool, H_pred);
    tensor_t *Y_pred_bef = tensor_mul_naive(pool, H, W2);
    tensor_t *Y_pred = tensor_add(pool, Y_pred_bef, b2);
    tensor_t *L_sub = tensor_sub(pool, Y_pred, Y);
    tensor_t *L_squ = tensor_square(pool, L_sub);
    tensor_t *mse = tensor_mean(pool, L_squ);
    cout << "=========================================== \n";
    cout << "============== LAZY EVAL DONE =============== \n";
    

    cout << "=========================================== \n";
    cout << "============== PREPARING THE KITCHEN =============== \n";
    std::vector<execution_node_t*> seq;
    bool oki = verifyIfDAG(pool_meta, mse, seq);
    assignBackendGraph(pool_gpu, seq);
    assignGradMemory(pool_grad_cpu, pool_grad_gpu, seq);
    
    
    if (oki) {
    cout << "=========================================== \n";
    cout << "============== STARTING TRAINING =============== \n";
        for (int i = 0; i < 10000; i++) {
            tensor_graph_forward_evaluate(pool, pool_gpu, seq);
            gradInitializer(seq);
            tensor_graph_backward(seq);
            if (i % 1000 == 0) {
                std::cout << i << "EPOCH";
                execution_node_t *mse = seq.back();
                printExecutionNode(mse);
            }
            tensor_sgd(seq, 0.01);
        }
    } else {
        cout << "WARNING: DAG Verification failed. Aborting expedition.\n";
    }
     
    tensor_pool_destroy(pool);
    tensor_pool_destroy(pool_meta);
    tensor_pool_destroy(pool_grad_cpu);
    tensor_pool_destroy(pool_grad_gpu);
    tensor_pool_destroy(pool_gpu);
    
    return 0;
}
