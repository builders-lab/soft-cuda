#include "internal_header.h"
// #include "vector"
#include <iostream>
#include <cuda_runtime.h>

using json = nlohmann::json;
void assignBackend(execution_node_t *e, json &data) {
    device_type dt = assignDevice(e->t->ndims, e->t->dims, e->t->op, e->t->nvalues, data);
    if (dt == device_type::CPU) {
        e->backend_fn = tensor_evaluate;
        e->device_ptr = NULL;
    } else {

        // TODO: Build Logic for GPU part
        e->backend_fn = tensor_evaluate_GPU;
        e->device_ptr = NULL; // pre-allocation happens in Pass 2
    }
}

// TODO: Implement CONFIG.soft parser and assignment on the basis of that
/* NOTE: Remember that when op is  tensor_op_t::NONE you assign it to CPU,
 *       The reason being that it would the dangling node.
 */
device_type assignDevice([[maybe_unused]] uint8_t ndims, [[maybe_unused]] uint32_t *dims,
                         [[maybe_unused]] tensor_op_t op, uint32_t nvalues, json &data) {
    if(op == tensor_op_t::NONE) {
        return device_type::CPU;
    }
    try {

        if (op == tensor_op_t::MUL_MATRIX) {
            auto params = data["ops"]["matmul"];
            for(auto param: params) {
                if(param["min"] <= nvalues && param["max"] > nvalues) {
                    if(param["backend"] == "cpu") {
                        return device_type::CPU;
                    } else {
                        return device_type::GPU;
                    }
                }
            }
        }
        if (op == tensor_op_t::ADD) {
            auto params = data["ops"]["add"];
            for(auto param: params) {
                if(!param.contains("min") || param["min"] <= nvalues && param["max"] > nvalues) {
                    if(param["backend"] == "cpu") {
                        return device_type::CPU;
                    } else {
                        return device_type::GPU;
                    }
                }
            }
        }
    } catch (const std::exception& e) {
       debug("Error: %s\n", e.what());
    }
    return device_type::GPU;
}

int32_t getTheExecutionNodeIndex(execution_node_t *node, uint32_t idx) {
    return node->parent_pos[idx];
}


void assignPlaceOnDeviceMemory(tensor_pool_t *pool, int32_t a_idx, std::vector<execution_node_t *> &nodes) {
    if (nodes[(size_t)a_idx]->device_ptr != NULL) {
        return; // Already allocated, retreat!
    }

    // TODO: support for multiple types for now we just assume we are dealing with float32_t
    //
    size_t size = nodes[(size_t)a_idx]->t->nvalues * sizeof(float) ;
    uint32_t id;
    nodes[(size_t)a_idx]->device_ptr = tensor_pool_alloc(pool, size, &id);

}

void assignBackendGraph(tensor_pool_t *pool,std::vector<execution_node_t *> &nodes) {
    setUpParentReference(nodes);
    json data = readJsonToMap("/home/wslarch/Documents/Coding/DEV/soft/soft-cuda/src/init/config/CONFIG.soft");
    for (auto node : nodes) {
        assignBackend(node, data);
    }

    for (auto node : nodes) {
        if (node->backend_fn == tensor_evaluate_GPU) {
            // TODO: Implement contagious logic
            // TODO: How to get access to the execution_node place in graph when we just know the
            // tensor
            assignPlaceOnDeviceMemory(pool, (int32_t)(node->pos), nodes);
            if (node->t->a == NULL)
                goto trp;
            if (node->t->a->device == device_type::CPU) {
                int32_t a_idx = getTheExecutionNodeIndex(node, 0);
                assert(a_idx != -1);
                nodes[((size_t)a_idx)]->to_device_needed = true;
                assignPlaceOnDeviceMemory(pool, a_idx, nodes);
            }
        trp:
            if (node->t->b == NULL)
                continue;
            if (node->t->b->device == device_type::CPU) {
                int32_t b_idx = getTheExecutionNodeIndex(node, 1);
                assert(b_idx != -1);
                nodes[((size_t)b_idx)]->to_device_needed = true;
                assignPlaceOnDeviceMemory(pool, b_idx, nodes);
            }
        }
    }
    // Yea no thrid check would be done after the eval graph had been learn so we can just see the
    // concistency of the eval
    for (auto node : nodes) {
        assert(node->backend_fn != NULL);
        if (node->backend_fn == tensor_evaluate_GPU) {
            if (node->t->a != NULL) {
                int32_t a_idx = getTheExecutionNodeIndex(node,0);
                assert(nodes[(size_t)a_idx]->to_device_needed == true ||
                       nodes[(size_t)a_idx]->t->device == device_type::GPU);
            }
            if (node->t->b != NULL) {
                int32_t b_idx = getTheExecutionNodeIndex(node, 1);
                assert(nodes[(size_t)b_idx]->to_device_needed == true ||
                       nodes[(size_t)b_idx]->t->device == device_type::GPU);
            }
        }
    }
}

bool tensor_graph_forward_evaluate(tensor_pool_t *pool_cpu, tensor_pool_t *pool_gpu, std::vector<execution_node_t *> &nodes) {
    for(auto node : nodes) {
        if(node->backend_fn == tensor_evaluate_GPU){
            // TODO: Write the memory copy for parents when child have this as well as evaluating it
            // NOTE: We have to check maybe parents are already on GPU
            int32_t a_idx = getTheExecutionNodeIndex(node, 0);
            if(a_idx != -1){
                auto parent_a = nodes[(size_t)a_idx];

                // NOTE: we are totally ignoring the to_device_needed flag am i missing something which i thought i needed.
                // TODO: Review this part once again maybe try for streams for better performance or whatever
                if (parent_a->t->device == device_type::CPU) {
                    size_t size_a = nodes[(size_t)a_idx]->t->nvalues * sizeof(float) ;
                    cudaError_t err_a = cudaMemcpy(parent_a->device_ptr, parent_a->t->data, size_a, cudaMemcpyHostToDevice);
                    if (err_a != cudaSuccess) {
                        debug("CRITICAL FAILURE: Parent A transfer failed! %s\n", cudaGetErrorString(err_a));
                    }
                    parent_a->t->device = device_type::GPU;
                }
            }
            int32_t b_idx = getTheExecutionNodeIndex(node, 1);
            if(b_idx != -1){
                auto parent_b = nodes[(size_t)b_idx];
                if (parent_b->t->device == device_type::CPU) {
                    size_t size_b = nodes[(size_t)b_idx]->t->nvalues * sizeof(float) ;
                    cudaError_t err_b = cudaMemcpy(parent_b->device_ptr, parent_b->t->data, size_b, cudaMemcpyHostToDevice);
                    if (err_b != cudaSuccess) {
                        debug("CRITICAL FAILURE: Parent B transfer failed! %s\n", cudaGetErrorString(err_b));
                    }
                    parent_b->t->device = device_type::GPU;
                }
            }
            // (*(node->backend_fn))(pool_gpu, node->t);
            // bool (*gpu_backend_fn)(tensor_t *t, float *d_a, float *d_b, float *d_res);
            float *d_a = NULL;
            if (a_idx != -1) d_a = (float*)(nodes[(size_t)a_idx]->device_ptr);
            float *d_b = NULL;
            if (b_idx != -1) d_b = (float*)(nodes[(size_t)b_idx]->device_ptr);
            float *d_out = (float*)(node->device_ptr);

            (*(node->backend_fn))(pool_cpu, node->t, d_a, d_b, d_out );
            node->t->device = device_type::GPU;
            // device_type device;
        } else if (node->backend_fn == tensor_evaluate) {
            float *dummy = nullptr;
            (*(node->backend_fn))(pool_cpu, node->t, dummy, dummy, dummy);
        }
    }
    return true;
}


void printExecutionNode(execution_node_t *et) {
    std::cout << et->pos << "\n";
    std::cout << et->id << "\n";
    tensor_print_data(et->t);
    std::cout << et->to_device_needed << "\n";
    std::cout << et->device_ptr << "\n";
    std::cout << et->device_ptr_grad << "\n";
    std::cout << (et->backend_fn != NULL) << "\n";
}


bool execution_node_to_host(execution_node_t *node) {
    if (node == NULL || node->device_ptr == NULL || node->t == NULL) {
        return false;
    }

    if (node->t->device == device_type::GPU) {
        size_t size = node->t->nvalues * sizeof(float);
        
        cudaError_t err = cudaMemcpy(node->t->data, node->device_ptr, size, cudaMemcpyDeviceToHost);
        
        if (err == cudaSuccess) {
            node->t->device = device_type::CPU; 
        } else {
            debug("CUDA Extraction Failed: %s\n", cudaGetErrorString(err));
            return false;
        }
    }
    return true;
}

// What we can do is keep it default and then have user declare leaf to NULL. As otherwise it would
// be difficult to implement the propogation logic
// NOTE: Since we are making the memory here and device_ptr_grad is here to what we will do is take
// the whole execution node and then pass it to the eagar OPS and then we will see if it does have
// `device_ptr_grad` if so then it's fine right other wise we will directly write to
// node->t->grad->data right. Then after all the ops we will make a new function which will scan the
// whole graph and then if it sees the device_ptr_grad is not null it will initiate data transfer.
void assignGradMemory(tensor_pool_t *pool_grad_cpu, tensor_pool_t *pool_grad_gpu, std::vector<execution_node_t *> &nodes) {
    for(auto &node : nodes) {
        // We are not assigning grad instance during tensor create.
        // Maybe we can do that but I feel it might cause recursion problem even if with assigning base case of leaf nodes
        // there is the problem that during creation time it could cause function call overhead and whatnot.
        // So we are going to make a tensor instance here and assign it here i guess. 
        
        if(node->t->grad_compute == false) continue;

        node->t->grad =  tensor_dtype_create(pool_grad_cpu, node->t->dtype, node->t->ndims, node->t->dims, NULL);

        if(node->backend_fn == tensor_evaluate_GPU) {
            size_t size = node->t->nvalues * sizeof(float) ;
            uint32_t id;
            node->device_ptr_grad = tensor_pool_alloc(pool_grad_gpu, size, &id);
        }
    }
}
