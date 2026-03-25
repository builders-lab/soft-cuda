#include "internal_header.h"
#include "vector"

using namespace std;

void assignBackend(execution_node_t *e) {
    device_type dt = assignDevice(e->t->ndims, e->t->dims, e->t->op);
    if(dt == device_type::CPU) {
        e->backend_fn = tensor_evaluate;
        e->device_ptr = NULL;
    }
    // TODO: Build Logic for GPU part
}

// TODO: Implement CONFIG.soft parser and assignment on the basis of that
/* NOTE: Remember that when op is  tensor_op_t::NONE you assign it to CPU,
 *       The reason being that it would the dangling node.
 */
device_type assignDevice([[maybe_unused]]uint8_t ndims, [[maybe_unused]]uint32_t *dims, [[maybe_unused]]tensor_op_t op) {
    return device_type::CPU;
}


int32_t getTheExecutionNodeIndex(uint32_t id, vector<execution_node_t*> &nodes) {
    for(auto &node: nodes) {
        if(node->t->id == id) {
            
          return (int32_t)(node->pos);
        }
    }
    return -1;
}




void assignBackendGraph(vector<execution_node_t*> &nodes) {
    for(auto node : nodes) {
        assignBackend(node);
    }
    for(auto node : nodes) {
        if (node->backend_fn == tensor_evaluate_GPU) {
            // TODO: Implement contagious logic
            // TODO: How to get access to the execution_node place in graph when we just know the tensor
            if (node->t->a == NULL) goto trp;
            if (node->t->a->device == device_type::CPU) {
                int32_t a_idx = getTheExecutionNodeIndex(node->t->a->id, nodes);
                assert(a_idx != -1);
                nodes[((size_t)a_idx)]->to_device_needed = true;
            }
trp:
            if (node->t->b == NULL) continue;
            if (node->t->b->device == device_type::CPU) {
                int32_t b_idx = getTheExecutionNodeIndex(node->t->b->id, nodes);
                assert(b_idx != -1);
                nodes[((size_t)b_idx)]->to_device_needed = true;
            }
        }
    }
    // Yea no thrid check would be done after the eval graph had been learn so we can just see the concistency of the eval
    // for(auto node : nodes) {
    //     assert(backend_fn != NULL);
    //     if(backend_fn == tensor_evaluate_GPU) {
    //         assert(node->a->)
    //     }
    // }

}
