#include "internal_header.h"
#include "vector"
#include <iostream>

void assignBackend(execution_node_t *e) {
    device_type dt = assignDevice(e->t->ndims, e->t->dims, e->t->op);
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
                         [[maybe_unused]] tensor_op_t op) {
    return device_type::GPU;
}

int32_t getTheExecutionNodeIndex(uint32_t id, std::vector<execution_node_t *> &nodes) {
    for (auto &node : nodes) {
        if (node->t->id == id) {

            return (int32_t)(node->pos);
        }
    }
    return -1;
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
    for (auto node : nodes) {
        assignBackend(node);
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
                int32_t a_idx = getTheExecutionNodeIndex(node->t->a->id, nodes);
                assert(a_idx != -1);
                nodes[((size_t)a_idx)]->to_device_needed = true;
                assignPlaceOnDeviceMemory(pool, a_idx, nodes);
            }
        trp:
            if (node->t->b == NULL)
                continue;
            if (node->t->b->device == device_type::CPU) {
                int32_t b_idx = getTheExecutionNodeIndex(node->t->b->id, nodes);
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
                int32_t a_idx = getTheExecutionNodeIndex(node->t->a->id, nodes);
                assert(nodes[(size_t)a_idx]->to_device_needed == true ||
                       nodes[(size_t)a_idx]->t->device == device_type::GPU);
            }
            if (node->t->b != NULL) {
                int32_t b_idx = getTheExecutionNodeIndex(node->t->b->id, nodes);
                assert(nodes[(size_t)b_idx]->to_device_needed == true ||
                       nodes[(size_t)b_idx]->t->device == device_type::GPU);
            }
        }
    }
}

void printExecutionNode(execution_node_t *et) {
    std::cout << et->pos << "\n";
    std::cout << et->id << "\n";
    tensor_print_data(et->t);
    std::cout << et->to_device_needed << "\n";
    std::cout << et->device_ptr << "\n";
    std::cout << (et->backend_fn != NULL) << "\n";
}
