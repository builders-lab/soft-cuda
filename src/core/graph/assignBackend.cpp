#include "internal_header.h"

void assignBackend(execution_node *e) {
    device_type dt = assignDevice(e->t->ndims, e->t->dims, e->t->op);
    if(dt == device_type::CPU) {
        e->backend_fn = tensor_evaluate;
        e->device_ptr = NULL;
    }
    // TODO: Build Logic for GPU part
}

// TODO: Implement CONFIG.soft parser and assignment on the basis of that
device_type assignDevice(uint8_t ndims, uint32_t *dims, tensor_op_t op) {
    return device_type::CPU;
}
