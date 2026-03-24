#pragma once

struct execution_node {
    
    // Pointer to the tensor we are gonna op on
    *tensor_t t;

    // backend_id It's gonna be a function pointer instead more of handrolled vtable
    void (*backend_fn)(tensor_t*, tensor_t*, tensor_t*);

    // Pointer to device VRAM alloc, NULL if not needed
    void* device_ptr;
    
    // Boolean flag storing weather it will need to be transfered based upon reading the childs OPS
    bool to_device_needed;

    // Position in array storing cause could be useful
    uint32_t pos;
};


bool verifyIfDAG(tensor_t *t, std::vector<execution_node> &seq);
