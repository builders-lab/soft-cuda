`tensor_pool_create(size, is_device)` — one flag, two paths, abstraction complete
`tensor_instance.device` — ground truth, where data lives right now
`execution_node` — plan metadata, transfer flag, VRAM ptr, backend func ptr
Transfer is one-way CPU→GPU, checked via device field before stamping flag
Three pools: CPU tensors, GPU tensors, graph/execution metadata

```C
struct tensor_instance {
    
    tensor_dtype_t dtype;
    
    // Np. of dimension and values
    uint8_t ndims;
    uint32_t nvalues;

    // Number of element in each dimension
    uint32_t dims[TENSOR_MAX_DIMS + 1];
    
    uint32_t stride[TENSOR_MAX_DIMS];
    
    uint32_t broadcast_stride[TENSOR_MAX_DIMS];
    // Data
    void *data;

    // Operation and arguments
    tensor_op_t op;
    tensor_t *a;
    tensor_t *b;
    
    // ID and name
    uint32_t id;
    
    // Tensor aware of it's postion could be useful
    device_type device; 
    
    // If autograd is required
    bool grad_compute;

    // If tensor is transposed
    bool is_transposed;

    // For DFS computation
    bool evaluated;

    // For storing autograd result
    tensor_t *grad;
    
    //Device ID
    // uint8_t backend_id; // We are not gonna do this in tensor it self
}


```
```C

struct execution_node {
    
    // Pointer to the tensor we are gonna op on
    *tensor_t t;

    // backend_id It's gonna be a function pointer instead more of handrolled vtable
    void* backend_if_function();
    
    // Pointer to memory on VRAM, NULL if not needed
    void* tensor_pool_t;
    
    // Boolean flag storing weather it will need to be transfered based upon reading the childs OPS
    bool to_device_needed;

    // Position in array storing cause could be useful
    uint32_t pos;
}
```

What we are gonna do is during compilation graph building part we will do 3 passes on the topo sorted array

Linearly walk over it assign the `backend_if_function` then once again walk over it and assign `to_device_needed` flag and then walk over it once more verify it's correctness cause why not.

The we assign for `to_device_needed` flag by first reading the pos of tensor then if it's on HOST then we assign `to_device_needed` call otherwise it's not assigned as prev it was handelled. 

Two invarients

<!-- GPU is first class model so if we have A, B, C and X, Y depend on A, B and B, C respectively and X required -->
<!-- GPU while Y required CPU -->

Yea don't need this thought cause CPU already would have a copy of B hence no need for that GPU first class
invairent in that context.

It's invarient in that if of the 3 A, B and C if even one on GPU we will just transfer everything to GPU as 
bringing to GPU to CPU is not worth it actually not even that is probelm cause GPU nope it's a problem so 
when we need something on GPU it can actually exist on just VRAM hence yea GPU is first class definition.

If even on leaf or parent is GPU op we just transfer everything to GPU.

There is also this assumption that once data in GPU there is no need to call it bach till we need it for just showing it back to user.


our invarient would be this

```pseudo-code
for each execution_node in sorted order:
    if node.backend_fn is CUDA:
        if node.t->a->device == CPU:
            execution_node[pos of a].to_device_needed = true
        if node.t->b->device == CPU:
            execution_node[pos of b].to_device_needed = true
```

our third check would be this

```pseudo-code
for each execution_node:
    assert(backend_fn != NULL)
    if backend_fn is CUDA:
        assert(input->device == CUDA || execution_node[input->pos].to_device_needed == true)
    assert(input->pos < current pos)
```
