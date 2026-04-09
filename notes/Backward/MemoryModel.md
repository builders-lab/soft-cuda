## Memory model for backward pass(Unstable)

The problem we have is that when we do hybrid forward pass, it causes some data to be on GPU and some to be on CPU, this I thought
would be a problem but that is not the case.

### Observation
> Since data flow is only one directional and due to our contagious logic

```
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


```
We can conclude that all the data we need for grad computation is available on each device. and since we copy the parents we can 
confidently say that we don't need to retransfer tensors inbetween eliminating the thrashing.

* So here comes the part where what we will do in
`void assignBackendGraph(tensor_pool_t *pool,std::vector<execution_node_t *> &nodes);` 
is we will allocate the grad space for tensors on GPU as `pool_gpu_grad` and on CPU as `pool_cpu_grad` then we assign it to the home tensor(refering to the current tensor)
then during backward pass data is assigned to each pool and is written as it is.
since we don't want transfer overhead during computation path we will construct another data transfering fucnction which will get grad data from gpu to host.
and then we do optimization and then free resources.

Also the device_ptr_gpu is stored in execution_node_t;

===================================================================================================

actually i think we can't do grad placement in the AssignBackendGraph function which we are implementing the contagious logic we will need a 4th pass i think where so we can do this maneuvour

P1->GPU, P2->GPU, C1->GPU

but now if P1 and P2 was bought to GPU and it's copy exist on cpu we can jump to it's DRAM storage and i think since it's address we will know due to well how we can detect it you may ask it would be if it have address on execution node and also on tensor instance right then we switch to CPU allocation right.

Now for this to work we will need to walk reverse in graph so one more reason for 4th seperate pass plus this also helps that we will not be touching much of things.

Now if we are already doing 4th pass it would be better to instead have a seperate explict function for this so func params also don't get bloated and confusing.
