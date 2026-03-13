#pragma once

struct tensor_pool_instance {
    size_t memsize;
    size_t memused;
    uint32_t nallocs;
    void *mem;
};

tensor_pool_t *tensor_pool_create(size_t memsize);
void tensor_pool_zero(tensor_pool_t *pool);
void tensor_pool_destroy(tensor_pool_t *pool);
void *tensor_pool_alloc(tensor_pool_t *pool, size_t size, uint32_t *id);
inline size_t tensor_pool_size(tensor_pool_t *pool);
inline size_t tensor_pool_used(tensor_pool_t *pool);
