#include <stdlib.h>
#include "internal_header.h"


#define TENSOR_POOL_ALIGN 8
// TODO: tensor pool destroy
// TODO: tensor pool alloc
// TODO: tensor pool size
// TODO: tensor pool used
// TODO: tensor pool remaining
//
// struct tensor_pool_instance {
//     size_t memsize;
//     size_t memused;
//     uint32_t nallocs;
//     void *mem;
// };
tensor_pool_t *tensor_pool_create(uint32_t memsize) {
    assert(memsize);

    tensor_pool_t *pool = (tensor_pool_t *)malloc(sizeof(tensor_pool_t));
    if(pool == NULL) {
        return NULL;
    }

    pool->memsize = memsize;
    pool->memused = 0;
    pool->nallocs = 0;
    pool->mem = malloc(memsize);
    if (pool->mem == NULL) {
        free(pool);
        return NULL;
    }
    return pool;
}

// Free resources from a tensor pool, but don't destroy the pool
void tensor_pool_zero(tensor_pool_t *pool) {
    assert(pool);
    assert(pool->mem);
    
    // Reset memory
    pool->memused = 0;
    pool->nallocs = 0;
}

// Free resources from tensor pool
void tensor_pool_destroy(tensor_pool_t *pool) {
    debug("tensor_pool_destroy: pool size=%zu used=%zu space", tensor_pool_size(pool), tensor_pool_used(pool));
    if (pool) {
        tensor_pool_zero(pool);
        free(pool->mem);
        free(pool);
    }
} 

// Allocate bytes on the pool, return NULL if memory exhausted
void *tensor_pool_alloc(tensor_pool_t *pool, size_t size, uint32_t *id) {
    assert(pool);
    assert(size);
    
    // Align size on boundary
    size = (size + TENSOR_POOL_ALIGN - 1) & ~(TENSOR_POOL_ALIGN - 1);
    if (pool->memused + size > pool->memsize) {
        debug("tensor_pool_alloc: memory exhausted, size=%zu, used=%zu, total=%zu", size, pool->memused, pool->memsize);
        return NULL;
    }
    
    void *ptr = (uint8_t*)pool->mem + pool->memused;
    pool->memused += size;

    // Set id if not NULL, and increase the nallocs
    if (id != NULL) {
        *id = ++pool->nallocs;
    }

    return ptr;
}

// Return size of memory pool
inline size_t tensor_pool_size(tensor_pool_t *pool) {
    assert(pool);
    return pool->memsize;
}

// Return used bytes of memory pool
inline size_t tensor_pool_used(tensor_pool_t *pool) {
    assert(pool);
    return pool->memused;
}
