#include <stdlib.h>
#include "internal_header.h"


#define TENSOR_POOL_ALIGN 8UL

// Creates a continuous pool of memory using bump allocator pattern
tensor_pool_t *tensor_pool_create(size_t memsize, bool isOfDevice) {
    assert(memsize);

    tensor_pool_t *pool = (tensor_pool_t *)malloc(sizeof(tensor_pool_t));
    if(pool == NULL) {
        return NULL;
    }
    // cudaError_t cudaError;
    pool->memsize = memsize;
    pool->memused = 0;
    pool->nallocs = 0;
    pool->isOfDevice = isOfDevice;
    if (isOfDevice) {
      cudaError_t cudaError = cudaMalloc(&pool->mem,memsize);
    } else {
      pool->mem = malloc(memsize);
    }
    // (void *)cudaError;
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
size_t tensor_pool_size(tensor_pool_t *pool) {
    assert(pool);
    return pool->memsize;
}

// Return used bytes of memory pool
size_t tensor_pool_used(tensor_pool_t *pool) {
    assert(pool);
    return pool->memused;
}
