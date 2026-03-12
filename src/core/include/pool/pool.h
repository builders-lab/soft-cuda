#pragma once

struct tensor_pool_instance {
    size_t memsize;
    size_t memused;
    uint32_t nallocs;
    void *mem;
};
