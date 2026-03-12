#pragma once

inline float tensor_float32_value(tensor_t *t) {
    assert(t != NULL);
    assert(t->dtype == tensor_dtype_t::FLOAT32_T);
    assert(tensor_is_scalar(t));
    return *((float *)t->data);
}


