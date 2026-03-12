#include <stdlib.h>
#include "string.h"
#include "internal_header.h"


/////////////////////////////////////////////////
// PUBLIC METHODS - GET VALUE
// Return a scalar value(tensor?) with the value of the tensor

inline float tensor_float32_value(tensor_t *t) {
    assert(t != NULL);
    assert(t->dtype == tensor_dtype_t::FLOAT32_T);
    assert(tensor_is_scalar(t));
    return *((float *)t->data);
}


