#include "internal_header.h"

tensor_t *tensor_mse_loss(tensor_pool_t *pool, tensor_t *Y_pred, tensor_t *Y) {
    tensor_t *L_sub      = tensor_sub(pool, Y_pred, Y);
    tensor_t *L_squ      = tensor_square(pool, L_sub);
    tensor_t *mse        = tensor_mean(pool, L_squ);
    return mse;
}
