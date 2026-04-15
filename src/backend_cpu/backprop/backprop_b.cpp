#include "internal_header.h"

// We are gonna write a backprop fucntion for matmul cause that is easy thing to do
// Since our backprop is an eagar evaluation we will make tensor objects inside the function
// transpose it execute it as well and then work with it

// Now I am slightly confused but I think it would have gone something like this
// Since we are going to get execution node we could have a decision made here if it's data will be
// written on device_ptr of node_.grad->data
// We are gonna do this cause making decsion on higher level will require us to make two kernels for similar work.

// Since this will be eagar evalution we are gonna execute it here now and we are gonna return a
// boolean.
// @return: boolean
// @params: execution_node_t
// Since all the memory is already executed we just need the execution node
bool backprop__(std::vector<execution_node_t *> &nodes) {

    for(auto &node : nodes) {
        if (node->t->grad_compute) {
            if (node->device_ptr_grad != NULL) {
                bool success = backprop_gpu(node);
                return success;
            } else {
                bool success = backprop_cpu(node);
                return sucess;
            }
        }
    }
}

bool backprop_cpu(execution_node_t *node) {

    assert(node->t != NULL);
    bool success = false;

    switch (t->op) {
    case tensor_op_t::NONE:
        success = true;
        break;
    case tensor_op_t::CAST:
        break;
        // TODO: Implement here
    case tensor_op_t::MUL_MATRIX:
        assert(t->a != NULL);
        assert(t->b != NULL);
        assert(t->a->dtype == t->b->dtype);
        success = tensor_mul_grad_op_matrix(pool, t);
        break;
    case tensor_op_t::MUL_SCALAR:
        assert(t->a != NULL);
        assert(t->b != NULL);
        assert(t->a->dtype == t->b->dtype);
        success = tensor_mul_grad_op_scalar(pool, t);
        break;
    case tensor_op_t::TRANSPOSE:
        assert(t->a != NULL);
        success = tensor_tranpose_grad_op_matrix(pool, t);
        break;
    case tensor_op_t::NAIVE_MATRIX_MUL:
        assert(t->a != NULL);
        assert(t->b != NULL);
        assert(t->a->dtype == t->b->dtype);
        success = tensor_mul_grad_op_matrix_naive(pool, t);
        break;
    case tensor_op_t::ADD:
        assert(t->a != NULL);
        assert(t->b != NULL);
        assert(t->a->dtype == t->b->dtype);
        success = tensor_grad_op_add(pool, t);
        break;
    case tensor_op_t::BROADCAST_ADD:
        assert(t->a != NULL);
        assert(t->b != NULL);
        assert(t->a->dtype == t->b->dtype);
        success = tensor_grad_op_broadcasting_add(pool, t);
        break;
    case tensor_op_t::RELU:
        assert(t->a != NULL);
        success = tensor_grad_op_relu(pool, t);
        break;
    case tensor_op_t::SUB:
        assert(t->a != NULL);
        assert(t->b != NULL);
        assert(t->a->dtype == t->b->dtype);
        success = tensor_grad_op_sub(pool, t);
        break;
    case tensor_op_t::MEAN:
        assert(t->a != NULL);
        success = tensor_grad_op_mean(pool, t);
        break;
    default:
        assert(false);
    }
    if (success) {
        debug("backprop_cpu: success\n");
    } else {
        debug("backprop_cpu: FUBAR\n");
    }
    return success;
}

/*************************************************************/
/*************************************************************/
/*************************************************************/
// TODO: Implement GPU module
bool backprop_gpu(execution_node_t *node) {
    assert(node->t != NULL);
    bool success = false;

    switch (node->t->op) {
    case tensor_op_t::NONE:
        success = true;
        break;
    case tensor_op_t::CAST:
        break;
        // TODO: Implement here
    case tensor_op_t::ADD:
        assert(t->a != NULL);
        assert(t->b != NULL);
        assert(t->a->dtype == t->b->dtype);
        success = tensor_grad_add_op_cuda(t, d_a, d_b, d_res);
        break;
    case tensor_op_t::MUL_MATRIX:
        assert(t->a != NULL);
        assert(t->b != NULL);
        assert(t->a->dtype == t->b->dtype);
        success = tensor_grad_mul_op_cuda(t, d_a, d_b, d_res);
        break;
    default:
        assert(false);
    }
    if (success) {
        debug("backprop_gpu: success\n");
    } else {
        debug("backprop_gpu: FUBAR\n");
    }
    return success;
}
