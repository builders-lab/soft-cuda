#pragma once
#include <vector>
#include "../third_party/json.hpp"

using json = nlohmann::json;


void assignBackend(execution_node_t *e);

device_type assignDevice(uint8_t ndims, uint32_t *dims, tensor_op_t op, uint32_t nvalues, json &data);
// returns -1 if no such parent
int32_t getTheExecutionNodeIndex(execution_node_t *node, uint32_t idx);

void assignBackendGraph(tensor_pool_t *pool, std::vector<execution_node_t *> &nodes);


void assignGradMemory(tensor_pool_t *pool_grad_cpu, tensor_pool_t *pool_grad_gpu, std::vector<execution_node_t *> &nodes);
