#pragma once
#include <vector>

void assignBackend(execution_node_t *e);

device_type assignDevice(uint8_t ndims, uint32_t *dims, tensor_op_t op);

int32_t getTheExecutionNodeIndex(uint32_t id, std::vector<execution_node_t *> &nodes);

void assignBackendGraph(std::vector<execution_node_t *> &nodes);
