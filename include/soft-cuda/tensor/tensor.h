#pragme once

void tensor_sgd(std::vector<execution_node_t *> &nodes, float learning_rate);

bool verifyIfDAG(tensor_pool_t *pool, tensor_t *t, std::vector<execution_node_t *> &seq);

void assignBackendGraph(tensor_pool_t *pool,std::vector<execution_node_t *> &nodes, backend_mode value = backend_mode::CPU);


void assignGradMemory(tensor_pool_t *pool_grad_cpu, tensor_pool_t *pool_grad_gpu, std::vector<execution_node_t *> &nodes);


bool tensor_graph_forward_evaluate(tensor_pool_t *pool_cpu, tensor_pool_t *pool_gpu, std::vector<execution_node_t *> &nodes);


void gradInitializer(std::vector<execution_node_t *> &nodes);


bool tensor_graph_backward(std::vector<execution_node_t *> &nodes);


void assignGradMemory(tensor_pool_t *pool_grad_cpu, tensor_pool_t *pool_grad_gpu, std::vector<execution_node_t *> &nodes);


void autogradGpuMemTranfer(std::vector<execution_node_t *> &nodes);


bool save_model(const std::string& filepath, const std::vector<tensor_t*>& weights);


bool load_model(const std::string& filepath, const std::vector<tensor_t*>& weights);
