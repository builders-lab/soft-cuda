
#include "internal_header.h"
#include "graph/DAGbuild.h"
#include "vector"
#include <cstdint>
#include <unordered_map>
// Actually let's try something better
bool verifyIfDAG(tensor_pool_t *pool, tensor_t *t, std::vector<execution_node_t *> &seq) {
    if (t->stateTracker == 1) {
        return false; // It's not a fucking DAG bitch
    }
    if (t->stateTracker == 2) {
        return true; // Processed this node clean
    }

    t->stateTracker = 1;
    if (t->a != NULL) {
        if (!verifyIfDAG(pool, t->a, seq))
            return false;
    }
    if (t->b != NULL) {
        if (!verifyIfDAG(pool, t->b, seq))
            return false;
    }

    t->stateTracker = 2;

    uint32_t id;
    execution_node_t *en =
        (execution_node_t *)tensor_pool_alloc(pool, sizeof(execution_node_t), &id);
    en->t = t;
    en->pos = (uint32_t)seq.size();
    en->id = id;
    en->to_device_needed = false;
    en->device_ptr = NULL;
    en->device_ptr_grad = NULL;
    en->backend_fn = NULL;
    en->parent_pos[0] = -1;
    en->parent_pos[1] = -1;
    seq.push_back(en);
    return true;
}

void setUpParentReference( std::vector<execution_node_t *> &nodes ) {
    std::unordered_map<uint32_t, int32_t> id_to_pos;
    for(auto node : nodes) {
        id_to_pos[node->t->id] = (int32_t)node->pos;
    }
    for(auto node : nodes) {
        if(node->t->a != NULL) {
            node->parent_pos[0] = id_to_pos[node->t->a->id];
        } else {
            node->parent_pos[0] = -1;
        }

        if(node->t->b != NULL) {
            node->parent_pos[1] = id_to_pos[node->t->b->id];
        } else {
            node->parent_pos[1] = -1;
        }
    }
}


int32_t getPosOfNode(execution_node_t *et) { return (int32_t)(et->pos); }
