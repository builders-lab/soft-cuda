#include "internal_header.h"
#include "vector"
#include "graph/DAGbuild.h"

// Actually let's try something better
bool verifyIfDAG(tensor_pool_t *pool, tensor_t *t, std::vector<execution_node_t*> &seq) {
    if (t->stateTracker == 1) {
        return false; // It's not a fucking DAG bitch 
    }
    if (t->stateTracker == 2) {
        return true; // Processed this node clean
    }
    
    t->stateTracker = 1;
    if(t->a != NULL) {
        if(!verifyIfDAG(pool,t->a, seq)) return false;
    }
    if(t->b != NULL) {
        if(!verifyIfDAG(pool,t->b, seq)) return false;
    }

    t->stateTracker = 2;
    
    uint32_t id;
    execution_node_t *en = (execution_node_t *)tensor_pool_alloc(pool, sizeof(execution_node_t), &id);
    en->t = t;
    en->pos = (uint32_t)seq.size();
    en->id = id;
    seq.push_back(en);
    return true;
}

int32_t getPosOfNode(execution_node_t *et) {
    return (int32_t)(et->pos);
}
