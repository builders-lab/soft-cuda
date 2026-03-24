#include "internal_header.h"
#include "vector"

// Actually let's try something better
bool verifyIfDAG(tensor_t *t, std::vector<execution_node> &seq) {
    if (t->stateTracker == 1) {
        return false; // It's not a fucking DAG bitch user 
    }
    if (t->stateTracker == 2) {
        return true; // Processed this node clean
    }
    
    t->stateTracker = 1;
    if(t->a != NULL) {
        if(!verifyIfDAG(t->a, seq)) return false;
    }
    if(t->b != NULL) {
        if(!verifyIfDAG(t->b, seq)) return false;
    }

    t->stateTracker = 2;
    
    execution_node en;
    en.t = t;
    en.pos = (uint32_t)seq.size();
    seq.push_back(en);
    return true;
}

