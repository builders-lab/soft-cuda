#include "internal_header.h"
#include "vector"

// Actually let's try something better
bool verifyIfDAG(tensor_t *t, std::vector<execution_node> &seq) {
    if (t->stateTracker == 1) {
        return false; // It's a fucking DAG bitch user 
    }
    if (t->stateTracker == 2) {
        return true; // Processed this node clean
    }
    
    t->stateTracker = 1;
    if(t->a != NULL) {
        verifyIfDAG(t->a, seq);
    }
    if(t->b != NULL) {
        verifyIfDAG(t->b, seq);
    }

    t->stateTracker = 2;
    
    // TODO: Construct execution_node and add it to the vector
}
