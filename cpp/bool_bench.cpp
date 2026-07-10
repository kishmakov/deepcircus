#include "aig.h"
#include "bool_bench.h"

// API

const char* bb_circuit_sets() {
    return CircuitSets().c_str();
}

const char* bb_circuit_cases(const char* set_name) {
    return CircuitCases(set_name).c_str();
}

size_t bb_circuit_inputs(const char* set_name, const char* case_name) {
    return CircuitInputs(set_name, case_name);
}

size_t bb_circuit_outputs(const char* set_name, const char* case_name) {
    return CircuitOutputs(set_name, case_name);
}

const char* bb_circuit_value(const char* set_name, const char* case_name, const char* input_state) {
    return CircuitValue(set_name, case_name, input_state);
}
