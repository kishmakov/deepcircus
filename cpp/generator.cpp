#include "aig.h"
#include "generator.h"

// API

const char* gen_circuit_sets() {
    return CircuitSets().c_str();
}

const char* gen_circuit_cases(const char* set_name) {
    return CircuitCases(set_name).c_str();
}

size_t gen_circuit_inputs(const char* set_name, const char* case_name) {
    return CircuitInputs(set_name, case_name);
}

size_t gen_circuit_outputs(const char* set_name, const char* case_name) {
    return CircuitOutputs(set_name, case_name);
}

const char* gen_circuit_value(const char* set_name, const char* case_name, const char* input_state) {
    return CircuitValue(set_name, case_name, input_state);
}
