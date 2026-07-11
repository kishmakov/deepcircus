#include "libgen.h"

#include "generator.h"

#include <cassert>
#include <string>

extern "C" {

size_t gen_tree_cases_number(uint16_t bitness) { return gen::TreeCasesNumber(bitness); }

const char *gen_tree_value(uint16_t bitness, size_t case_id, const char *input) {
    assert(input != nullptr);
    thread_local std::string value;
    value = gen::TreeValue(bitness, case_id, input);
    return value.c_str();
}

uint16_t gen_table_solvable_bitness() { return gen::TableSolvableBitness(); }

const char *gen_table_value(uint16_t bitness, size_t case_id, const char *input) {
    assert(input != nullptr);
    thread_local std::string value;
    value = gen::TableValue(bitness, case_id, input);
    return value.c_str();
}

const char *gen_circuit_sets() { return gen::CircuitSets().c_str(); }

const char *gen_circuit_cases(const char *set_name) {
    assert(set_name != nullptr);
    return gen::CircuitCases(set_name).c_str();
}

size_t gen_circuit_inputs(const char *set_name, const char *case_name) {
    assert(set_name != nullptr && case_name != nullptr);
    return gen::CircuitInputs(set_name, case_name);
}

size_t gen_circuit_outputs(const char *set_name, const char *case_name) {
    assert(set_name != nullptr && case_name != nullptr);
    return gen::CircuitOutputs(set_name, case_name);
}

const char *gen_circuit_value(const char *set_name, const char *case_name, const char *input_state) {
    assert(set_name != nullptr && case_name != nullptr && input_state != nullptr);
    thread_local std::string value;
    value = gen::CircuitValue(set_name, case_name, input_state);
    return value.c_str();
}

} // extern "C"
