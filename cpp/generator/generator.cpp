#include "generator.h"

#include "aig.h"

namespace gen {

const std::string &CircuitSets() { return ::CircuitSets(); }

const std::string &CircuitCases(const std::string &set_name) { return ::CircuitCases(set_name.c_str()); }

size_t CircuitInputs(const std::string &set_name, const std::string &case_name) {
    return ::CircuitInputs(set_name.c_str(), case_name.c_str());
}

size_t CircuitOutputs(const std::string &set_name, const std::string &case_name) {
    return ::CircuitOutputs(set_name.c_str(), case_name.c_str());
}

std::string CircuitValue(const std::string &set_name, const std::string &case_name, const std::string &input_state) {
    return ::CircuitValue(set_name.c_str(), case_name.c_str(), input_state.c_str());
}

} // namespace gen
