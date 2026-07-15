#pragma once

#include <stddef.h>

#include <string>
#include <vector>

// Discovers AIG circuits under circuits/ and computes their value samples.
// The aiger library and the internal circuit structures stay private to aig.cpp.

namespace gen::aig {

// Sorted circuit set names discovered under circuits/.
const std::vector<std::string>& CircuitSets();

// Sorted circuit case names for the given set.
const std::vector<std::string>& CircuitCases(const char* set_name);

// Number of inputs (primary inputs + latches) for a circuit case.
size_t CircuitInputs(const char* set_name, const char* case_name);

// Number of outputs (primary outputs + bad-state signals) for a circuit case.
size_t CircuitOutputs(const char* set_name, const char* case_name);

// Computes value samples for an AIG circuit into a thread_local buffer and
// returns it. The layout matches gen_circuit_value (see generator.h).
const char* CircuitValue(const char* set_name, const char* case_name, const char* input_state);

}  // namespace gen::aig
