#pragma once

#include <stddef.h>

#include <string>

// Discovers AIG circuits under circuits/ and computes their value samples.
// The aiger library and the internal circuit structures stay private to aig.cpp.

// Sorted newline-separated circuit set names discovered under circuits/.
const std::string& CircuitSets();

// Sorted newline-separated circuit case names for the given set.
const std::string& CircuitCases(const char* set_name);

// Number of inputs (primary inputs + latches) for a circuit case.
size_t CircuitInputs(const char* set_name, const char* case_name);

// Number of outputs (primary outputs + bad-state signals) for a circuit case.
size_t CircuitOutputs(const char* set_name, const char* case_name);

// Computes value samples for an AIG circuit into a thread_local buffer and
// returns it. The layout matches bb_circuit_value (see bool_bench.h).
const char* CircuitValue(
    const char* set_name,
    const char* case_name,
    const char* input_state);
