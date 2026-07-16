#pragma once

// Minimal C ABI for direct golden-value checks. C++ callers should use gen::.

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/********************************* tree **************************************/

size_t gen_tree_cases_number(uint16_t bitness);

/********************************* table *************************************/

uint16_t gen_table_solvable_bitness();

// Input: 0/1 string of length bitness. Output length: 2 * bitness + 1.
const char* gen_table_value(uint16_t bitness, size_t case_id, uint64_t seed, const char* input);

/******************************** circuit ************************************/

const char* gen_circuit_sets();
const char* gen_circuit_cases(const char* set_name);
size_t gen_circuit_inputs(const char* set_name, const char* case_name);
size_t gen_circuit_outputs(const char* set_name, const char* case_name);
const char* gen_circuit_value(const char* set_name, const char* case_name, const char* input_state);

#ifdef __cplusplus
}
#endif
