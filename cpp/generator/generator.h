#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gen_data gen_data;
typedef struct gen_tensor gen_tensor;

uint16_t gen_data_bitness(const gen_data *data);
size_t gen_data_cases(const gen_data *data);
size_t gen_data_reps(const gen_data *data);
size_t gen_data_value_count(const gen_data *data);
size_t gen_data_target_count(const gen_data *data);
void gen_data_write_values(const gen_data *data, float *output);
void gen_data_write_targets(const gen_data *data, float *output);
size_t gen_data_restriction_count(const gen_data *data);
const gen_tensor *gen_data_restriction(const gen_data *data, size_t index);
void gen_data_destroy(gen_data *data);

uint16_t gen_tensor_bitness(const gen_tensor *tensor);
size_t gen_tensor_cases(const gen_tensor *tensor);
size_t gen_tensor_reps(const gen_tensor *tensor);
size_t gen_tensor_value_count(const gen_tensor *tensor);
void gen_tensor_write_values(const gen_tensor *tensor, float *output);

/********************************* tree **************************************/

uint16_t gen_min_tree_bitness();
size_t gen_tree_cases_number(uint16_t bitness);

// Input: 0/1 string of length bitness. Output length: 2 * bitness + 1.
const char *gen_tree_value(uint16_t bitness, size_t case_id, const char *input);

// Synchronously generates a compact, ready-to-read batch.
gen_data *gen_tree_value_tensor(uint16_t bitness, size_t cases, size_t reps, uint64_t seed);

/********************************* table *************************************/

uint16_t gen_table_solvable_bitness();
size_t gen_table_cases_number(uint16_t bitness);

// Input: 0/1 string of length bitness. Output length: 2 * bitness + 1.
const char *gen_table_value(uint16_t bitness, size_t case_id, const char *input);

// For recursive tables, all restriction chunks are generated synchronously
// and owned by the returned data handle.
gen_data *gen_table_value_tensor(uint16_t bitness, size_t cases, size_t reps, size_t restriction_chunk_cases,
                                 uint64_t seed);

/******************************** circuit ************************************/

const char *gen_circuit_sets();
const char *gen_circuit_cases(const char *set_name);
size_t gen_circuit_inputs(const char *set_name, const char *case_name);
size_t gen_circuit_outputs(const char *set_name, const char *case_name);
const char *gen_circuit_value(const char *set_name, const char *case_name, const char *input_state);

#ifdef __cplusplus
}
#endif
