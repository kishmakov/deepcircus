#pragma once

#include <cstddef>
#include <cstdint>

#include <string>
#include <string_view>
#include <vector>

namespace gen {

// Bit-packed +/-1 restriction tensor. Values stay packed until a caller
// materializes them into a provided float buffer.
class Tensor {
public:
    Tensor(uint16_t bitness, size_t cases, size_t reps, std::vector<bool> values);

    uint16_t Bitness() const { return bitness_; }
    size_t Cases() const { return cases_; }
    size_t Reps() const { return reps_; }
    size_t ValueCount() const { return values_.size(); }
    void WriteValues(float *output) const;

private:
    uint16_t bitness_;
    size_t cases_;
    size_t reps_;
    std::vector<bool> values_;
};

// Synchronously generated, ready-to-read batch. Values remain bit-packed and
// recursive table restrictions are complete and owned here.
class Data {
public:
    Data(uint16_t bitness, size_t cases, size_t reps, std::vector<bool> values, std::vector<float> targets,
         std::vector<Tensor> restrictions);

    uint16_t Bitness() const { return bitness_; }
    size_t Cases() const { return cases_; }
    size_t Reps() const { return reps_; }
    size_t ValueCount() const { return values_.size(); }
    size_t TargetCount() const { return targets_.size(); }
    void WriteValues(float *output) const;
    void WriteTargets(float *output) const;
    const std::vector<Tensor> &Restrictions() const { return restrictions_; }

private:
    uint16_t bitness_;
    size_t cases_;
    size_t reps_;
    std::vector<bool> values_;
    std::vector<float> targets_;
    std::vector<Tensor> restrictions_;
};

/********************************* tree **************************************/

uint16_t MinTreeBitness();
size_t TreeCasesNumber(uint16_t bitness);

// Input: 0/1 string of length bitness. Output length: 2 * bitness + 1.
std::string TreeValue(uint16_t bitness, size_t case_id, std::string_view input);

// Synchronously generates a compact, ready-to-read batch.
Data TreeValueTensor(uint16_t bitness, size_t cases, size_t reps, uint64_t seed);

/********************************* table *************************************/

uint16_t TableSolvableBitness();
size_t TableCasesNumber(uint16_t bitness);

// Input: 0/1 string of length bitness. Output length: 2 * bitness + 1.
std::string TableValue(uint16_t bitness, size_t case_id, std::string_view input);

// For recursive tables, all restriction chunks are generated synchronously
// and owned by the returned data.
Data TableValueTensor(uint16_t bitness, size_t cases, size_t reps, size_t restriction_chunk_cases, uint64_t seed);

/******************************** circuit ************************************/

// Sorted newline-separated circuit set names discovered under circuits/.
const std::string &CircuitSets();

// Sorted newline-separated circuit case names for the given set.
const std::string &CircuitCases(const std::string &set_name);

// Number of inputs (primary inputs + latches) for a circuit case.
size_t CircuitInputs(const std::string &set_name, const std::string &case_name);

// Number of outputs (primary outputs + bad-state signals) for a circuit case.
size_t CircuitOutputs(const std::string &set_name, const std::string &case_name);

// Value samples for an AIG circuit as a 0/1 string.
std::string CircuitValue(const std::string &set_name, const std::string &case_name, const std::string &input_state);

} // namespace gen
