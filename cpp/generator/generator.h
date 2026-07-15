#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace gen {

// Dense, bit-packed matrix. Rows are cases and columns contain all generated
// repetitions (and, for restrictions, all restrictions) for one case. Tagged
// so value and restriction matrices stay distinct types.
template <typename Tag>
class BitMatrix {
public:
    explicit BitMatrix(std::vector<std::vector<bool>> data);

    static BitMatrix Concat(std::vector<BitMatrix> chunks);

    size_t Rows() const { return data_.size(); }
    size_t Columns() const { return data_.front().size(); }
    size_t ValueCount() const { return Rows() * Columns(); }
    size_t RowBytes() const { return (Columns() + 7) / 8; }
    size_t ByteCount() const { return Rows() * RowBytes(); }
    // Bit-packs the matrix, each row padded to a whole number of bytes and
    // filled little-endian bit order; bit b unpacks to the value 2*b - 1.
    void WritePacked(uint8_t* output) const;

private:
    std::vector<std::vector<bool>> data_;
};

struct ValuesTag;
struct RestrictionsTag;

using Values = BitMatrix<ValuesTag>;
using Restrictions = BitMatrix<RestrictionsTag>;

// Inverse of BitMatrix::WritePacked: expands `rows` byte-padded rows of
// `columns` little-endian bits into floats 2*b - 1, row-major into output.
void UnpackRows(const uint8_t* packed, size_t rows, size_t columns, float* output);

// Exact targets per case, interleaved: (bitness - depth, log2(2^bitness - size)),
// where size counts internal nodes. Both score 0 for the worst case (full
// tree) and bitness for the best (constant function).
inline constexpr size_t kTargetsPerCase = 2;

struct GeneratedValues {
    Values values;
    std::vector<float> targets;
};

struct GeneratedRestrictions {
    Values values;
    Restrictions restrictions;
};

/********************************* tree **************************************/

size_t TreeCasesNumber(uint16_t bitness);

// Input: 0/1 string of length bitness. Output length: 2 * bitness + 1.
std::string TreeValue(uint16_t bitness, size_t case_id, uint64_t seed, std::string_view input);

// Deterministic, chunk-order-independent case-id sample: splitting the result
// into contiguous groups and generating each with TreeValuesForCases
// reproduces exactly what a single call over the full list would produce.
std::vector<size_t> TreeSampleCaseIds(uint16_t bitness, size_t cases, uint64_t seed);

// Synchronously generates a compact, ready-to-read batch for an explicit,
// pre-sampled chunk of case ids (see TreeSampleCaseIds).
GeneratedValues TreeValuesForCases(uint16_t bitness, const std::vector<size_t>& case_ids, size_t reps, uint64_t seed);

/********************************* table *************************************/

uint16_t TableSolvableBitness();
size_t TableCasesNumber(uint16_t bitness);

// Input: 0/1 string of length bitness. Output length: 2 * bitness + 1.
std::string TableValue(uint16_t bitness, size_t case_id, uint64_t seed, std::string_view input);

std::vector<size_t> TableSampleCaseIds(uint16_t bitness, size_t cases, uint64_t seed);

GeneratedValues TableValuesForCases(uint16_t bitness, const std::vector<size_t>& case_ids, size_t reps, uint64_t seed);

// Synchronously generates recursive table values and one dense restriction
// matrix for an explicit, pre-sampled chunk of case ids.
GeneratedRestrictions TableRestrictionsForCases(uint16_t bitness, const std::vector<size_t>& case_ids, size_t reps,
                                                uint64_t seed);

/******************************** circuit ************************************/

// Sorted circuit set names discovered under circuits/.
const std::vector<std::string>& CircuitSets();

// Sorted circuit case names for the given set.
const std::vector<std::string>& CircuitCases(const std::string& set_name);

// Number of inputs (primary inputs + latches) for a circuit case.
size_t CircuitInputs(const std::string& set_name, const std::string& case_name);

// Number of outputs (primary outputs + bad-state signals) for a circuit case.
size_t CircuitOutputs(const std::string& set_name, const std::string& case_name);

// Value samples for an AIG circuit as a 0/1 string.
std::string CircuitValue(const std::string& set_name, const std::string& case_name, const std::string& input_state);

}  // namespace gen
