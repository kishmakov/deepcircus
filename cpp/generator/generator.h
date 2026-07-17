#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gen {

// Dense, bit-packed matrix. Rows are cases and columns contain all generated
// repetitions (and, for restrictions, all restrictions) for one case. The bits
// live in one contiguous, row-major buffer; Rows()/Columns() are the only
// helpers that split it into independent rows. Tagged so value and restriction
// matrices stay distinct types.
template <typename Tag>
class BitMatrix {
public:
    BitMatrix(size_t rows, size_t columns, std::vector<bool> data);

    static BitMatrix Concat(std::vector<BitMatrix> chunks);

    size_t Rows() const { return rows_; }
    size_t Columns() const { return columns_; }
    size_t ValueCount() const { return rows_ * columns_; }
    size_t RowBytes() const { return (columns_ + 7) / 8; }
    size_t ByteCount() const { return rows_ * RowBytes(); }
    // Bit-packs the matrix, each row padded to a whole number of bytes and
    // filled little-endian bit order; bit b unpacks to the value 2*b - 1.
    void WritePacked(uint8_t* output) const;

private:
    size_t rows_;
    size_t columns_;
    std::vector<bool> data_;
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

// Input sampling shape shared by the tree and table value batches: `batches`
// independent samplings, each expanded into `batch_size` (power-of-two) points.
struct InputShape {
    uint16_t batches;
    uint16_t batch_size;
};

// Expands one base sequence per batch into the block-and-random input walk
// shared with case sampling: batch 0 follows the Gray-code NextSequence walk,
// each later batch flips bit groups keyed by (batch, point id). `sequences`
// must hold shape.batches equal-length sequences; returns batches * batch_size
// points, each sequence-length bits, concatenated.
std::vector<bool> ExpandInputs(InputShape shape, const std::vector<std::vector<bool>>& sequences);

/********************************* tree **************************************/

size_t TreeCasesNumber(uint16_t bitness);

// Input: bitness bits. Output length: 2 * bitness + 1.
std::string TreeValue(uint16_t bitness, size_t case_id, const std::vector<bool>& input);

// Deterministic, chunk-order-independent case-id sample: splitting the result
// into contiguous groups and generating each with TreeValuesForCases
// reproduces exactly what a single call over the full list would produce.
std::vector<size_t> TreeSampleCaseIds(uint16_t bitness, size_t cases, uint64_t seed);

// Synchronously generates a compact, ready-to-read batch for an explicit,
// pre-sampled chunk of case ids (see TreeSampleCaseIds).
GeneratedValues TreeValuesForCases(uint16_t bitness, const std::vector<size_t>& case_ids, InputShape shape);

/********************************* table *************************************/

uint16_t TableSolvableBitness();
size_t TableCasesNumber(uint16_t bitness);

// Input: bitness bits. Output length: 2 * bitness + 1.
std::string TableValue(uint16_t bitness, size_t case_id, const std::vector<bool>& input);

std::vector<size_t> TableSampleCaseIds(uint16_t bitness, size_t cases, uint64_t seed);

GeneratedValues TableValuesForCases(uint16_t bitness, const std::vector<size_t>& case_ids, InputShape shape);

// Synchronously generates recursive table values and one dense restriction
// matrix for an explicit, pre-sampled chunk of case ids.
GeneratedRestrictions TableRestrictionsForCases(uint16_t bitness, const std::vector<size_t>& case_ids, InputShape shape);

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
