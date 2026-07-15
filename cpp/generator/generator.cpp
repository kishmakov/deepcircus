#include "generator.h"

#include <cassert>
#include <cstring>
#include <utility>

#include "aig.h"

namespace gen {

template <typename Tag>
BitMatrix<Tag>::BitMatrix(size_t rows, size_t columns, std::vector<bool> data)
    : rows_(rows), columns_(columns), data_(std::move(data)) {
    assert(rows_ > 0);
    assert(columns_ > 0);
    assert(data_.size() == rows_ * columns_);
}

template <typename Tag>
BitMatrix<Tag> BitMatrix<Tag>::Concat(std::vector<BitMatrix> chunks) {
    assert(!chunks.empty());
    const size_t columns = chunks.front().Columns();
    size_t rows = 0;
    for (const BitMatrix& chunk : chunks) {
        assert(chunk.Columns() == columns);
        rows += chunk.Rows();
    }

    std::vector<bool> data;
    data.reserve(rows * columns);
    for (BitMatrix& chunk : chunks) {
        data.insert(data.end(), chunk.data_.begin(), chunk.data_.end());
    }
    return BitMatrix(rows, columns, std::move(data));
}

template <typename Tag>
void BitMatrix<Tag>::WritePacked(uint8_t* output) const {
    assert(output != nullptr);
    std::memset(output, 0, ByteCount());
    const size_t row_bytes = RowBytes();
    for (size_t row = 0; row < rows_; ++row) {
        uint8_t* bytes = output + row * row_bytes;
        const size_t base = row * columns_;
        for (size_t bit = 0; bit < columns_; ++bit) {
            bytes[bit / 8] |= static_cast<uint8_t>(data_[base + bit]) << (bit % 8);
        }
    }
}

template class BitMatrix<ValuesTag>;
template class BitMatrix<RestrictionsTag>;

void UnpackRows(const uint8_t* packed, size_t rows, size_t columns, float* output) {
    assert(packed != nullptr);
    assert(output != nullptr);
    const size_t row_bytes = (columns + 7) / 8;
    for (size_t row = 0; row < rows; ++row) {
        const uint8_t* bytes = packed + row * row_bytes;
        float* values = output + row * columns;
        for (size_t bit = 0; bit < columns; ++bit) {
            values[bit] = (bytes[bit / 8] >> (bit % 8)) & 1 ? 1.0f : -1.0f;
        }
    }
}

const std::vector<std::string>& CircuitSets() { return aig::CircuitSets(); }

const std::vector<std::string>& CircuitCases(const std::string& set_name) {
    return aig::CircuitCases(set_name.c_str());
}

size_t CircuitInputs(const std::string& set_name, const std::string& case_name) {
    return aig::CircuitInputs(set_name.c_str(), case_name.c_str());
}

size_t CircuitOutputs(const std::string& set_name, const std::string& case_name) {
    return aig::CircuitOutputs(set_name.c_str(), case_name.c_str());
}

std::string CircuitValue(const std::string& set_name, const std::string& case_name, const std::string& input_state) {
    return aig::CircuitValue(set_name.c_str(), case_name.c_str(), input_state.c_str());
}

}  // namespace gen
