#include "generator.h"

#include <cassert>
#include <cstring>
#include <utility>

#include "aig.h"

namespace gen {

template <typename Tag>
BitMatrix<Tag>::BitMatrix(std::vector<std::vector<bool>> data) : data_(std::move(data)) {
    assert(!data_.empty());
    assert(!data_.front().empty());
    for (const std::vector<bool>& row : data_) {
        assert(row.size() == data_.front().size());
    }
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

    std::vector<std::vector<bool>> data;
    data.reserve(rows);
    for (BitMatrix& chunk : chunks) {
        for (std::vector<bool>& row : chunk.data_) {
            data.push_back(std::move(row));
        }
    }
    return BitMatrix(std::move(data));
}

template <typename Tag>
void BitMatrix<Tag>::WritePacked(uint8_t* output) const {
    assert(output != nullptr);
    std::memset(output, 0, ByteCount());
    const size_t row_bytes = RowBytes();
    for (size_t row = 0; row < data_.size(); ++row) {
        uint8_t* bytes = output + row * row_bytes;
        const std::vector<bool>& bits = data_[row];
        for (size_t bit = 0; bit < bits.size(); ++bit) {
            bytes[bit / 8] |= static_cast<uint8_t>(bits[bit]) << (bit % 8);
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

const std::vector<std::string>& CircuitSets() { return ::CircuitSets(); }

const std::vector<std::string>& CircuitCases(const std::string& set_name) { return ::CircuitCases(set_name.c_str()); }

size_t CircuitInputs(const std::string& set_name, const std::string& case_name) {
    return ::CircuitInputs(set_name.c_str(), case_name.c_str());
}

size_t CircuitOutputs(const std::string& set_name, const std::string& case_name) {
    return ::CircuitOutputs(set_name.c_str(), case_name.c_str());
}

std::string CircuitValue(const std::string& set_name, const std::string& case_name, const std::string& input_state) {
    return ::CircuitValue(set_name.c_str(), case_name.c_str(), input_state.c_str());
}

}  // namespace gen
