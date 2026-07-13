#include "generator.h"

#include "aig.h"

#include <cassert>
#include <utility>

namespace {

    void WriteBits(const std::vector<std::vector<bool>> &data, float *output) {
        assert(output != nullptr);
        size_t offset = 0;
        for (const std::vector<bool> &row: data) {
            for (bool value: row) {
                output[offset++] = value ? 1.0f : -1.0f;
            }
        }
    }

} // namespace

namespace gen {

template<typename Tag>
BitMatrix<Tag>::BitMatrix(std::vector<std::vector<bool>> data) : data_(std::move(data)) {
    assert(!data_.empty());
    assert(!data_.front().empty());
    for (const std::vector<bool> &row: data_) {
        assert(row.size() == data_.front().size());
    }
}

template<typename Tag>
BitMatrix<Tag> BitMatrix<Tag>::Concat(std::vector<BitMatrix> chunks) {
    assert(!chunks.empty());
    const size_t columns = chunks.front().Columns();
    size_t rows = 0;
    for (const BitMatrix &chunk: chunks) {
        assert(chunk.Columns() == columns);
        rows += chunk.Rows();
    }

    std::vector<std::vector<bool>> data;
    data.reserve(rows);
    for (BitMatrix &chunk: chunks) {
        for (std::vector<bool> &row: chunk.data_) {
            data.push_back(std::move(row));
        }
    }
    return BitMatrix(std::move(data));
}

template<typename Tag>
void BitMatrix<Tag>::WriteValues(float *output) const { WriteBits(data_, output); }

template class BitMatrix<ValuesTag>;
template class BitMatrix<RestrictionsTag>;

const std::vector<std::string> &CircuitSets() { return ::CircuitSets(); }

const std::vector<std::string> &CircuitCases(const std::string &set_name) { return ::CircuitCases(set_name.c_str()); }

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
