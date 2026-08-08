#include "offline/read_write.h"

#include <cassert>
#include <filesystem>
#include <ios>
#include <limits>
#include <ostream>

namespace offline {
namespace {

constexpr size_t kHeaderBytes = sizeof(uint32_t) + sizeof(uint8_t);

void WriteUint8(std::ostream& output, uint8_t value) { output.put(static_cast<char>(value)); }

void WriteUint16(std::ostream& output, uint16_t value) {
    WriteUint8(output, value & 0xff);
    WriteUint8(output, value >> 8);
}

void WriteUint32(std::ostream& output, uint32_t value) {
    for (size_t byte = 0; byte < sizeof(value); ++byte) {
        WriteUint8(output, value >> (8 * byte));
    }
}

uint8_t ReadUint8(std::istream& input) {
    const int value = input.get();
    assert(value != std::char_traits<char>::eof());
    return static_cast<uint8_t>(value);
}

uint16_t ReadUint16(std::istream& input) {
    uint16_t value = 0;
    for (size_t byte = 0; byte < sizeof(value); ++byte) {
        value |= uint16_t{ReadUint8(input)} << (8 * byte);
    }
    return value;
}

uint32_t ReadUint32(std::istream& input) {
    uint32_t value = 0;
    for (size_t byte = 0; byte < sizeof(value); ++byte) {
        value |= uint32_t{ReadUint8(input)} << (8 * byte);
    }
    return value;
}

void ReadBytes(std::istream& input, std::vector<uint8_t>& bytes) {
    input.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    assert(input.good());
}

}  // namespace

size_t TableBytes(uint16_t bitness) {
    assert(bitness >= kMinBitness && bitness <= kMaxBitness);
    return size_t{1} << (bitness - 3);
}

size_t EntryBytes(uint16_t bitness) { return 2 * TableBytes(bitness) + sizeof(uint8_t) + sizeof(uint16_t); }

Writer::Writer(const std::string& path, uint32_t entries, uint16_t bitness)
    : output_(path, std::ios::binary | std::ios::trunc), entries_(entries), bitness_(bitness) {
    assert(output_.is_open());
    TableBytes(bitness_);
    WriteUint32(output_, entries_);
    WriteUint8(output_, static_cast<uint8_t>(bitness_));
    assert(output_.good());
}

Writer::~Writer() {
    assert(entries_written_ == entries_);
    output_.flush();
    assert(output_.good());
}

void Writer::Write(const Entry& entry) {
    assert(entries_written_ < entries_);
    const size_t table_bytes = TableBytes(bitness_);
    assert(entry.g.size() == table_bytes);
    assert(entry.fx.size() == table_bytes);
    assert(entry.min_depth <= bitness_);
    assert(entry.min_size < (uint32_t{1} << bitness_));

    output_.write(reinterpret_cast<const char*>(entry.g.data()), entry.g.size());
    output_.write(reinterpret_cast<const char*>(entry.fx.data()), entry.fx.size());
    WriteUint8(output_, entry.min_depth);
    WriteUint16(output_, entry.min_size);
    assert(output_.good());
    ++entries_written_;
}

Reader::Reader(const std::string& path) : input_(path, std::ios::binary), entries_(0), bitness_(0) {
    assert(input_.is_open());
    entries_ = ReadUint32(input_);
    bitness_ = ReadUint8(input_);

    const uintmax_t expected_size = kHeaderBytes + uintmax_t{entries_} * EntryBytes(bitness_);
    assert(std::filesystem::file_size(path) == expected_size);
}

Entry Reader::Read(uint32_t index) {
    assert(index < entries_);
    const uintmax_t offset = kHeaderBytes + uintmax_t{index} * EntryBytes(bitness_);
    assert(offset <= static_cast<uintmax_t>(std::numeric_limits<std::streamoff>::max()));
    input_.seekg(static_cast<std::streamoff>(offset));
    assert(input_.good());

    const size_t table_bytes = TableBytes(bitness_);
    Entry entry{std::vector<uint8_t>(table_bytes), std::vector<uint8_t>(table_bytes), 0, 0};
    ReadBytes(input_, entry.g);
    ReadBytes(input_, entry.fx);
    entry.min_depth = ReadUint8(input_);
    entry.min_size = ReadUint16(input_);
    assert(entry.min_depth <= bitness_);
    assert(entry.min_size < (uint32_t{1} << bitness_));
    return entry;
}

}  // namespace offline
