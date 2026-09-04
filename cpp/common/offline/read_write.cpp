#include "offline/read_write.h"

#include <cassert>
#include <filesystem>
#include <ios>
#include <limits>
#include <ostream>
#include <utility>
#include <vector>

namespace offline {
namespace {

constexpr size_t kHeaderBytes = sizeof(uint32_t) + sizeof(uint8_t);
constexpr size_t kFunctionHeaderBytes = sizeof(uint8_t) + sizeof(uint32_t);
constexpr size_t kTargetBytes = sizeof(uint8_t) + sizeof(uint16_t);

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

void WriteUint64(std::ostream& output, uint64_t value) {
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

uint64_t ReadUint64(std::istream& input) {
    uint64_t value = 0;
    for (size_t byte = 0; byte < sizeof(value); ++byte) {
        value |= uint64_t{ReadUint8(input)} << (8 * byte);
    }
    return value;
}

void ReadBytes(std::istream& input, std::vector<uint8_t>& bytes) {
    input.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    assert(input.good());
}

uint64_t Position(std::streampos position) {
    assert(position >= 0);
    return static_cast<uint64_t>(position);
}

void AssertFunctionKind(FunctionKind kind) {
    assert(kind == FunctionKind::kTable || kind == FunctionKind::kTree || kind == FunctionKind::kTreeOverTable);
}

void AssertTarget(const Entry& entry, uint16_t bitness) {
    const bool unknown_depth = entry.min_depth == kUnknownDepth;
    const bool unknown_size = entry.min_size == kUnknownSize;
    assert(unknown_depth == unknown_size);
    if (unknown_depth) return;

    assert(entry.min_depth <= bitness);
    if (bitness < std::numeric_limits<uint16_t>::digits) {
        assert(entry.min_size < (uint32_t{1} << bitness));
    }
}

void WriteFunction(std::ostream& output, const Function& function) {
    AssertFunctionKind(function.kind);
    assert(function.payload.size() <= UINT32_MAX);
    WriteUint8(output, static_cast<uint8_t>(function.kind));
    WriteUint32(output, static_cast<uint32_t>(function.payload.size()));
    output.write(reinterpret_cast<const char*>(function.payload.data()), function.payload.size());
}

Function ReadFunction(std::istream& input, uint64_t entry_end) {
    const auto kind = static_cast<FunctionKind>(ReadUint8(input));
    AssertFunctionKind(kind);
    const uint32_t size = ReadUint32(input);
    assert(Position(input.tellg()) + size <= entry_end);

    std::vector<uint8_t> payload(size);
    ReadBytes(input, payload);
    return Function{kind, std::move(payload)};
}

}  // namespace

uint64_t EntryBytes(const Entry& entry) {
    return 2 * kFunctionHeaderBytes + entry.g.payload.size() + entry.f.payload.size() + kTargetBytes;
}

Writer::Writer(const std::string& path, uint32_t entries, uint16_t bitness)
    : output_(path, std::ios::binary | std::ios::trunc), offsets_(entries), entries_(entries), bitness_(bitness) {
    assert(output_.is_open());
    assert(bitness_ >= kMinBitness && bitness_ <= kMaxBitness);
    WriteUint32(output_, entries_);
    WriteUint8(output_, static_cast<uint8_t>(bitness_));
    for (uint32_t index = 0; index < entries_; ++index) WriteUint64(output_, 0);
    assert(output_.good());
}

Writer::~Writer() {
    assert(entries_written_ == entries_);
    const std::streampos end = output_.tellp();
    output_.seekp(kHeaderBytes);
    assert(output_.good());
    for (const uint64_t offset : offsets_) WriteUint64(output_, offset);
    output_.seekp(end);
    output_.flush();
    assert(output_.good());
}

void Writer::Write(const Entry& entry) {
    assert(entries_written_ < entries_);
    AssertTarget(entry, bitness_);
    offsets_[entries_written_] = Position(output_.tellp());

    WriteFunction(output_, entry.g);
    WriteFunction(output_, entry.f);
    WriteUint8(output_, entry.min_depth);
    WriteUint16(output_, entry.min_size);
    assert(output_.good());
    ++entries_written_;
}

Reader::Reader(const std::string& path) : input_(path, std::ios::binary), file_bytes_(0), entries_(0), bitness_(0) {
    assert(input_.is_open());
    entries_ = ReadUint32(input_);
    bitness_ = ReadUint8(input_);
    assert(bitness_ >= kMinBitness && bitness_ <= kMaxBitness);

    offsets_.resize(entries_);
    for (uint64_t& offset : offsets_) offset = ReadUint64(input_);

    file_bytes_ = std::filesystem::file_size(path);
    const uint64_t entries_begin = kHeaderBytes + uint64_t{entries_} * sizeof(uint64_t);
    if (entries_ == 0) {
        assert(file_bytes_ == entries_begin);
        return;
    }

    assert(offsets_.front() == entries_begin);
    for (uint32_t index = 0; index < entries_; ++index) {
        assert(offsets_[index] < (index + 1 < entries_ ? offsets_[index + 1] : file_bytes_));
    }
}

Entry Reader::Read(uint32_t index) {
    assert(index < entries_);
    const uint64_t offset = offsets_[index];
    const uint64_t entry_end = index + 1 < entries_ ? offsets_[index + 1] : file_bytes_;
    assert(entry_end <= static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max()));
    input_.seekg(static_cast<std::streamoff>(offset));
    assert(input_.good());

    Entry entry{ReadFunction(input_, entry_end), ReadFunction(input_, entry_end), 0, 0};
    entry.min_depth = ReadUint8(input_);
    entry.min_size = ReadUint16(input_);
    assert(Position(input_.tellg()) == entry_end);
    AssertTarget(entry, bitness_);
    return entry;
}

}  // namespace offline
