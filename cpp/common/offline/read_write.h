#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace offline {

inline constexpr uint16_t kMinBitness = 8;
inline constexpr uint16_t kMaxBitness = UINT8_MAX;

inline constexpr uint8_t kUnknownDepth = UINT8_MAX;
inline constexpr uint16_t kUnknownSize = UINT16_MAX;

enum class FunctionKind : uint8_t {
    kTable = 0,
    kTree = 1,
    kTreeOverTable = 2,
};

struct Function {
    FunctionKind kind;
    std::vector<uint8_t> payload;
};

struct Entry {
    Function g;
    Function f;
    uint8_t min_depth;
    uint16_t min_size;

    bool TargetKnown() const { return min_depth != kUnknownDepth; }
};

uint64_t EntryBytes(const Entry& entry);

class Writer {
public:
    Writer(const std::string& path, uint32_t entries, uint16_t bitness);
    ~Writer();

    Writer(const Writer&) = delete;
    Writer& operator=(const Writer&) = delete;

    void Write(const Entry& entry);

    uint32_t Entries() const { return entries_; }
    uint16_t Bitness() const { return bitness_; }

private:
    std::ofstream output_;
    std::vector<uint64_t> offsets_;
    uint32_t entries_;
    uint32_t entries_written_ = 0;
    uint16_t bitness_;
};

class Reader {
public:
    explicit Reader(const std::string& path);

    Reader(const Reader&) = delete;
    Reader& operator=(const Reader&) = delete;

    Entry Read(uint32_t index);

    uint32_t Entries() const { return entries_; }
    uint16_t Bitness() const { return bitness_; }

private:
    std::ifstream input_;
    std::vector<uint64_t> offsets_;
    uint64_t file_bytes_;
    uint32_t entries_;
    uint16_t bitness_;
};

}  // namespace offline
