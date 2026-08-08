#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace offline {

inline constexpr uint16_t kMinBitness = 8;
inline constexpr uint16_t kMaxBitness = 12;

struct Entry {
    std::vector<uint8_t> g;
    std::vector<uint8_t> fx;
    uint8_t min_depth;
    uint16_t min_size;
};

size_t TableBytes(uint16_t bitness);
size_t EntryBytes(uint16_t bitness);

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
    uint32_t entries_;
    uint16_t bitness_;
};

}  // namespace offline
