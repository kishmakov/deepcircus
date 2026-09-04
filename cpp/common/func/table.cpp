#include "func/table.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>

#include "tools/random.h"

namespace func {

namespace {

uint64_t DeserializeSeed(const std::vector<uint8_t>& bytes) {
    assert(bytes.size() == sizeof(uint64_t));

    uint64_t seed = 0;
    std::memcpy(&seed, bytes.data(), sizeof(seed));
    return seed;
}

}  // namespace

TableFunc::TableFunc(uint16_t bitness, uint64_t seed) : func::Func(bitness), seed_(seed) {
    assert(bitness_ >= kMinBitness && bitness_ <= kMaxBitness);
}

TableFunc::TableFunc(uint16_t bitness, std::vector<uint8_t> bytes)
    : func::Func(bitness), seed_(DeserializeSeed(bytes)) {
    assert(bitness_ >= kMinBitness && bitness_ <= kMaxBitness);
}

bool TableFunc::operator()(const FuncInput& input) const {
    assert(input.size() == bitness_);
    return tools::RandomFuncValue(bitness_, seed_, input);
}

std::vector<uint8_t> TableFunc::serialize() const {
    std::vector<uint8_t> bytes(sizeof(seed_));
    std::memcpy(bytes.data(), &seed_, sizeof(seed_));
    return bytes;
}

}  // namespace func
