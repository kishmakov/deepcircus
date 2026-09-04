#include "func/tt.h"

#include <assert.h>
#include <string.h>

#include <utility>
#include <vector>

#include "func/table.h"
#include "func/tree.h"
#include "tools/random.h"

namespace func {

namespace {

// The two children draw from separate domains of the one seed, so neither
// one's stream depends on the other's.
constexpr uint64_t kTTTableDomain = 0x74745f7461626c65ull;
constexpr uint64_t kTTTreeDomain = 0x74745f7472656500ull;

// Length of the table's share of the serialized bytes.
uint32_t TableBytes(const std::vector<uint8_t>& bytes) {
    assert(bytes.size() >= sizeof(uint32_t));

    uint32_t table_bytes = 0;
    memcpy(&table_bytes, bytes.data(), sizeof(table_bytes));
    assert(bytes.size() >= sizeof(uint32_t) + size_t{table_bytes});
    return table_bytes;
}

std::vector<uint8_t> TablePart(const std::vector<uint8_t>& bytes) {
    const auto begin = bytes.begin() + sizeof(uint32_t);
    return std::vector<uint8_t>(begin, begin + TableBytes(bytes));
}

std::vector<uint8_t> TreePart(const std::vector<uint8_t>& bytes) {
    return std::vector<uint8_t>(bytes.begin() + sizeof(uint32_t) + TableBytes(bytes), bytes.end());
}

}  // namespace

TTFunc::TTFunc(uint16_t bitness, uint64_t seed)
    : TTFunc(bitness, TableFunc(bitness, tools::DomainSeed(seed, kTTTableDomain, bitness)),
             TreeFunc(TreeBitness(bitness), tools::DomainSeed(seed, kTTTreeDomain, bitness))) {}

TTFunc::TTFunc(uint16_t bitness, TableFunc table, TreeFunc tree)
    : func::Func(bitness), table_(std::move(table)), tree_(std::move(tree)) {
    assert(bitness_ >= kMinBitness && bitness_ <= kMaxTTBitness);
}

TTFunc::TTFunc(uint16_t bitness, const std::vector<uint8_t>& bytes)
    : TTFunc(bitness, TableFunc(bitness, TablePart(bytes)), TreeFunc(TreeBitness(bitness), TreePart(bytes))) {}

bool TTFunc::operator()(const FuncInput& input) const {
    assert(input.size() == bitness_);

    // The table's value joins the primary inputs as the tree's last variable.
    FuncInput extended = input;
    extended.push_back(table_(input));
    return tree_(extended);
}

std::vector<uint8_t> TTFunc::serialize() const {
    const std::vector<uint8_t> table_bytes = table_.serialize();
    const std::vector<uint8_t> tree_bytes = tree_.serialize();
    assert(table_bytes.size() <= UINT32_MAX);

    std::vector<uint8_t> bytes(sizeof(uint32_t) + table_bytes.size() + tree_bytes.size());
    const uint32_t table_size = static_cast<uint32_t>(table_bytes.size());
    memcpy(bytes.data(), &table_size, sizeof(table_size));
    memcpy(bytes.data() + sizeof(table_size), table_bytes.data(), table_bytes.size());
    memcpy(bytes.data() + sizeof(table_size) + table_bytes.size(), tree_bytes.data(), tree_bytes.size());
    return bytes;
}

}  // namespace func
