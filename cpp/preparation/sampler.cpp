#include "sampler.h"

#include <cassert>
#include <cstddef>
#include <numeric>
#include <vector>

#include "func/table.h"
#include "tools/binary_tree.h"
#include "tools/random.h"
#include "tools/solver.h"

namespace preparation {

namespace {

std::vector<uint8_t> Pack(const std::vector<bool>& table) {
    std::vector<uint8_t> packed(table.size() / 8);
    for (size_t bit = 0; bit < table.size(); ++bit) {
        packed[bit / 8] |= static_cast<uint8_t>(table[bit]) << (bit % 8);
    }
    return packed;
}

// `helper_value` matters only for a series-1 helper query.
bool Evaluate(const tools::BinaryTree& tree, uint16_t bitness, size_t assignment, bool helper_value) {
    uint32_t node = 0;
    while (!tree[node].IsLeaf()) {
        const uint32_t query = tree[node].value;
        const bool answer = query == bitness ? helper_value : ((assignment >> query) & 1) != 0;
        node = tree[node].child[answer];
    }
    return tree[node].value != 0;
}

// Seed offset keeping the `_small` entries' stream clear of RandomEntry's over
// the same coordinates.
constexpr uint64_t kSmallStream = 0x5A11ULL;

}  // namespace

offline::Entry RandomEntry(const Parameters& parameters, uint16_t series, uint16_t bitness, uint32_t index) {
    assert(series == 1 || series == 2);
    tools::Random random(tools::EntrySeed(parameters.seed, series, bitness, index));
    // Both functions are drawn the way the generator draws one: a TableCase off
    // a seed this entry's key produced, read back as its truth table.
    const func::TableFunc target_case(bitness, random.Next());
    const func::TableFunc second_case(bitness, random.Next());
    const std::vector<bool>& truth_table = target_case.TruthTable();
    const std::vector<bool>& second_table = second_case.TruthTable();

    size_t min_depth = 0;
    size_t min_size = 0;
    if (series == 1) {
        min_depth = tools::SolveForDepthGiven(bitness, truth_table, second_table);
        min_size = tools::SolveForSizeGiven(bitness, truth_table, second_table);
    } else {
        min_depth = tools::SolveForDepthRestricted(bitness, truth_table, second_table);
        min_size = tools::SolveForSizeRestricted(bitness, truth_table, second_table);
    }
    assert(min_depth <= bitness);
    assert(min_size < (size_t{1} << bitness));
    return offline::Entry{Pack(truth_table), Pack(second_table), static_cast<uint8_t>(min_depth),
                          static_cast<uint16_t>(min_size)};
}

offline::Entry SmallEntry(const Parameters& parameters, uint16_t series, uint16_t bitness, uint32_t index) {
    assert(series == 1 || series == 2);
    assert(parameters.small_size_from >= 1);
    assert(parameters.small_size_from <= parameters.small_size_to);
    // A witness of `k` internal nodes has `k + 1` leaves to tell apart, and
    // there are only `2^bitness` assignments to do it with.
    assert(parameters.small_size_to < (size_t{1} << bitness));

    const uint16_t targets = parameters.small_size_to - parameters.small_size_from + 1;
    const uint16_t nodes = parameters.small_size_from + static_cast<uint16_t>(index % targets);
    const size_t rows = size_t{1} << bitness;

    tools::Random random(tools::EntrySeed(parameters.seed + kSmallStream, series, bitness, index));

    // Every input bit is a query; series 1's witness may also consult `f`.
    std::vector<uint32_t> ids(bitness);
    std::iota(ids.begin(), ids.end(), uint32_t{0});
    if (series == 1) ids.push_back(bitness);

    // A path spends one id per node, so a depth of `bitness` never runs out.
    const auto tree = tools::BinaryTree::Sample(random.Next(), bitness, nodes, ids);

    std::vector<bool> truth_table(rows);
    const func::TableFunc second_case(bitness, random.Next());
    const std::vector<bool>& second_table = second_case.TruthTable();
    if (series == 1) {
        for (size_t row = 0; row < rows; ++row) {
            truth_table[row] = Evaluate(tree, bitness, row, second_table[row]);
        }
    } else {
        const func::TableFunc outside_case(bitness, random.Next());
        const std::vector<bool>& outside_table = outside_case.TruthTable();
        // Off the subset `g` is drawn at random, which is what makes the subset
        // worth knowing: the witness is tiny on `X` and computing `g` everywhere
        // costs the hundreds of nodes a random table does.
        for (size_t row = 0; row < rows; ++row) {
            truth_table[row] = second_table[row] ? Evaluate(tree, bitness, row, false) : outside_table[row];
        }
    }

    const size_t min_size = series == 1 ? tools::SolveForSizeGiven(bitness, truth_table, second_table)
                                        : tools::SolveForSizeRestricted(bitness, truth_table, second_table);
    const size_t min_depth = series == 1 ? tools::SolveForDepthGiven(bitness, truth_table, second_table)
                                         : tools::SolveForDepthRestricted(bitness, truth_table, second_table);

    // The witness is what the pair was built around, so it bounds the answer;
    // the solver is what the answer is. A draw whose distinctions partly
    // collapse lands under `nodes` and is kept exactly as it came out.
    assert(min_size <= nodes);
    assert(min_depth <= bitness);
    return offline::Entry{Pack(truth_table), Pack(second_table), static_cast<uint8_t>(min_depth),
                          static_cast<uint16_t>(min_size)};
}

}  // namespace preparation
