#include "sampler.h"

#include <cassert>
#include <cstddef>
#include <vector>

#include "func/table.h"
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

// A witness tree's node: `query` is an input bit id, or `bitness` for the
// helper query the S_1 model lets a path make once, or `kLeaf`.
constexpr int kLeaf = -1;

struct Node {
    int query = kLeaf;
    int child[2] = {kLeaf, kLeaf};
    bool value = false;  // leaf output
};

// The helper's weight against the free input bits when a series-1 node picks
// its query, as a multiple of the bitness. At 2 the root consults `f` with
// probability 2/3: a witness that never queries `f` leaves an entry whose
// target is the same with and without it, which is nothing for the S_1 model to
// learn from.
constexpr uint32_t kHelperWeightPerBit = 2;

// Redraw bound for the shape loop below, which accepts the large majority of
// its draws: exhausting this many means no shape of that many nodes fits inside
// the bitness rather than merely that the draws were unlucky.
constexpr uint32_t kMaxAttempts = 4096;

// Random shape: start from one leaf and `internal_nodes` times replace a
// uniformly chosen leaf with an internal node carrying two fresh leaves. The
// queries themselves are assigned by the walk below, which needs the internal
// nodes marked apart from the leaves first.
std::vector<Node> GrowShape(uint16_t internal_nodes, tools::Random& random) {
    std::vector<Node> tree(1);
    std::vector<int> leaves = {0};
    for (uint16_t node_id = 0; node_id < internal_nodes; ++node_id) {
        const uint32_t slot = static_cast<uint32_t>(random.Below(leaves.size()));
        const int node = leaves[slot];
        leaves[slot] = leaves.back();
        leaves.pop_back();
        for (int side = 0; side < 2; ++side) {
            tree[node].child[side] = static_cast<int>(tree.size());
            tree.push_back(Node{});
            leaves.push_back(tree[node].child[side]);
        }
        tree[node].query = 0;  // internal; the real query lands in AssignQueries
    }
    return tree;
}

// Gives every internal node a query an assignment reaching it has not answered
// yet: an input bit still free on this path, or the helper if this path has not
// consulted it. Fails when a path is longer than the queries available to it,
// which only a shape deeper than `bitness + 1` manages -- the caller redraws.
bool AssignQueries(std::vector<Node>& tree, int node, uint16_t bitness, uint32_t fixed_bits, bool helper_used,
                   uint32_t helper_weight, tools::Random& random) {
    if (tree[node].query == kLeaf) return true;

    std::vector<uint16_t> free_bits;
    for (uint16_t bit = 0; bit < bitness; ++bit) {
        if (!(fixed_bits & (uint32_t{1} << bit))) free_bits.push_back(bit);
    }
    const uint32_t helper_share = helper_used ? 0 : helper_weight;
    const uint32_t total = static_cast<uint32_t>(free_bits.size()) + helper_share;
    if (total == 0) return false;

    const uint32_t draw = static_cast<uint32_t>(random.Below(total));
    const bool take_helper = draw >= free_bits.size();
    tree[node].query = take_helper ? bitness : free_bits[draw];

    const uint32_t next_fixed = take_helper ? fixed_bits : (fixed_bits | (uint32_t{1} << tree[node].query));
    const bool next_helper = helper_used || take_helper;
    for (int side = 0; side < 2; ++side) {
        if (!AssignQueries(tree, tree[node].child[side], bitness, next_fixed, next_helper, helper_weight, random)) {
            return false;
        }
    }
    return true;
}

// Random leaf outputs, then a pass forcing sibling leaves apart: a node whose
// two leaves agree is removable, so leaving it in place would put the exact
// answer under the target for no reason but sloppy labelling.
void AssignLeaves(std::vector<Node>& tree, tools::Random& random) {
    for (Node& node : tree) {
        if (node.query == kLeaf) node.value = random.Bool();
    }
    for (size_t node_id = 0; node_id < tree.size(); ++node_id) {
        if (tree[node_id].query == kLeaf) continue;
        Node& zero = tree[tree[node_id].child[0]];
        Node& one = tree[tree[node_id].child[1]];
        if (zero.query == kLeaf && one.query == kLeaf && zero.value == one.value) one.value = !zero.value;
    }
}

std::vector<Node> SampleTree(uint16_t bitness, uint16_t internal_nodes, uint32_t helper_weight, tools::Random& random) {
    for (uint32_t attempt = 0;; ++attempt) {
        assert(attempt < kMaxAttempts);
        std::vector<Node> tree = GrowShape(internal_nodes, random);
        if (!AssignQueries(tree, 0, bitness, 0, false, helper_weight, random)) continue;
        AssignLeaves(tree, random);
        return tree;
    }
}

// `helper_value` is read only by a node the series-1 weighting produced; a
// series-2 witness has no helper query, so what is passed there never matters.
bool Evaluate(const std::vector<Node>& tree, uint16_t bitness, size_t assignment, bool helper_value) {
    int node = 0;
    while (tree[node].query != kLeaf) {
        const int query = tree[node].query;
        const bool answer = query == bitness ? helper_value : ((assignment >> query) & 1) != 0;
        node = tree[node].child[answer];
    }
    return tree[node].value;
}

// Keyed off the entry's coordinates alone, so a `_small` entry is a pure
// function of the config and of its own index -- never of how many draws the
// entries before it happened to burn.
uint64_t SmallKey(uint64_t seed, uint16_t series, uint16_t bitness, uint32_t index) {
    uint64_t state = tools::Mix(seed);
    state = tools::Mix(state ^ 0x5A11ULL);  // keeps this stream clear of RandomEntry's
    state = tools::Mix(state ^ series);
    state = tools::Mix(state ^ bitness);
    return tools::Mix(state ^ index);
}

uint64_t RandomKey(uint64_t seed, uint16_t series, uint16_t bitness, uint32_t index) {
    uint64_t state = tools::Mix(seed);
    state = tools::Mix(state ^ series);
    state = tools::Mix(state ^ bitness);
    return tools::Mix(state ^ index);
}

}  // namespace

offline::Entry RandomEntry(const Parameters& parameters, uint16_t series, uint16_t bitness, uint32_t index) {
    assert(series == 1 || series == 2);
    tools::Random random(RandomKey(parameters.seed, series, bitness, index));
    // Both functions are drawn the way the generator draws one: a TableCase off
    // a seed this entry's key produced, read back as its truth table.
    const func::TableCase target_case(bitness, random.Next());
    const func::TableCase second_case(bitness, random.Next());
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

    tools::Random random(SmallKey(parameters.seed, series, bitness, index));

    // Series 1's witness may consult `f`, series 2's is a plain tree over the
    // inputs and the second table is the subset it must be right on.
    const uint32_t helper_weight = series == 1 ? kHelperWeightPerBit * bitness : 0;
    const std::vector<Node> tree = SampleTree(bitness, nodes, helper_weight, random);

    std::vector<bool> truth_table(rows);
    const func::TableCase second_case(bitness, random.Next());
    const std::vector<bool>& second_table = second_case.TruthTable();
    if (series == 1) {
        for (size_t row = 0; row < rows; ++row) {
            truth_table[row] = Evaluate(tree, bitness, row, second_table[row]);
        }
    } else {
        const func::TableCase outside_case(bitness, random.Next());
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
