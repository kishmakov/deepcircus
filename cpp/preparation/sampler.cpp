#include "sampler.h"

#include <cassert>
#include <cstddef>
#include <utility>
#include <vector>

#include "tools/solver.h"

namespace preparation {

namespace {

// SplitMix64 finalizer, same mixer the generator uses to key randomness off a
// case's coordinates.
uint64_t Mix(uint64_t value) {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

// A rolling `Mix` stream. Every draw a sampler makes comes off one of these, so
// an entry is a pure function of the coordinates its state was keyed with.
class Random {
public:
    explicit Random(uint64_t state) : state_(state) {}

    uint64_t Next() {
        state_ = Mix(state_);
        return state_;
    }

    // The bounds here are a handful of tree nodes or input bits against 2^64,
    // so the modulo bias is far below anything a sample could show.
    uint32_t Below(uint32_t bound) { return static_cast<uint32_t>(Next() % bound); }

    bool Bool() { return (Next() & 1) != 0; }

private:
    uint64_t state_;
};

std::vector<uint8_t> Pack(const std::vector<bool>& table) {
    std::vector<uint8_t> packed(table.size() / 8);
    for (size_t bit = 0; bit < table.size(); ++bit) {
        packed[bit / 8] |= static_cast<uint8_t>(table[bit]) << (bit % 8);
    }
    return packed;
}

std::vector<bool> Unpack(const std::vector<uint8_t>& packed, uint16_t bitness) {
    std::vector<bool> table(size_t{1} << bitness);
    for (size_t bit = 0; bit < table.size(); ++bit) {
        table[bit] = (packed[bit / 8] >> (bit % 8)) & 1;
    }
    return table;
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

// A series-2 point belongs to `X` on a fair coin. Sparser subsets let the
// witness collapse below the target too often; denser ones leave too little of
// `g` random to keep the subset worth knowing.
constexpr uint32_t kSubsetNumerator = 1;
constexpr uint32_t kSubsetDenominator = 2;

// Rejection bound shared by the two loops below. Both accept the large majority
// of their draws, so exhausting this many means the target is unreachable
// rather than merely unlucky.
constexpr uint32_t kMaxAttempts = 4096;

// Random shape: start from one leaf and `internal_nodes` times replace a
// uniformly chosen leaf with an internal node carrying two fresh leaves. The
// queries themselves are assigned by the walk below, which needs the internal
// nodes marked apart from the leaves first.
std::vector<Node> GrowShape(uint16_t internal_nodes, Random& random) {
    std::vector<Node> tree(1);
    std::vector<int> leaves = {0};
    for (uint16_t node_id = 0; node_id < internal_nodes; ++node_id) {
        const uint32_t slot = random.Below(static_cast<uint32_t>(leaves.size()));
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
                   uint32_t helper_weight, Random& random) {
    if (tree[node].query == kLeaf) return true;

    std::vector<uint16_t> free_bits;
    for (uint16_t bit = 0; bit < bitness; ++bit) {
        if (!(fixed_bits & (uint32_t{1} << bit))) free_bits.push_back(bit);
    }
    const uint32_t helper_share = helper_used ? 0 : helper_weight;
    const uint32_t total = static_cast<uint32_t>(free_bits.size()) + helper_share;
    if (total == 0) return false;

    const uint32_t draw = random.Below(total);
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
void AssignLeaves(std::vector<Node>& tree, Random& random) {
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

std::vector<Node> SampleTree(uint16_t bitness, uint16_t internal_nodes, uint32_t helper_weight, Random& random) {
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

// Keyed off the entry's coordinates plus the attempt counter the rejection loop
// walks, so a `_small` entry stays a pure function of the config while still
// being free to redraw.
uint64_t SmallKey(uint64_t seed, uint16_t series, uint16_t bitness, uint32_t index, uint32_t attempt) {
    uint64_t state = Mix(seed);
    state = Mix(state ^ 0x5A11ULL);  // keeps this stream clear of RandomEntry's
    state = Mix(state ^ series);
    state = Mix(state ^ bitness);
    state = Mix(state ^ index);
    return Mix(state ^ attempt);
}

}  // namespace

offline::Entry RandomEntry(const Parameters& parameters, uint16_t series, uint16_t bitness, uint32_t index) {
    assert(series == 1 || series == 2);
    uint64_t state = Mix(parameters.seed);
    state = Mix(state ^ series);
    state = Mix(state ^ bitness);
    state = Mix(state ^ index);

    const size_t table_bytes = offline::TableBytes(bitness);
    std::vector<uint8_t> g(table_bytes);
    std::vector<uint8_t> fx(table_bytes);
    for (std::vector<uint8_t>* table : {&g, &fx}) {
        for (uint8_t& byte : *table) {
            state = Mix(state);
            byte = static_cast<uint8_t>(state);
        }
    }

    const std::vector<bool> truth_table = Unpack(g, bitness);
    const std::vector<bool> second_table = Unpack(fx, bitness);

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
    return offline::Entry{std::move(g), std::move(fx), static_cast<uint8_t>(min_depth),
                          static_cast<uint16_t>(min_size)};
}

offline::Entry SmallEntry(const Parameters& parameters, uint16_t series, uint16_t bitness, uint32_t index) {
    assert(series == 1 || series == 2);
    assert(parameters.small_size_from >= 1);
    assert(parameters.small_size_from <= parameters.small_size_to);
    // A tree of `target` internal nodes has `target + 1` leaves to tell apart,
    // and there are only `2^bitness` assignments to do it with.
    assert(parameters.small_size_to < (size_t{1} << bitness));

    const uint16_t targets = parameters.small_size_to - parameters.small_size_from + 1;
    const uint16_t target = parameters.small_size_from + static_cast<uint16_t>(index % targets);
    const size_t rows = size_t{1} << bitness;

    for (uint32_t attempt = 0;; ++attempt) {
        // A witness tree usually is the optimum, so the loop turns over once or
        // twice; this many redraws means the target cannot be hit at all here.
        assert(attempt < kMaxAttempts);
        Random random(SmallKey(parameters.seed, series, bitness, index, attempt));

        // Series 1's witness may consult `f`, series 2's is a plain tree over
        // the inputs and the second table is the subset it must be right on.
        const uint32_t helper_weight = series == 1 ? kHelperWeightPerBit * bitness : 0;
        const std::vector<Node> tree = SampleTree(bitness, target, helper_weight, random);

        std::vector<bool> truth_table(rows);
        std::vector<bool> second_table(rows);
        if (series == 1) {
            for (size_t row = 0; row < rows; ++row) second_table[row] = random.Bool();
            for (size_t row = 0; row < rows; ++row) {
                truth_table[row] = Evaluate(tree, bitness, row, second_table[row]);
            }
        } else {
            for (size_t row = 0; row < rows; ++row) {
                second_table[row] = random.Below(kSubsetDenominator) < kSubsetNumerator;
            }
            // Off the subset `g` is drawn at random, which is what makes the
            // subset worth knowing: the witness is tiny on `X` and computing
            // `g` everywhere costs the hundreds of nodes a random table does.
            for (size_t row = 0; row < rows; ++row) {
                truth_table[row] = second_table[row] ? Evaluate(tree, bitness, row, false) : random.Bool();
            }
        }

        const size_t min_size = series == 1 ? tools::SolveForSizeGiven(bitness, truth_table, second_table)
                                            : tools::SolveForSizeRestricted(bitness, truth_table, second_table);
        if (min_size != target) continue;

        const size_t min_depth = series == 1 ? tools::SolveForDepthGiven(bitness, truth_table, second_table)
                                             : tools::SolveForDepthRestricted(bitness, truth_table, second_table);
        assert(min_depth <= bitness);
        return offline::Entry{Pack(truth_table), Pack(second_table), static_cast<uint8_t>(min_depth),
                              static_cast<uint16_t>(min_size)};
    }
}

}  // namespace preparation
