#include "decision_tree.h"
#include "table.h"
#include "utils.h"

#include <algorithm>
#include <cassert>
#include <bit>
#include <limits>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct SelectedBits {
    uint16_t mask = 0;
    uint16_t values = 0;

    bool operator==(const SelectedBits& other) const {
        return mask == other.mask && values == other.values;
    }
};

struct SelectedBitsHash {
    size_t operator()(SelectedBits bits) const {
        return static_cast<size_t>(bits.mask) | (static_cast<size_t>(bits.values) << 16);
    }
};

struct StateStats {
    uint8_t seen = 0;
};

struct StatePlan {
    bool is_leaf = true;
    bool value = false;
    uint16_t bit_id = 0;
    size_t depth = 0;
    size_t size = 0;
};

enum class TruthTableObjective {
    Depth,
    Size,
};

SelectedBits WithBit(SelectedBits bits, uint16_t bit_id, bool value) {
    const uint16_t bit = static_cast<uint16_t>(1u << bit_id);
    bits.mask |= bit;
    if (value) {
        bits.values |= bit;
    } else {
        bits.values &= static_cast<uint16_t>(~bit);
    }
    return bits;
}

bool IsBetterPlan(const StatePlan& candidate, const StatePlan& best, TruthTableObjective objective) {
    if (objective == TruthTableObjective::Depth) {
        return std::tie(candidate.depth, candidate.size, candidate.bit_id) <
            std::tie(best.depth, best.size, best.bit_id);
    }
    return std::tie(candidate.size, candidate.depth, candidate.bit_id) <
        std::tie(best.size, best.depth, best.bit_id);
}

class TruthTableTreeBuilder {
public:
    TruthTableTreeBuilder(uint16_t bitness, std::vector<bool> truth_table, TruthTableObjective objective)
        : bitness_(bitness)
        , truth_table_(std::move(truth_table))
        , objective_(objective)
        , tree_(bitness)
    {
        assert(bitness_ <= 16);
        assert(truth_table_.size() == (size_t{1} << bitness_));
    }

    DecisionTree Build() {
        tree_.nodes.push_back(false);
        if (bitness_ <= 6) {
            BuildOptimalNodes(0, {});
        } else {
            BuildOrderedNodes(0, {});
        }
        tree_.Finalize();
        return std::move(tree_);
    }

private:
    void BuildOptimalNodes(size_t node_id, SelectedBits bits) {
        const StatePlan plan = ChoosePlan(bits);

        if (plan.is_leaf) {
            tree_.nodes[node_id] = plan.value;
            ++tree_.num_leafs;
            return;
        }

        const size_t child0 = tree_.nodes.size();
        const size_t child1 = tree_.nodes.size() + 1;
        tree_.nodes[node_id] = Div{plan.bit_id, child0, child1};
        tree_.used_bits[plan.bit_id] = true;
        tree_.nodes.push_back(false);
        tree_.nodes.push_back(false);

        BuildOptimalNodes(child0, WithBit(bits, plan.bit_id, false));
        BuildOptimalNodes(child1, WithBit(bits, plan.bit_id, true));
    }

    void BuildOrderedNodes(size_t node_id, SelectedBits bits) {
        const StateStats stats = Analyze(bits);
        if (stats.seen != 3) {
            tree_.nodes[node_id] = stats.seen == 2;
            ++tree_.num_leafs;
            return;
        }

        const uint16_t bit_id = ChooseOrderedBit(bits);
        const size_t child0 = tree_.nodes.size();
        const size_t child1 = tree_.nodes.size() + 1;
        tree_.nodes[node_id] = Div{bit_id, child0, child1};
        tree_.used_bits[bit_id] = true;
        tree_.nodes.push_back(false);
        tree_.nodes.push_back(false);

        BuildOrderedNodes(child0, WithBit(bits, bit_id, false));
        BuildOrderedNodes(child1, WithBit(bits, bit_id, true));
    }

    StateStats Analyze(SelectedBits bits) {
        auto cached = stats_memo_.find(bits);
        if (cached != stats_memo_.end()) {
            return cached->second;
        }

        StateStats stats;
        const uint16_t all_bits = AllBitsMask();
        const uint16_t free_mask = static_cast<uint16_t>(all_bits ^ bits.mask);
        uint16_t sub = free_mask;
        do {
            const uint16_t input_id = static_cast<uint16_t>(sub | bits.values);
            stats.seen |= static_cast<uint8_t>(1u << TableValue(input_id));
            sub = static_cast<uint16_t>((sub - 1) & free_mask);
        } while (sub != free_mask);

        stats_memo_[bits] = stats;
        return stats;
    }

    StatePlan ChoosePlan(SelectedBits bits) {
        auto cached = plan_memo_.find(bits);
        if (cached != plan_memo_.end()) {
            return cached->second;
        }

        const StateStats stats = Analyze(bits);
        if (stats.seen != 3) {
            const StatePlan plan{true, stats.seen == 2, 0, 0, 0};
            plan_memo_[bits] = plan;
            return plan;
        }

        StatePlan best;
        best.is_leaf = false;
        best.depth = std::numeric_limits<size_t>::max();
        best.size = std::numeric_limits<size_t>::max();

        for (uint16_t bit_id = 0; bit_id < bitness_; ++bit_id) {
            if ((bits.mask & (1u << bit_id)) != 0) {
                continue;
            }

            const StatePlan left = ChoosePlan(WithBit(bits, bit_id, false));
            const StatePlan right = ChoosePlan(WithBit(bits, bit_id, true));
            const StatePlan candidate{
                false,
                false,
                bit_id,
                1 + std::max(left.depth, right.depth),
                1 + left.size + right.size,
            };
            if (IsBetterPlan(candidate, best, objective_)) {
                best = candidate;
            }
        }

        plan_memo_[bits] = best;
        return best;
    }

    uint16_t AllBitsMask() const {
        return static_cast<uint16_t>((uint32_t{1} << bitness_) - 1u);
    }

    uint16_t ChooseOrderedBit(SelectedBits bits) {
        uint16_t best_bit_id = bitness_;
        uint8_t best_mixed_children = std::numeric_limits<uint8_t>::max();

        for (uint16_t bit_id = 0; bit_id < bitness_; ++bit_id) {
            if ((bits.mask & (1u << bit_id)) != 0) {
                continue;
            }

            const StateStats left = Analyze(WithBit(bits, bit_id, false));
            const StateStats right = Analyze(WithBit(bits, bit_id, true));
            const uint8_t mixed_children =
                static_cast<uint8_t>((left.seen == 3) + (right.seen == 3));
            if (std::tie(mixed_children, bit_id) < std::tie(best_mixed_children, best_bit_id)) {
                best_mixed_children = mixed_children;
                best_bit_id = bit_id;
            }
        }

        assert(best_bit_id != bitness_);
        return best_bit_id;
    }

    bool TableValue(uint16_t input_id) const {
        return truth_table_[input_id];
    }

    uint16_t bitness_;
    std::vector<bool> truth_table_;
    TruthTableObjective objective_;
    DecisionTree tree_;
    std::unordered_map<SelectedBits, StateStats, SelectedBitsHash> stats_memo_;
    std::unordered_map<SelectedBits, StatePlan, SelectedBitsHash> plan_memo_;
};

size_t RandomUnusedBit(const std::vector<bool>& path_used_bits, size_t free_bits, std::mt19937& rng) {
    assert(free_bits > 0);

    const size_t selected = std::uniform_int_distribution<size_t>(0, free_bits - 1)(rng);
    size_t seen = 0;
    for (size_t bit = 0; bit < path_used_bits.size(); ++bit) {
        if (path_used_bits[bit]) continue;
        if (seen == selected) return bit;
        ++seen;
    }
    assert(false);
    return 0;
}

std::pair<size_t, size_t> SplitBudget(size_t n, std::mt19937& rng) {
    assert(n >= 1);
    const size_t remaining = n - 1;
    if (remaining == 0) return {0, 0};
    const size_t left = std::uniform_int_distribution<size_t>(0, remaining)(rng);
    return {left, remaining - left};
}

size_t MaxInternalNodes(size_t free_bits) {
    if (free_bits >= std::numeric_limits<size_t>::digits) {
        return std::numeric_limits<size_t>::max();
    }
    return (size_t{1} << free_bits) - 1;
}

size_t ComputeDepth(const std::vector<Node>& nodes, size_t node_id) {
    const Node& node = nodes[node_id];
    const Div* division = std::get_if<Div>(&node);
    if (division == nullptr) {
        return 0;
    }
    return 1 + std::max(
        ComputeDepth(nodes, division->child0),
        ComputeDepth(nodes, division->child1));
}

std::vector<bool> TruthTableVector(uint16_t bitness, uint64_t truth_table) {
    assert(bitness <= 6);
    std::vector<bool> values(size_t{1} << bitness);
    for (uint16_t input_id = 0; input_id < values.size(); ++input_id) {
        values[input_id] = ((truth_table >> input_id) & 1ull) != 0;
    }
    return values;
}

}  // namespace

DecisionTree::DecisionTree(uint16_t bitness)
    : used_bits(bitness, false)
    , bitness_(bitness)
{
}

uint16_t DecisionTree::Bitness() const {
    return bitness_;
}

size_t DecisionTree::AddLeaf(bool value) {
    const size_t node_id = nodes.size();
    nodes.push_back(value);
    ++num_leafs;
    return node_id;
}

size_t DecisionTree::BuildSubtree(
    size_t budget,
    std::vector<bool>& path_used_bits,
    size_t path_used_count,
    bool required_value,
    RandomBoolGenerator& rng)
{
    assert(path_used_bits.size() == bitness_);
    assert(path_used_count <= bitness_);

    const size_t free_bits = bitness_ - path_used_count;
    if (budget == 0 || free_bits == 0) {
        return AddLeaf(required_value);
    }

    const size_t node_id = nodes.size();
    nodes.push_back(false);

    const size_t bit_id = RandomUnusedBit(path_used_bits, free_bits, rng.RNG());
    used_bits[bit_id] = true;
    path_used_bits[bit_id] = true;

    auto [left_budget, right_budget] = SplitBudget(budget, rng.RNG());

    const size_t max_child_nodes = MaxInternalNodes(free_bits - 1);
    left_budget = std::min(left_budget, max_child_nodes);
    right_budget = std::min(right_budget, max_child_nodes);

    const bool child0_required_value = rng.Generate();
    const bool child1_required_value = !child0_required_value;

    const size_t child0 = BuildSubtree(
        left_budget,
        path_used_bits,
        path_used_count + 1,
        child0_required_value,
        rng);
    const size_t child1 = BuildSubtree(
        right_budget,
        path_used_bits,
        path_used_count + 1,
        child1_required_value,
        rng);
    path_used_bits[bit_id] = false;
    nodes[node_id] = Div{bit_id, child0, child1};
    return node_id;
}

void DecisionTree::Finalize() {
    assert(!nodes.empty());
    depth = ComputeDepth(nodes, 0);
}

bool DecisionTree::Evaluate(std::string_view input) const {
    assert(input.size() == bitness_);

    size_t node_id = 0;
    while (true) {
        const Node& node = nodes[node_id];
        const Div* division = std::get_if<Div>(&node);
        if (division == nullptr) {
            return std::get<bool>(node);
        }
        assert(division->bitId < input.size());
        node_id = input[division->bitId] == '1'
            ? division->child1
            : division->child0;
    }
}

void DecisionTree::FillValueTensor(
    size_t reps,
    uint64_t seed,
    float* out) const
{
    const auto evaluate = [this](std::string_view input) {
        return Evaluate(input);
    };
    FillGeneratedValueTensor(
        bitness_,
        reps,
        seed,
        out,
        evaluate);
}

void DecisionTree::FillRestrictionsTensor(
    size_t reps,
    uint64_t seed,
    float* out) const
{
    const auto evaluate = [this](std::string_view input) {
        return Evaluate(input);
    };
    FillGeneratedRestrictionsTensor(
        bitness_,
        reps,
        seed,
        out,
        evaluate);
}

DecisionTree BuildDepthOptimalDecisionTree(uint16_t bitness, uint64_t truth_table) {
    return TruthTableTreeBuilder(
        bitness,
        TruthTableVector(bitness, truth_table),
        TruthTableObjective::Depth).Build();
}

DecisionTree BuildSizeOptimalDecisionTree(uint16_t bitness, uint64_t truth_table) {
    return TruthTableTreeBuilder(
        bitness,
        TruthTableVector(bitness, truth_table),
        TruthTableObjective::Size).Build();
}

DecisionTree BuildDepthOptimalDecisionTree(uint16_t bitness, const std::vector<bool>& truth_table) {
    return TruthTableTreeBuilder(
        bitness,
        truth_table,
        TruthTableObjective::Depth).Build();
}

namespace {

// Each DP state (mask, values) is packed into a single byte:
//   bits 0-1: "seen"  - which output values (0 and/or 1) are still
//             reachable given the bits fixed so far.
//   bits 2-7: "depth" - optimal remaining query depth for that state
//             (valid only once seen == 3, i.e. the value isn't forced yet).
// This halves per-state storage vs. two parallel uint8_t arrays and,
// combined with flattening below, avoids ~2 * 2^bitness separate small
// heap allocations that the original vector<vector<uint8_t>> design paid
// for one per mask.
constexpr uint8_t kSeenMask = 0b11;
constexpr int kDepthShift = 2;

uint8_t MakeDepthCell(uint8_t seen, uint8_t depth) {
    return seen | (depth << kDepthShift);
}
uint8_t DepthOf(uint8_t cell) { return cell >> kDepthShift; }

uint32_t MakeSizeCell(uint8_t seen, uint32_t size) {
    return static_cast<uint32_t>(seen) | (size << kDepthShift);
}
uint32_t SizeOf(uint32_t cell) { return cell >> kDepthShift; }

template<typename Cell>
uint8_t SeenOf(Cell cell) { return static_cast<uint8_t>(cell & kSeenMask); }


// Number of bits already fixed in `mask` below position `bit_id` - i.e.
// the position `bit_id` occupies once fixed bits are stripped out.
uint16_t FixedBitsBefore(uint32_t mask, uint16_t bit_id) {
    return static_cast<uint16_t>(std::popcount(mask & ((uint32_t{1} << bit_id) - 1)));
}

// Inserts `bit_value` as a new bit at `bit_pos` within the packed
// "free bits only" value `values`, shifting higher bits up by one.
size_t InsertFixedBit(size_t values, uint16_t bit_pos, bool bit_value) {
    const size_t lower_mask = (size_t{1} << bit_pos) - 1;
    const size_t lower = values & lower_mask;
    const size_t upper = values & ~lower_mask;
    return lower | (static_cast<size_t>(bit_value) << bit_pos) | (upper << 1);
}

// Everything about a "query this bit next" transition that does NOT
// depend on `values`, hoisted out of the inner loop and computed once
// per mask instead of once per (mask, values) pair.
template <typename Cell>
struct FreeBit {
    uint16_t bit_pos;    // position of this bit among mask's free bits
    const Cell* child;   // pointer to the child mask's state row
};

} // namespace

size_t SolveForDepth(uint16_t bitness, const std::vector<bool>& truth_table) {
    assert(bitness <= kSolvableTableBitness);
    assert(truth_table.size() == (size_t{1} << bitness));

    const uint32_t masks_number = uint32_t{1} << bitness;
    const uint32_t all_bits_mask = masks_number - 1;

    // Bucket masks by popcount once, so each DP level below is a plain
    // scan instead of an O(masks_number) pass with a popcount+skip filter.
    std::vector<std::vector<uint32_t>> masks_by_popcount(bitness + 1);
    for (uint32_t mask = 0; mask < masks_number; ++mask) {
        masks_by_popcount[std::popcount(mask)].push_back(mask);
    }

    // Flatten all (mask, values) states into one contiguous buffer of size
    // exactly sum_mask 2^popcount(mask) == 3^bitness (one slot per ternary
    // assignment: each variable is unqueried / fixed-0 / fixed-1).
    std::vector<size_t> offset(masks_number);
    size_t total_states = 0;
    for (uint32_t mask = 0; mask < masks_number; ++mask) {
        offset[mask] = total_states;
        total_states += size_t{1} << std::popcount(mask);
    }

    std::vector<uint8_t> table(total_states);

    // Base case: every bit fixed, state maps 1:1 onto a truth-table entry.
    for (size_t values = 0; values < truth_table.size(); ++values) {
        table[offset[all_bits_mask] + values] = MakeDepthCell(1u << truth_table[values], 0);
    }

    // Reused across masks to avoid a heap allocation per mask.
    std::vector<FreeBit<uint8_t>> free_bits;
    free_bits.reserve(bitness);

    for (uint16_t fixed_count = bitness; fixed_count-- > 0;) {
        for (uint32_t mask : masks_by_popcount[fixed_count]) {
            const size_t states_number = size_t{1} << fixed_count;
            uint8_t* row = table.data() + offset[mask];

            free_bits.clear();
            for (uint16_t bit_id = 0; bit_id < bitness; ++bit_id) {
                const uint32_t bit = uint32_t{1} << bit_id;
                if (mask & bit) continue;
                const uint32_t child_mask = mask | bit;
                free_bits.push_back(
                    {FixedBitsBefore(mask, bit_id), table.data() + offset[child_mask]});
            }

            // Smallest free bit id, used solely to propagate "seen" up from the next level;
            // which free bit is used here doesn't affect correctness since the recursion
            // already folds in every other free bit by induction.
            const uint8_t* seen_child = free_bits.front().child;
            const uint16_t seen_bit_pos = free_bits.front().bit_pos;

            for (size_t values = 0; values < states_number; ++values) {
                const uint8_t state_seen = static_cast<uint8_t>(
                    SeenOf(seen_child[InsertFixedBit(values, seen_bit_pos, false)]) |
                    SeenOf(seen_child[InsertFixedBit(values, seen_bit_pos, true)])
                );

                if (state_seen != 3) {
                    // Output already forced given the fixed bits - no more
                    // queries needed.
                    row[values] = MakeDepthCell(state_seen, 0);
                    continue;
                }

                uint8_t best_depth = static_cast<uint8_t>(bitness);
                for (const FreeBit<uint8_t>& fb : free_bits) {
                    const uint8_t child_depth = static_cast<uint8_t>(1 + std::max(
                        DepthOf(fb.child[InsertFixedBit(values, fb.bit_pos, false)]),
                        DepthOf(fb.child[InsertFixedBit(values, fb.bit_pos, true)])));
                    best_depth = std::min(best_depth, child_depth);
                    if (best_depth == 1) break;  // depth 1 is the global minimum
                }
                row[values] = MakeDepthCell(3, best_depth);
            }
        }
    }

    return DepthOf(table[offset[0] + 0]);
}

size_t SolveForSize(uint16_t bitness, const std::vector<bool>& truth_table) {
    assert(bitness <= kSolvableTableBitness);
    assert(truth_table.size() == (size_t{1} << bitness));

    const uint32_t masks_number = uint32_t{1} << bitness;
    const uint32_t all_bits_mask = masks_number - 1;

    std::vector<std::vector<uint32_t>> masks_by_popcount(bitness + 1);
    for (uint32_t mask = 0; mask < masks_number; ++mask) {
        masks_by_popcount[std::popcount(mask)].push_back(mask);
    }

    std::vector<size_t> offset(masks_number);
    size_t total_states = 0;
    for (uint32_t mask = 0; mask < masks_number; ++mask) {
        offset[mask] = total_states;
        total_states += size_t{1} << std::popcount(mask);
    }

    std::vector<uint32_t> table(total_states);

    for (size_t values = 0; values < truth_table.size(); ++values) {
        table[offset[all_bits_mask] + values] = MakeSizeCell(1u << truth_table[values], 0);
    }

    std::vector<FreeBit<uint32_t>> free_bits;
    free_bits.reserve(bitness);

    for (uint16_t fixed_count = bitness; fixed_count-- > 0;) {
        for (uint32_t mask : masks_by_popcount[fixed_count]) {
            const size_t states_number = size_t{1} << fixed_count;
            uint32_t* row = table.data() + offset[mask];

            free_bits.clear();
            for (uint16_t bit_id = 0; bit_id < bitness; ++bit_id) {
                const uint32_t bit = uint32_t{1} << bit_id;
                if (mask & bit) continue;
                const uint32_t child_mask = mask | bit;
                free_bits.push_back(
                    {FixedBitsBefore(mask, bit_id), table.data() + offset[child_mask]});
            }

            const uint32_t* seen_child = free_bits.front().child;
            const uint16_t seen_bit_pos = free_bits.front().bit_pos;

            for (size_t values = 0; values < states_number; ++values) {
                const uint8_t state_seen = static_cast<uint8_t>(
                    SeenOf(seen_child[InsertFixedBit(values, seen_bit_pos, false)]) |
                    SeenOf(seen_child[InsertFixedBit(values, seen_bit_pos, true)])
                );

                if (state_seen != 3) {
                    row[values] = MakeSizeCell(state_seen, 0);
                    continue;
                }

                uint32_t best_size = std::numeric_limits<uint32_t>::max();
                for (const FreeBit<uint32_t>& fb : free_bits) {
                    const uint32_t child_size =
                        1 + SizeOf(fb.child[InsertFixedBit(values, fb.bit_pos, false)])
                          + SizeOf(fb.child[InsertFixedBit(values, fb.bit_pos, true)]);
                    best_size = std::min(best_size, child_size);
                    if (best_size == 1) break;
                }
                row[values] = MakeSizeCell(3, best_size);
            }
        }
    }

    return SizeOf(table[offset[0] + 0]);
}
