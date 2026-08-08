#include "tools/solver.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstdint>
#include <limits>
#include <vector>

namespace tools {

namespace {

// Each DP state (mask, values) is packed into a single cell:
//   bits 0-1: "seen" - which output values (0 and/or 1) are still reachable
//             given the bits fixed so far; 0 means no assignment reaches this
//             state at all.
//   bits 2+ : "cost" - optimal remaining cost for that state (valid only once
//             seen == 3, i.e. the value isn't forced yet).
// This packing halves per-state storage vs. two parallel arrays and, combined
// with the flattening below, avoids ~2 * 2^bitness separate small heap
// allocations that a vector<vector<...>> design pays one of per mask.
constexpr uint32_t kSeenMask = 0b11;
constexpr int kCostShift = 2;

template <typename Cell>
Cell MakeCell(uint8_t seen, uint32_t cost) {
    return static_cast<Cell>(seen | (cost << kCostShift));
}

template <typename Cell>
uint32_t CostOf(Cell cell) {
    return static_cast<uint32_t>(cell) >> kCostShift;
}

template <typename Cell>
uint8_t SeenOf(Cell cell) {
    return static_cast<uint8_t>(cell & kSeenMask);
}

// A queried node costs one level on top of its deeper branch, so the cost never
// exceeds the number of queryable bits and six bits of it fit in a byte.
struct DepthPolicy {
    using Cell = uint8_t;
    static uint32_t Combine(uint32_t zero, uint32_t one) { return 1 + std::max(zero, one); }
    static uint32_t Worst(uint16_t bitness) { return bitness; }
};

// A queried node costs itself plus both of its branches.
struct SizePolicy {
    using Cell = uint32_t;
    static uint32_t Combine(uint32_t zero, uint32_t one) { return 1 + zero + one; }
    static uint32_t Worst(uint16_t) { return std::numeric_limits<uint32_t>::max() >> kCostShift; }
};

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
    uint16_t bit_pos;   // position of this bit among mask's free bits
    const Cell* child;  // pointer to the child mask's state row
};

// One layer holds every (mask, values) state flattened into a contiguous buffer
// of size exactly sum_mask 2^popcount(mask) == 3^bitness (one slot per ternary
// assignment: each variable is unqueried / fixed-0 / fixed-1). That 3^bitness
// walk is what caps `kMaxSolvableBitness`.
struct StateLayout {
    explicit StateLayout(uint16_t bitness) : masks_by_popcount(bitness + 1), offset(size_t{1} << bitness) {
        const uint32_t masks_number = uint32_t{1} << bitness;
        for (uint32_t mask = 0; mask < masks_number; ++mask) {
            // Bucketing masks by popcount up front makes each DP level below a
            // plain scan instead of a full pass with a popcount+skip filter.
            masks_by_popcount[std::popcount(mask)].push_back(mask);
            offset[mask] = total_states;
            total_states += size_t{1} << std::popcount(mask);
        }
    }

    // Offset of the all-bits-fixed mask, whose states map 1:1 onto truth-table
    // entries: the layer's leaves.
    size_t LeavesOffset() const { return offset.back(); }

    std::vector<std::vector<uint32_t>> masks_by_popcount;
    std::vector<size_t> offset;
    size_t total_states = 0;
};

// Writes a layer's leaves: every bit fixed, so the state is the single
// assignment `values`. `reachable_table`, when given, keeps only the
// assignments where it holds `reachable_value` -- the subset a restricted solve
// is confined to, or the side of the split a fixed-helper layer stands for. The
// rest are left unreachable (seen == 0), which the recursion below reads as
// "any branch landing here is dead and costs nothing".
template <typename Cell>
void FillLeaves(const StateLayout& layout, const std::vector<bool>& truth_table,
                const std::vector<bool>* reachable_table, bool reachable_value, Cell* layer) {
    Cell* leaves = layer + layout.LeavesOffset();
    for (size_t values = 0; values < truth_table.size(); ++values) {
        const bool reachable = reachable_table == nullptr || (*reachable_table)[values] == reachable_value;
        leaves[values] = MakeCell<Cell>(reachable ? 1u << truth_table[values] : 0, 0);
    }
}

// Solves every non-leaf state of `layer`, mask by mask from the most fixed bits
// down to none; its leaves must already be filled. When `zero_layer` and
// `one_layer` are given they are the two layers reached by querying the helper
// function, and each state gets that query offered as one more transition.
template <typename Policy>
void FillLayer(uint16_t bitness, const StateLayout& layout, typename Policy::Cell* layer,
               const typename Policy::Cell* zero_layer, const typename Policy::Cell* one_layer) {
    using Cell = typename Policy::Cell;

    // Reused across masks to avoid a heap allocation per mask.
    std::vector<FreeBit<Cell>> free_bits;
    free_bits.reserve(bitness);

    for (uint16_t fixed_count = bitness; fixed_count-- > 0;) {
        for (uint32_t mask : layout.masks_by_popcount[fixed_count]) {
            const size_t states_number = size_t{1} << fixed_count;
            Cell* row = layer + layout.offset[mask];

            free_bits.clear();
            for (uint16_t bit_id = 0; bit_id < bitness; ++bit_id) {
                const uint32_t bit = uint32_t{1} << bit_id;
                if (mask & bit) continue;
                const uint32_t child_mask = mask | bit;
                free_bits.push_back({FixedBitsBefore(mask, bit_id), layer + layout.offset[child_mask]});
            }

            // Smallest free bit id, used solely to propagate "seen" up from the next level;
            // which free bit is used here doesn't affect correctness since the recursion
            // already folds in every other free bit by induction.
            const Cell* seen_child = free_bits.front().child;
            const uint16_t seen_bit_pos = free_bits.front().bit_pos;

            for (size_t values = 0; values < states_number; ++values) {
                const uint8_t state_seen =
                    static_cast<uint8_t>(SeenOf(seen_child[InsertFixedBit(values, seen_bit_pos, false)]) |
                                         SeenOf(seen_child[InsertFixedBit(values, seen_bit_pos, true)]));

                if (state_seen != 3) {
                    // Output already forced given the fixed bits (or nothing
                    // reaches this state) - no more queries needed.
                    row[values] = MakeCell<Cell>(state_seen, 0);
                    continue;
                }

                uint32_t best = Policy::Worst(bitness);
                if (zero_layer != nullptr) {
                    // Querying the helper splits this state by its value while
                    // keeping every bit the state has fixed, so both branches
                    // sit at the same (mask, values) one layer over.
                    const size_t state = layout.offset[mask] + values;
                    best = Policy::Combine(CostOf(zero_layer[state]), CostOf(one_layer[state]));
                }
                for (const FreeBit<Cell>& fb : free_bits) {
                    const uint32_t child_cost =
                        Policy::Combine(CostOf(fb.child[InsertFixedBit(values, fb.bit_pos, false)]),
                                        CostOf(fb.child[InsertFixedBit(values, fb.bit_pos, true)]));
                    best = std::min(best, child_cost);
                    if (best == 1) break;  // a single query is the global minimum
                }
                row[values] = MakeCell<Cell>(3, best);
            }
        }
    }
}

// `subset_table`, when given, is the indicator of the subset the tree has to be
// right on; without it the tree has to be right everywhere.
template <typename Policy>
size_t Solve(uint16_t bitness, const std::vector<bool>& truth_table, const std::vector<bool>* subset_table) {
    using Cell = typename Policy::Cell;

    assert(bitness <= kMaxSolvableBitness);
    assert(truth_table.size() == (size_t{1} << bitness));

    const StateLayout layout(bitness);
    std::vector<Cell> layer(layout.total_states);

    FillLeaves(layout, truth_table, subset_table, true, layer.data());
    FillLayer<Policy>(bitness, layout, layer.data(), nullptr, nullptr);

    // The root: nothing queried yet.
    return CostOf(layer[layout.offset[0]]);
}

template <typename Policy>
size_t SolveGiven(uint16_t bitness, const std::vector<bool>& truth_table, const std::vector<bool>& helper_table) {
    using Cell = typename Policy::Cell;

    assert(bitness <= kMaxSolvableBitness);
    assert(truth_table.size() == (size_t{1} << bitness));
    assert(helper_table.size() == truth_table.size());

    const StateLayout layout(bitness);

    // Three layers over the same states, since a path knows the helper's value
    // or does not: the two fixed-helper layers depend only on themselves and are
    // solved first, then the unqueried layer that may branch into them.
    std::vector<Cell> zero_layer(layout.total_states);
    std::vector<Cell> one_layer(layout.total_states);
    std::vector<Cell> layer(layout.total_states);

    FillLeaves(layout, truth_table, &helper_table, false, zero_layer.data());
    FillLeaves(layout, truth_table, &helper_table, true, one_layer.data());
    FillLeaves(layout, truth_table, nullptr, true, layer.data());

    FillLayer<Policy>(bitness, layout, zero_layer.data(), nullptr, nullptr);
    FillLayer<Policy>(bitness, layout, one_layer.data(), nullptr, nullptr);
    FillLayer<Policy>(bitness, layout, layer.data(), zero_layer.data(), one_layer.data());

    return CostOf(layer[layout.offset[0]]);
}

}  // namespace

size_t SolveForDepth(uint16_t bitness, const std::vector<bool>& truth_table) {
    return Solve<DepthPolicy>(bitness, truth_table, nullptr);
}

size_t SolveForSize(uint16_t bitness, const std::vector<bool>& truth_table) {
    return Solve<SizePolicy>(bitness, truth_table, nullptr);
}

size_t SolveForDepthGiven(uint16_t bitness, const std::vector<bool>& truth_table,
                          const std::vector<bool>& helper_table) {
    return SolveGiven<DepthPolicy>(bitness, truth_table, helper_table);
}

size_t SolveForSizeGiven(uint16_t bitness, const std::vector<bool>& truth_table,
                         const std::vector<bool>& helper_table) {
    return SolveGiven<SizePolicy>(bitness, truth_table, helper_table);
}

size_t SolveForDepthRestricted(uint16_t bitness, const std::vector<bool>& truth_table,
                               const std::vector<bool>& subset_table) {
    assert(subset_table.size() == truth_table.size());
    return Solve<DepthPolicy>(bitness, truth_table, &subset_table);
}

size_t SolveForSizeRestricted(uint16_t bitness, const std::vector<bool>& truth_table,
                              const std::vector<bool>& subset_table) {
    assert(subset_table.size() == truth_table.size());
    return Solve<SizePolicy>(bitness, truth_table, &subset_table);
}

}  // namespace tools
