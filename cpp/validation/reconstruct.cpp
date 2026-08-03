#include "reconstruct.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>
#include <set>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace func {

namespace {

// Every assignment of `slots` slots, ascending: the rows of a full truth table.
std::vector<SchemeInput> AllRows(size_t slots) {
    std::vector<SchemeInput> rows(size_t{1} << slots);
    for (size_t row = 0; row < rows.size(); ++row) {
        rows[row] = static_cast<SchemeInput>(row);
    }
    return rows;
}

// Fixed validation rows shared by every state: exhaustive within the budget,
// otherwise deterministically sampled with replacement.
const std::vector<SchemeInput>& ValidationRows(size_t bitness) {
    static const size_t drawn_for = bitness;
    static const std::vector<SchemeInput> rows = [bitness] {
        const size_t inputs = size_t{1} << bitness;
        if (inputs <= kValidationBudget) {
            return AllRows(bitness);
        }

        std::mt19937_64 rng(20260803);
        std::uniform_int_distribution<size_t> pick(0, inputs - 1);

        std::vector<SchemeInput> sampled(kValidationBudget);
        for (SchemeInput& row : sampled) {
            row = static_cast<SchemeInput>(pick(rng));
        }
        return sampled;
    }();

    // One run rebuilds one scheme, so the rows are drawn for one bitness.
    assert(bitness == drawn_for);
    return rows;
}

// Sorted hashes of the unbound slot functions over validation rows. Used for
// order-independent deduplication; sampled fingerprints may collide.
std::vector<uint64_t> SlotsFingerprint(const Scheme& scheme) {
    const SlotIds& unbound = scheme.Unbound();
    std::vector<uint64_t> hashes(unbound.size(), 1469598103934665603ULL);

    for (SchemeInput row : ValidationRows(scheme.InputCount(0))) {
        const Slots values = scheme.ComputeAll(row);
        for (size_t id = 0; id < unbound.size(); ++id) {
            hashes[id] = (hashes[id] ^ static_cast<uint64_t>(values[unbound[id]])) * 1099511628211ULL;
        }
    }

    std::sort(hashes.begin(), hashes.end());
    return hashes;
}

// Builds and scores a state's residual table. Rejects residuals that ignore a
// slot, since no completion could use it.
bool Assess(ReconstructionState& state, const Scheme& original) {
    // Exact scoring requires every assignment of the unbound slots.
    const Evaluation residual = state.Evaluate(AllRows(state.scheme.Unbound().size()), original);
    if (!residual.UsesEverySlot()) {
        return false;
    }

    state.score = Score(residual);
    return true;
}

// Stops heap growth while allowing already queued states to drain.
constexpr size_t kMaxProcessed = 4096;

// Orders the heap by tree depth, descending `log_size`, then slots remaining.
// Arguments are swapped because `std::push_heap` puts the greatest first.
struct BestOnTop {
    bool operator()(const ReconstructionState& lhs, const ReconstructionState& rhs) const {
        return std::make_tuple(rhs.score.depth, -rhs.score.log_size, rhs.scheme.Unbound().size()) <
               std::make_tuple(lhs.score.depth, -lhs.score.log_size, lhs.scheme.Unbound().size());
    }
};

// Starts with one identity slot per input and the target as the residual.
ReconstructionState InitialState(const Scheme& original) {
    const size_t bitness = original.InputCount(0);

    ReconstructionState state{Scheme(bitness)};
    assert(state.Validate(original) == 0);
    const bool assessed = Assess(state, original);
    assert(assessed);
    return state;
}

// Ascending combinations of `slots` taken `picked.size()` at a time.
bool NextCombination(std::vector<size_t>& picked, size_t slots) {
    for (size_t id = picked.size(); id-- > 0;) {
        if (picked[id] + (picked.size() - id) < slots) {
            ++picked[id];
            for (size_t next = id + 1; next < picked.size(); ++next) {
                picked[next] = picked[next - 1] + 1;
            }
            return true;
        }
    }
    return false;
}

// Logs the current state and cumulative search cost.
void PrintStep(const ReconstructionState& state, size_t heap_size, size_t pushed, size_t visited) {
    std::cout << "    ";
    if (state.scheme.depth == 0) {
        std::cout << "start";
    } else {
        std::cout << state.scheme.OperationAt(state.scheme.depth - 1);
    }
    std::cout << ", tree (depth " << state.score.depth << ", log size " << state.score.log_size << "), "
              << state.scheme.Unbound().size() << " slots left, heap " << heap_size << ", " << pushed << " pushed, "
              << visited << " slot sets seen\n";
}

}  // namespace

SchemeInput ReconstructionState::ConstructInputs(SchemeInput row, uint8_t level) const {
    const SlotIds& live_in = scheme.LiveIn(level);
    assert((row >> live_in.size()) == 0);

    Slots values(scheme.slots);
    for (size_t id = 0; id < live_in.size(); ++id) {
        values[live_in[id]] = ((row >> id) & 1U) != 0;
    }

    // Walk backward so each operation's required output is known before inversion.
    for (uint8_t lev_id = level; lev_id-- > 0;) {
        const OperationElement& element = scheme.OperationAt(lev_id);
        const op::OperationInput input = element.Preimage(values[element.output_id]);
        for (size_t id = 0; id < element.input_ids.size(); ++id) {
            values[element.input_ids[id]] = ((input >> id) & 1U) != 0;
        }
    }

    const SlotIds& inputs = scheme.LiveIn(0);
    SchemeInput input_row = 0;
    for (size_t id = 0; id < inputs.size(); ++id) {
        input_row |= static_cast<SchemeInput>(values[inputs[id]]) << id;
    }
    return input_row;
}

Evaluation ReconstructionState::Evaluate(const std::vector<SchemeInput>& rows, const Scheme& original) const {
    Evaluation result;

    for (SchemeInput row : rows) {
        // Unbound slots are live-in at the scheme's current depth.
        const SchemeInput input_row = ConstructInputs(row, scheme.depth);
        result.Append(row, original.ComputeValue(input_row));
    }

    return result;
}

ReconstructionState ReconstructionState::Grow(const op::Operation& operation, const std::vector<size_t>& picked) const {
    assert(operation.kArity == picked.size());
    assert(IsCompatibleWith(operation));
    // Preserves the ascending slot order required by `AddOperation`.
    assert(std::is_sorted(picked.begin(), picked.end()));
    assert(picked.back() < scheme.Unbound().size());

    const SlotIds& unbound = scheme.Unbound();
    std::vector<size_t> input_ids;
    for (size_t id : picked) {
        input_ids.push_back(unbound[id]);
    }

    Scheme next = scheme;
    next.AddOperation(operation, input_ids);

    return ReconstructionState{std::move(next)};
}

bool ReconstructionState::IsAssembled(const Scheme& original) const {
    if (!scheme.IsCompleted()) {
        return false;
    }

    // Completion also requires agreement with `original` on validation rows.
    for (SchemeInput row : ValidationRows(original.InputCount(0))) {
        if (scheme.ComputeValue(row) != original.ComputeValue(row)) {
            return false;
        }
    }

    return true;
}

size_t ReconstructionState::Validate(const Scheme& original) const {
    const std::vector<SchemeInput>& rows = ValidationRows(original.InputCount(0));

    // Sparse mapping from unbound values to their required target.
    std::unordered_map<size_t, bool> seen;
    seen.reserve(rows.size());

    size_t mismatches = 0;
    for (SchemeInput row : rows) {
        const size_t vector_id = scheme.ComputeUnbound(row);
        const bool value = original.ComputeValue(row);

        const auto recorded = seen.find(vector_id);
        if (recorded == seen.end()) {
            seen.emplace(vector_id, value);
        } else if (recorded->second != value) {
            // These unbound values already stand for the other target.
            ++mismatches;
        }
    }

    return mismatches;
}

ReconstructionState Reconstruct(const Scheme& original) {
    ReconstructionState initial = InitialState(original);

    // Deduplicate states by the functions their slots compute.
    std::set<std::vector<uint64_t>> visited;
    visited.insert(SlotsFingerprint(initial.scheme));

    std::vector<ReconstructionState> heap;
    heap.push_back(std::move(initial));
    std::push_heap(heap.begin(), heap.end(), BestOnTop{});

    size_t pushed = 1;

    while (!heap.empty()) {
        std::pop_heap(heap.begin(), heap.end(), BestOnTop{});
        ReconstructionState state = std::move(heap.back());
        heap.pop_back();

        PrintStep(state, heap.size(), pushed, visited.size());
        if (state.IsAssembled(original)) {
            return state;
        }

        for (const op::Operation& operation : op::kOperations) {
            if (!state.IsCompatibleWith(operation)) {
                continue;
            }

            std::vector<size_t> picked(operation.kArity);
            for (size_t id = 0; id < picked.size(); ++id) {
                picked[id] = id;
            }

            do {
                // Stop adding at the budget, but search all queued states.
                if (pushed >= kMaxProcessed) {
                    break;
                }

                ReconstructionState next = state.Grow(operation, picked);
                // Deduplicate and validate before paying for exact assessment.
                if (!visited.insert(SlotsFingerprint(next.scheme)).second || next.Validate(original) != 0 ||
                    !Assess(next, original)) {
                    continue;
                }

                heap.push_back(std::move(next));
                std::push_heap(heap.begin(), heap.end(), BestOnTop{});
                ++pushed;
            } while (NextCombination(picked, state.scheme.Unbound().size()));
        }
    }

    std::cerr << "search ran out of states after " << pushed << " of them\n";
    std::abort();
}

}  // namespace func
