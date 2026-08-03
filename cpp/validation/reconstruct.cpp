#include "reconstruct.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <optional>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

namespace func {

namespace {

// One truth-table column: the value a slot takes at every assignment of the
// original inputs.
using Column = std::vector<bool>;

// The columns of a state's unbound slots, in `scheme.Unbound()` order. They are
// what turns an input row into an entry of a table over those slots, and the
// only thing a state's future depends on.
struct Columns {
    std::vector<Column> slots;

    // One column per input slot, each holding that input's own value.
    static Columns Inputs(size_t bitness);

    size_t Slots() const { return slots.size(); }
    size_t Rows() const { return slots.empty() ? 0 : slots.front().size(); }

    // Where input row `row` lands in a table over these slots: bit `id` of the
    // index is the value slot `id` takes at that row.
    size_t VectorAt(size_t row) const {
        size_t vector_id = 0;
        for (size_t id = 0; id < slots.size(); ++id) {
            vector_id |= static_cast<size_t>(slots[id][row]) << id;
        }
        return vector_id;
    }

    // These columns with `picked` consumed and the column `operation` computes
    // from them appended -- what `AddOperation` does to the scheme's slots.
    Columns Apply(const op::Operation& operation, const std::vector<size_t>& picked) const;

    // The same columns in a canonical order, so that schemes reaching the same
    // slots by different routes are searched once.
    Columns Sorted() const {
        Columns sorted = *this;
        std::sort(sorted.slots.begin(), sorted.slots.end());
        return sorted;
    }

    // The search keys a `std::set` on them.
    bool operator<(const Columns& other) const { return slots < other.slots; }
};

Columns Columns::Inputs(size_t bitness) {
    const size_t bit_vecs = size_t{1} << bitness;

    Columns columns{std::vector<Column>(bitness, Column(bit_vecs))};
    for (size_t bit_vec = 0; bit_vec < bit_vecs; ++bit_vec) {
        for (size_t input_id = 0; input_id < bitness; ++input_id) {
            columns.slots[input_id][bit_vec] = ((bit_vec >> input_id) & 1U) != 0;
        }
    }
    return columns;
}

Columns Columns::Apply(const op::Operation& operation, const std::vector<size_t>& picked) const {
    assert(operation.kArity == picked.size());

    Column column(Rows());
    for (size_t row = 0; row < column.size(); ++row) {
        op::OperationInput input = 0;
        for (size_t id = 0; id < picked.size(); ++id) {
            input |= static_cast<op::OperationInput>(slots[picked[id]][row]) << id;
        }
        column[row] = operation(input);
    }

    Columns next;
    next.slots.reserve(Slots() - picked.size() + 1);
    for (size_t id = 0; id < Slots(); ++id) {
        if (!std::binary_search(picked.begin(), picked.end(), id)) {
            next.slots.push_back(slots[id]);
        }
    }
    next.slots.push_back(std::move(column));
    return next;
}

// Enough for the searches that stay affordable; a search that hits it is a bug
// report, not a result.
constexpr size_t kMaxExpansions = 4096;

struct State {
    Scheme scheme;
    // One column per unbound slot, in `scheme.Unbound()` order.
    Columns columns;
    // The target over those unbound slots -- what is left to compute.
    TruthTable residual;
    TreeScore score;
    // Order of insertion into the heap, the last tie-break -- it is what
    // makes the search deterministic.
    size_t order = 0;

    // One column per unbound slot is what the whole search rests on, so every
    // state goes through here.
    State(Scheme scheme, Columns columns, TruthTable residual, TreeScore score, size_t order)
        : scheme(std::move(scheme)),
          columns(std::move(columns)),
          residual(std::move(residual)),
          score(score),
          order(order) {
        assert(this->scheme.Unbound().size() == this->columns.Slots());
    }

    // A completed scheme whose only unbound slot is the target itself; the
    // identity residual is what tells it from the one computing the complement.
    bool IsAssembled() const { return scheme.IsCompleted() && residual.IsTrivial(); }

    // Whether the state has slots enough for `operation` to read.
    bool IsCompatibleWith(const op::Operation& operation) const { return operation.kArity <= columns.Slots(); }

    // The state one `operation` on the unbound slots at `picked` away from this
    // one, numbered `order`. Nothing comes back when the step is not worth
    // searching: when `visited` already holds the slots it reaches, when it
    // drops a distinction the target needs, or when it leaves a slot the target
    // ignores.
    std::optional<State> Grow(const op::Operation& operation, const std::vector<size_t>& picked,
                              const TruthTable& target, std::set<Columns>& visited, size_t order) const;
};

// Best first: the cheapest decision tree wins, then the scheme closest to being
// completed, then the older state. A cheaper tree scores a *higher* `log_size`,
// hence the negation.
struct ByScore {
    bool operator()(const State& lhs, const State& rhs) const {
        return std::make_tuple(lhs.score.depth, -lhs.score.log_size, lhs.columns.Slots(), lhs.order) <
               std::make_tuple(rhs.score.depth, -rhs.score.log_size, rhs.columns.Slots(), rhs.order);
    }
};

// The target seen as a function of the unbound slots. Fails when two input rows
// with equal slot values disagree on the target: the scheme cannot tell those
// rows apart any more, so no completion of it computes the target.
bool Residual(const Columns& columns, const TruthTable& target, TruthTable& residual) {
    const size_t dst_vecs = size_t{1} << columns.Slots();
    std::vector<int8_t> dst_values(dst_vecs, -1);

    for (size_t bit_vec = 0; bit_vec < target.values.size(); ++bit_vec) {
        const size_t vector_id = columns.VectorAt(bit_vec);
        const int8_t value = target.values[bit_vec];
        if (dst_values[vector_id] < 0) {
            dst_values[vector_id] = value;
        } else if (dst_values[vector_id] != value) {
            return false;
        }
    }

    residual.values.assign(dst_vecs, false);
    for (size_t bit_vec = 0; bit_vec < dst_vecs; ++bit_vec) {
        // Check that every unbound combination is reachable.
        assert(dst_values[bit_vec] >= 0);
        residual.values[bit_vec] = dst_values[bit_vec] != 0;
    }
    return true;
}

// Every operation is sensitive in each argument, so a completed scheme depends
// on every slot it was grown from. A residual that ignores one is a dead end.
bool UsesEverySlot(const TruthTable& residual) {
    for (size_t id = 0; id < residual.Slots(); ++id) {
        const size_t bit = size_t{1} << id;
        bool sensitive = false;
        for (size_t vector_id = 0; vector_id < residual.values.size() && !sensitive; ++vector_id) {
            sensitive = residual.values[vector_id] != residual.values[vector_id ^ bit];
        }
        if (!sensitive) {
            return false;
        }
    }
    return true;
}

std::optional<State> State::Grow(const op::Operation& operation, const std::vector<size_t>& picked,
                                 const TruthTable& target, std::set<Columns>& visited, size_t order) const {
    Columns grown = columns.Apply(operation, picked);
    if (!visited.insert(grown.Sorted()).second) {
        return std::nullopt;
    }

    TruthTable residual;
    if (!Residual(grown, target, residual) || !UsesEverySlot(residual)) {
        return std::nullopt;
    }

    // Both `picked` and `Unbound()` are ascending, so the slot ids come out
    // increasing, which is how `AddOperation` wants them.
    const SlotIds& unbound = scheme.Unbound();
    std::vector<size_t> input_ids;
    for (size_t id : picked) {
        input_ids.push_back(unbound[id]);
    }

    Scheme next = scheme;
    next.AddOperation(operation, input_ids);

    const TreeScore score = Score(residual.values);
    return State{std::move(next), std::move(grown), std::move(residual), score, order};
}

// The bare inputs the search starts from: one slot per input, each holding it.
State InitialState(const TruthTable& target) {
    const size_t bitness = target.Slots();

    State state{Scheme(bitness), Columns::Inputs(bitness), {}, {}, 0};
    const bool feasible = Residual(state.columns, target, state.residual);
    assert(feasible);
    state.score = Score(state.residual.values);
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

void PrintStep(size_t expansion, const State& state, size_t heap_size) {
    std::cout << "    step " << expansion << ": ";
    if (state.scheme.depth == 0) {
        std::cout << "start";
    } else {
        std::cout << state.scheme.OperationAt(state.scheme.depth - 1);
    }
    std::cout << ", tree (depth " << state.score.depth << ", log size " << state.score.log_size << "), "
              << state.columns.Slots() << " slots left, heap " << heap_size << '\n';
}

size_t Mismatches(const Scheme& scheme, const TruthTable& target) {
    const TruthTable table = Tabulate(scheme);
    assert(table.values.size() == target.values.size());

    size_t mismatches = 0;
    for (size_t row = 0; row < target.values.size(); ++row) {
        mismatches += table.values[row] != target.values[row];
    }
    return mismatches;
}

}  // namespace

size_t TruthTable::Slots() const {
    assert(std::has_single_bit(values.size()));
    return static_cast<size_t>(std::countr_zero(values.size()));
}

TruthTable Tabulate(const Scheme& scheme) {
    const size_t rows = size_t{1} << scheme.InputCount(0);
    const size_t output_id = scheme.OutputId();

    TruthTable table{std::vector<bool>(rows)};
    for (size_t row = 0; row < rows; ++row) {
        table.values[row] = scheme.Compute(static_cast<SchemeInput>(row))[output_id];
    }
    return table;
}

Reconstruction Reconstruct(const Scheme& original) {
    const TruthTable target = Tabulate(original);

    State initial = InitialState(target);
    const TreeScore initial_score = initial.score;

    std::set<Columns> visited;
    visited.insert(initial.columns.Sorted());

    std::set<State, ByScore> heap;
    heap.insert(std::move(initial));

    size_t expansions = 0;
    size_t pushed = 1;

    while (!heap.empty() && expansions < kMaxExpansions) {
        State state = std::move(heap.extract(heap.begin()).value());
        PrintStep(expansions, state, heap.size());
        if (state.IsAssembled()) {
            // Read before `state.scheme` is moved out below.
            const size_t mismatches = Mismatches(state.scheme, target);
            return Reconstruction{
                .scheme = std::move(state.scheme),
                .initial_score = initial_score,
                .final_score = state.score,
                .expansions = expansions,
                .pushed = pushed,
                .visited = visited.size(),
                .mismatches = mismatches,
            };
        }
        ++expansions;

        for (const op::Operation& operation : op::kOperations) {
            if (!state.IsCompatibleWith(operation)) {
                continue;
            }

            std::vector<size_t> picked(operation.kArity);
            for (size_t id = 0; id < picked.size(); ++id) {
                picked[id] = id;
            }

            do {
                std::optional<State> next = state.Grow(operation, picked, target, visited, pushed);
                if (next) {
                    heap.insert(std::move(*next));
                    ++pushed;
                }
            } while (NextCombination(picked, state.columns.Slots()));
        }
    }

    std::cerr << "search gave up after " << expansions << " expansions\n";
    std::abort();
}

}  // namespace func
