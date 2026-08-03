#pragma once

// Rebuilds a `Scheme` from the truth table of another one by a best-first search
// over partially grown schemes: progress is measured by how cheap a decision
// tree over the slots at hand is.
//
// A search state is a scheme that still has several unbound slots. Every
// operation added later reads unbound slots only, so the target has to be a
// function of the current unbound slot values. That residual function is what a
// state is scored by -- the `TreeScore` of its optimal decision tree, compared
// lexicographically -- and its absence is what prunes: once two input rows share
// their unbound slot values but disagree on the target, the scheme has dropped a
// distinction no completion can bring back.

#include <cstddef>
#include <vector>

#include "scheme.h"
#include "tree_scorer.h"

namespace func {

// The truth table of a function over an ordered set of slots: bit `id` of an
// entry's index is the value of slot `id`, so a table over `k` slots has 2^k
// entries. The target is the table over a scheme's inputs, and what a partially
// grown scheme still has to compute is the table over its unbound slots.
struct TruthTable {
    std::vector<bool> values;

    // The slots the table is over: the log2 of its entry count.
    size_t Slots() const;

    bool IsTrivial() const {
        return values.size() == 2 && !values[0] && values[1];
    }
};

// What one search run reached, plus what it cost to get there.
struct Reconstruction {
    // The rebuilt scheme; it need not be the one the target came from, only one
    // computing the same function.
    Scheme scheme;
    // Score of the bare inputs the search started from.
    TreeScore initial_score;
    // Score of the completed scheme: a single slot holding the target, one
    // query over one node, so (1, log2(2 - 1) = 0).
    TreeScore final_score;
    // States taken off the heap, put on it, and slot sets ever reached.
    size_t expansions = 0;
    size_t pushed = 0;
    size_t visited = 0;
    // Input assignments where the rebuilt scheme disagrees with the target;
    // zero, and checked rather than assumed since Release builds drop asserts.
    size_t mismatches = 0;
};

// The function a completed scheme computes, tabulated over its input slots.
TruthTable Tabulate(const Scheme& scheme);

// Rebuilds a scheme from `original`'s truth table alone, logging every expansion
// to stdout. Aborts if the search runs out of states or of patience.
Reconstruction Reconstruct(const Scheme& original);

}  // namespace func
