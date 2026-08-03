#pragma once

// Rebuilds a `Scheme` with best-first search over partial schemes, scored by
// the optimal decision tree for the function left on their unbound slots.
// States that lose a target distinction cannot be completed and are pruned.
// Residuals are computed lazily; states keep only their schemes.

#include <cstddef>
#include <utility>
#include <vector>

#include "scheme.h"
#include "tree_scorer.h"

namespace func {

// Validation is exhaustive up to this many rows and sampled above it.
constexpr size_t kValidationBudget = 4096;

// A partial scheme and the decision-tree score of its residual function.
struct ReconstructionState {
    Scheme scheme;
    TreeScore score;

    explicit ReconstructionState(Scheme scheme) : scheme(std::move(scheme)) {}

    // Whether the scheme is complete and matches `original` on validation rows.
    bool IsAssembled(const Scheme& original) const;

    // Whether enough unbound slots remain for `operation`.
    bool IsCompatibleWith(const op::Operation& operation) const { return operation.kArity <= scheme.Unbound().size(); }

    // Finds an input whose live slots at `level` equal `row`.
    SchemeInput ConstructInputs(SchemeInput row, uint8_t level) const;

    // Evaluates the residual function at `rows`.
    Evaluation Evaluate(const std::vector<SchemeInput>& rows, const Scheme& original) const;

    // Adds `operation` over the unbound slots at `picked`.
    ReconstructionState Grow(const op::Operation& operation, const std::vector<size_t>& picked) const;

    // Checks that the target remains a function of the unbound slots.
    bool Validate(const Scheme& original) const;
};

// Rebuilds an equivalent scheme from `original`'s behavior, logging expansions.
// Aborts if the search exhausts its states or patience.
ReconstructionState Reconstruct(const Scheme& original);

}  // namespace func
