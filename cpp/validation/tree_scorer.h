#pragma once

// Scores optimal decision trees on the model's training scale. Dense truth
// tables and exact solving live here and are created only when scored.

#include <cassert>
#include <cstddef>
#include <vector>

#include "scheme.h"

namespace func {

// Function values at selected rows. Bit `k` of a row is slot `k`'s value.
// Scoring is exact, so an evaluation still has to cover every row.
struct Evaluation {
    std::vector<SchemeInput> rows;
    std::vector<bool> values;

    void Append(SchemeInput row, bool value) {
        rows.push_back(row);
        values.push_back(value);
    }

    size_t Rows() const {
        assert(rows.size() == values.size());
        return rows.size();
    }

    // log2 of the row count.
    size_t Slots() const;

    // Whether changing every slot can affect the function.
    bool UsesEverySlot() const;
};

// Evaluates a completed scheme at every one of its inputs.
Evaluation Tabulate(const Scheme& scheme);

// Optimal tree cost: query depth and `log2(2^slots - size)`. Cheaper trees have
// smaller `depth` but larger `log_size`.
struct TreeScore {
    size_t depth = 0;
    double log_size = 0.0;
};

// Scores the optimal decision tree for `evaluation`.
TreeScore Score(const Evaluation& evaluation);

}  // namespace func
