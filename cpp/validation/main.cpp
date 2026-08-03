#include <algorithm>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "reconstruct.h"
#include "scheme.h"
#include "tree_scorer.h"

using func::op::Operation;
using func::op::OperationInput;
using func::op::kOperations;

// Candidate generation walks combinations of unbound slots rather than
// orderings, and drops a state whose residual ignores an unbound slot. Both
// hold only because every operation is symmetric in its arguments and sensitive
// in each of them, so check that instead of trusting `op::kOperations`.
void CheckOperations() {
    for (const Operation& operation : kOperations) {
        const OperationInput inputs = static_cast<OperationInput>(OperationInput{1} << operation.kArity);
        std::vector<bool> sensitive(operation.kArity, false);

        for (OperationInput input = 0; input < inputs; ++input) {
            const OperationInput sorted = static_cast<OperationInput>((1U << std::popcount(input)) - 1);
            assert(operation(input) == operation(sorted));

            for (uint8_t argument = 0; argument < operation.kArity; ++argument) {
                const OperationInput flipped = static_cast<OperationInput>(input ^ (1U << argument));
                sensitive[argument] = sensitive[argument] || operation(input) != operation(flipped);
            }
        }

        assert(std::all_of(sensitive.begin(), sensitive.end(), [](bool bit) { return bit; }));
    }
}


// The rebuilt scheme has to compute the very function it was rebuilt from. The
// search does not check that -- nothing in it ever compares the two schemes --
// so counting the assignments they disagree on is this binary's job.
size_t Mismatches(const func::Evaluation& rebuilt, const func::Evaluation& target) {
    assert(rebuilt.rows == target.rows);

    size_t mismatches = 0;
    for (size_t row = 0; row < target.Rows(); ++row) {
        mismatches += rebuilt.values[row] != target.values[row];
    }
    return mismatches;
}

int main(int argc, char** argv) {
    assert(argc <= 3);
    CheckOperations();

    const std::string bitness_str = argc >= 2 ? argv[1] : "12";
    const std::string seed_str = argc >= 3 ? argv[2] : "122";

    const size_t bitness = std::stoul(bitness_str);
    const uint64_t seed = std::stoull(seed_str);

    assert(func::kMinBitness <= bitness && bitness <= func::kMaxBitness);

    const func::Scheme scheme = func::RandomScheme(bitness, seed);
    std::cout << "Random scheme over " << bitness << " inputs (seed " << seed << "), " << int(scheme.depth)
              << " gates:\n"
              << scheme;

    const func::Evaluation target = func::Tabulate(scheme);
    const func::TreeScore initial_score = func::Score(target);

    std::cout << "\n  search:\n";
    const func::ReconstructionState result = func::Reconstruct(scheme);

    std::cout << "\n  reconstructed scheme, " << int(result.scheme.depth) << " gates:\n" << result.scheme;

    std::cout << "  tree score: (depth " << initial_score.depth << ", log size " << initial_score.log_size
              << ") -> (depth " << result.score.depth << ", log size " << result.score.log_size << ")\n";

    const size_t rows = target.Rows();
    const size_t mismatches = Mismatches(func::Tabulate(result.scheme), target);
    std::cout << "  verified: " << rows - mismatches << "/" << rows << " input assignments\n";
    assert(mismatches == 0);

    return 0;
}
