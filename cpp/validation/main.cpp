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

    // The rebuilt scheme has to compute the very function it was rebuilt from,
    // so far as the validation rows can tell.
    const size_t mismatches = result.Validate(scheme);
    assert(mismatches == 0);
    std::cout << "  verified: no mismatch over up to " << func::kValidationBudget << " validation rows\n";

    return 0;
}
