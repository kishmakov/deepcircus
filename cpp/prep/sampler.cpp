#include "sampler.h"

#include <cassert>
#include <cstdint>
#include <limits>
#include <vector>

#include "func/table.h"
#include "func/tree.h"
#include "func/tt.h"
#include "tools/random.h"
#include "tools/solver.h"

namespace prep {

namespace {

offline::Function Serialize(offline::FunctionKind kind, const func::Func& function) {
    return offline::Function{kind, function.serialize()};
}

func::TreeFunc StorableTree(uint16_t tree_bitness, uint16_t target_bitness, tools::Random& random) {
    while (true) {
        func::TreeFunc tree(tree_bitness, random.Next());
        const bool size_fits_target =
            target_bitness >= std::numeric_limits<uint16_t>::digits || tree.Size() < (uint32_t{1} << target_bitness);
        if (tree.Depth() <= target_bitness && tree.Size() < offline::kUnknownSize && size_fits_target) return tree;
    }
}

offline::Entry WithTargets(Model model, uint16_t bitness, offline::FunctionKind kind, const func::Func& target,
                           const func::TableFunc& condition, uint32_t depth, uint32_t size) {
    if (bitness <= tools::kMaxSolvableBitness) {
        const std::vector<bool> target_table = target.TruthTable();
        const std::vector<bool> condition_table = condition.TruthTable();
        if (model == Model::kM1) {
            depth = tools::SolveForDepthGiven(bitness, target_table, condition_table);
            size = tools::SolveForSizeGiven(bitness, target_table, condition_table);
        } else {
            depth = tools::SolveForDepthRestricted(bitness, target_table, condition_table);
            size = tools::SolveForSizeRestricted(bitness, target_table, condition_table);
        }
    }

    assert(depth <= offline::kUnknownDepth);
    assert(size <= offline::kUnknownSize);
    assert((depth == offline::kUnknownDepth) == (size == offline::kUnknownSize));
    return offline::Entry{Serialize(kind, target), Serialize(offline::FunctionKind::kTable, condition),
                          static_cast<uint8_t>(depth), static_cast<uint16_t>(size)};
}

}  // namespace

offline::Entry TTEntry(const Parameters& parameters, Model model, uint16_t bitness, uint32_t index) {
    assert(model == Model::kM1 || model == Model::kM2);
    assert(bitness >= offline::kMinBitness && bitness <= offline::kMaxBitness);
    tools::Random random(tools::EntrySeed(parameters.seed, static_cast<uint16_t>(model), bitness, index));
    const func::TableFunc condition(bitness, random.Next());
    const func::TreeFunc witness = StorableTree(bitness + 1, bitness, random);
    const func::TTFunc target(bitness, condition, witness);
    // On M2's subset the attached table is one, so helper queries can be removed.
    return WithTargets(model, bitness, offline::FunctionKind::kTreeOverTable, target, condition, witness.Depth(),
                       witness.Size());
}

offline::Entry GeneralEntry(const Parameters& parameters, Model model, uint16_t bitness, uint32_t index) {
    assert(model == Model::kM1 || model == Model::kM2);
    assert(bitness >= offline::kMinBitness && bitness <= offline::kMaxBitness);
    tools::Random random(tools::EntrySeed(parameters.seed, static_cast<uint16_t>(model), bitness, index));
    const func::TableFunc target(bitness, random.Next());
    const func::TableFunc condition(bitness, random.Next());
    return WithTargets(model, bitness, offline::FunctionKind::kTable, target, condition, offline::kUnknownDepth,
                       offline::kUnknownSize);
}

}  // namespace prep
