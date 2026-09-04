#include "sampler.h"

#include <cassert>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "func/table.h"
#include "func/tree.h"
#include "func/tt.h"
#include "tools/random.h"
#include "tools/solver.h"

namespace preparation {

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

offline::Entry M1Solved(const Parameters& parameters, uint16_t bitness, uint32_t index) {
    tools::Random random(tools::EntrySeed(parameters.seed, static_cast<uint16_t>(Model::kM1), bitness, index));
    const uint64_t table_seed = random.Next();
    const func::TableFunc helper(bitness, table_seed);
    const func::TreeFunc witness = StorableTree(bitness + 1, bitness, random);
    const func::TTFunc target(bitness, func::TableFunc(bitness, table_seed), witness);

    uint32_t depth = witness.Depth();
    uint32_t size = witness.Size();
    if (bitness <= tools::kMaxSolvableBitness) {
        const std::vector<bool> target_table = target.TruthTable();
        const std::vector<bool> helper_table = helper.TruthTable();
        depth = tools::SolveForDepthGiven(bitness, target_table, helper_table);
        size = tools::SolveForSizeGiven(bitness, target_table, helper_table);
    }

    assert(depth < offline::kUnknownDepth);
    assert(size < offline::kUnknownSize);
    return offline::Entry{Serialize(offline::FunctionKind::kTreeOverTable, target),
                          Serialize(offline::FunctionKind::kTable, helper), static_cast<uint8_t>(depth),
                          static_cast<uint16_t>(size)};
}

offline::Entry M2Solved(const Parameters& parameters, uint16_t bitness, uint32_t index) {
    tools::Random random(tools::EntrySeed(parameters.seed, static_cast<uint16_t>(Model::kM2), bitness, index));
    const func::TableFunc subset(bitness, random.Next());

    if (bitness <= tools::kMaxSolvableBitness) {
        const func::TableFunc target(bitness, random.Next());
        const std::vector<bool> target_table = target.TruthTable();
        const std::vector<bool> subset_table = subset.TruthTable();
        const uint32_t depth = tools::SolveForDepthRestricted(bitness, target_table, subset_table);
        const uint32_t size = tools::SolveForSizeRestricted(bitness, target_table, subset_table);
        return offline::Entry{Serialize(offline::FunctionKind::kTable, target),
                              Serialize(offline::FunctionKind::kTable, subset), static_cast<uint8_t>(depth),
                              static_cast<uint16_t>(size)};
    }

    const func::TreeFunc target = StorableTree(bitness, bitness, random);
    return offline::Entry{Serialize(offline::FunctionKind::kTree, target),
                          Serialize(offline::FunctionKind::kTable, subset), static_cast<uint8_t>(target.Depth()),
                          static_cast<uint16_t>(target.Size())};
}

offline::Entry Unknown(const Parameters& parameters, Model model, uint16_t bitness, uint32_t index) {
    assert(bitness > tools::kMaxSolvableBitness);
    tools::Random random(tools::EntrySeed(parameters.seed, static_cast<uint16_t>(model), bitness, index));
    const func::TableFunc target(bitness, random.Next());
    const func::TableFunc condition(bitness, random.Next());
    return offline::Entry{Serialize(offline::FunctionKind::kTable, target),
                          Serialize(offline::FunctionKind::kTable, condition), offline::kUnknownDepth,
                          offline::kUnknownSize};
}

}  // namespace

offline::Entry SolvedEntry(const Parameters& parameters, Model model, uint16_t bitness, uint32_t index) {
    assert(bitness >= offline::kMinBitness && bitness <= offline::kMaxBitness);
    if (model == Model::kM1) return M1Solved(parameters, bitness, index);
    assert(model == Model::kM2);
    return M2Solved(parameters, bitness, index);
}

offline::Entry UnsolvedEntry(const Parameters& parameters, Model model, uint16_t bitness, uint32_t index) {
    assert(model == Model::kM1 || model == Model::kM2);
    assert(bitness >= offline::kMinBitness && bitness <= offline::kMaxBitness);
    return Unknown(parameters, model, bitness, index);
}

}  // namespace preparation
