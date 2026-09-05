#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include "tools/solver.h"

namespace {

enum class Kind { kDepth, kSize };

size_t Combine(Kind kind, size_t zero, size_t one) {
    return kind == Kind::kDepth ? 1 + std::max(zero, one) : 1 + zero + one;
}

// Exponential reference solver: carries the assignments still reaching this
// node and tries every query the path has not spent yet. `helper_table`, when
// given, is queryable once per path just like an input bit; the assignments the
// tree owes nothing on are simply left out of the starting `points`.
size_t BruteForce(Kind kind, uint16_t bitness, const std::vector<bool>& truth_table,
                  const std::vector<bool>* helper_table, const std::vector<size_t>& points, uint32_t queried,
                  bool helper_queried) {
    if (points.empty()) return 0;
    if (std::all_of(points.begin(), points.end(), [&](size_t p) { return truth_table[p] == truth_table[points[0]]; })) {
        return 0;
    }

    size_t best = std::numeric_limits<size_t>::max();

    for (uint16_t bit_id = 0; bit_id < bitness; ++bit_id) {
        const uint32_t bit = uint32_t{1} << bit_id;
        if (queried & bit) continue;

        std::vector<size_t> sides[2];
        for (size_t point : points) sides[(point >> bit_id) & 1].push_back(point);
        best = std::min(
            best, Combine(kind,
                          BruteForce(kind, bitness, truth_table, helper_table, sides[0], queried | bit, helper_queried),
                          BruteForce(kind, bitness, truth_table, helper_table, sides[1], queried | bit,
                                     helper_queried)));
    }

    if (helper_table != nullptr && !helper_queried) {
        std::vector<size_t> sides[2];
        for (size_t point : points) sides[(*helper_table)[point]].push_back(point);
        best = std::min(best,
                        Combine(kind, BruteForce(kind, bitness, truth_table, helper_table, sides[0], queried, true),
                                BruteForce(kind, bitness, truth_table, helper_table, sides[1], queried, true)));
    }

    return best;
}

size_t Reference(Kind kind, uint16_t bitness, const std::vector<bool>& truth_table,
                 const std::vector<bool>* helper_table, const std::vector<bool>* subset_table) {
    std::vector<size_t> points;
    for (size_t point = 0; point < (size_t{1} << bitness); ++point) {
        if (subset_table == nullptr || (*subset_table)[point]) points.push_back(point);
    }
    return BruteForce(kind, bitness, truth_table, helper_table, points, 0, false);
}

// Bit `values` of `bits` is the table's output at assignment `values`, so a
// literal table is limited to bitness 6.
std::vector<bool> TableFromBits(uint16_t bitness, uint64_t bits) {
    std::vector<bool> table(size_t{1} << bitness);
    for (size_t values = 0; values < table.size(); ++values) table[values] = (bits >> values) & 1;
    return table;
}

// One truth table off a shared stream, a 64-bit draw per 64 assignments.
std::vector<bool> SeededTable(uint16_t bitness, std::mt19937_64& rng) {
    std::vector<bool> table(size_t{1} << bitness);
    uint64_t word = 0;
    for (size_t values = 0; values < table.size(); ++values) {
        if (values % 64 == 0) word = rng();
        table[values] = (word >> (values % 64)) & 1;
    }
    return table;
}

std::pair<std::vector<bool>, std::vector<bool>> SeededPair(uint16_t bitness, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<bool> truth_table = SeededTable(bitness, rng);
    std::vector<bool> second_table = SeededTable(bitness, rng);
    return {std::move(truth_table), std::move(second_table)};
}

// Every target of one truth table plus one second table, read both ways the
// offline series read theirs: as the helper function f of `S_1[g | f]`, and as
// the indicator of the subset X of `S_2[g | X]`.
struct SolverTargets {
    size_t depth;
    size_t size;
    size_t given_depth;
    size_t given_size;
    size_t restricted_depth;
    size_t restricted_size;
};

struct SolverGoldenCase {
    uint16_t bitness;
    uint64_t truth_bits;
    uint64_t second_bits;
    SolverTargets targets;
};

// Each entry is reproduced by the brute-force reference below; the ones marked
// out are also readable by hand, and the rest pin the solvers against refactors.
constexpr SolverGoldenCase kSolverGoldenCases[] = {
    // Parity of 3 bits with x0 ^ x1: as a helper it leaves x2 to query, and on
    // the half where it is 1 the parity is just ~x2.
    {3, 0x96, 0x66, {3, 7, 2, 3, 1, 1}},
    // NOR of x0, x1 once restricted to x2 = 1.
    {3, 0x1b, 0xf0, {2, 3, 2, 3, 2, 2}},
    // Parity of 4 bits with x2 ^ x3: either reading leaves x0 ^ x1 to compute.
    {4, 0x6996, 0xff0, {4, 15, 3, 7, 2, 3}},
    {4, 0xa53c, 0xcccc, {3, 7, 3, 7, 3, 5}},
    {5, 0x5a3c96f0, 0xf0ff0f0, {5, 17, 4, 11, 4, 8}},
    {5, 0xdeadbeef, 0x12345678, {5, 14, 4, 12, 3, 5}},
    // A negated 4:1 multiplexer with its own negation: as a helper one query
    // decides, and as a subset it is exactly where the function is constant 0.
    {6, 0x0123456789abcdefULL, 0xfedcba9876543210ULL, {3, 7, 1, 1, 0, 0}},
    // "All bits equal" with ~x0: as a helper it is redundant, while as a subset
    // it drops the all-ones point and leaves a five-query chain.
    {6, 0x8000000000000001ULL, 0x5555555555555555ULL, {6, 11, 6, 11, 5, 5}},
};

struct SolverSeededGoldenCase {
    uint16_t bitness;
    uint64_t seed;
    SolverTargets targets;
};

// The same targets at bitnesses too large to spell a table out for, over the
// table pair `SeededPair` draws.
constexpr SolverSeededGoldenCase kSolverSeededGoldenCases[] = {
    {8, 1, {8, 106, 7, 98, 7, 54}},      {8, 42, {8, 101, 8, 95, 7, 48}},
    {10, 1, {10, 426, 9, 385, 9, 200}},  {10, 42, {10, 420, 9, 376, 8, 190}},
    {12, 1, {12, 1673, 11, 1510, 11, 795}}, {12, 42, {12, 1682, 11, 1507, 11, 783}},
};

void CheckGolden(uint16_t bitness, const std::vector<bool>& truth_table, const std::vector<bool>& second_table,
                 const SolverTargets& expected) {
    EXPECT_EQ(tools::SolveForDepth(bitness, truth_table), expected.depth) << "bitness=" << bitness;
    EXPECT_EQ(tools::SolveForSize(bitness, truth_table), expected.size) << "bitness=" << bitness;
    EXPECT_EQ(tools::SolveForDepthGiven(bitness, truth_table, second_table), expected.given_depth)
        << "bitness=" << bitness;
    EXPECT_EQ(tools::SolveForSizeGiven(bitness, truth_table, second_table), expected.given_size)
        << "bitness=" << bitness;
    EXPECT_EQ(tools::SolveForDepthRestricted(bitness, truth_table, second_table), expected.restricted_depth)
        << "bitness=" << bitness;
    EXPECT_EQ(tools::SolveForSizeRestricted(bitness, truth_table, second_table), expected.restricted_size)
        << "bitness=" << bitness;
}

void CheckAgainstReference(uint16_t bitness, const std::vector<bool>& truth_table,
                           const std::vector<bool>& second_table) {
    EXPECT_EQ(tools::SolveForDepth(bitness, truth_table),
              Reference(Kind::kDepth, bitness, truth_table, nullptr, nullptr));
    EXPECT_EQ(tools::SolveForSize(bitness, truth_table), Reference(Kind::kSize, bitness, truth_table, nullptr, nullptr));
    EXPECT_EQ(tools::SolveForDepthGiven(bitness, truth_table, second_table),
              Reference(Kind::kDepth, bitness, truth_table, &second_table, nullptr));
    EXPECT_EQ(tools::SolveForSizeGiven(bitness, truth_table, second_table),
              Reference(Kind::kSize, bitness, truth_table, &second_table, nullptr));
    EXPECT_EQ(tools::SolveForDepthRestricted(bitness, truth_table, second_table),
              Reference(Kind::kDepth, bitness, truth_table, nullptr, &second_table));
    EXPECT_EQ(tools::SolveForSizeRestricted(bitness, truth_table, second_table),
              Reference(Kind::kSize, bitness, truth_table, nullptr, &second_table));
}

}  // namespace

TEST(SolverTest, InputPermutationPreservesBothModelTargets) {
    for (const auto& example : kSolverGoldenCases) {
        const auto g = TableFromBits(example.bitness, example.truth_bits);
        const auto f = TableFromBits(example.bitness, example.second_bits);
        std::vector<bool> permuted_g(g.size()), permuted_f(f.size());
        for (size_t input = 0; input < g.size(); ++input) {
            const size_t rotated = ((input << 1) & (g.size() - 1)) | (input >> (example.bitness - 1));
            permuted_g[input] = g[rotated];
            permuted_f[input] = f[rotated];
        }
        CheckGolden(example.bitness, permuted_g, permuted_f, example.targets);
    }
}

TEST(SolverTest, GoldenTargets) {
    for (const SolverGoldenCase& c : kSolverGoldenCases) {
        CheckGolden(c.bitness, TableFromBits(c.bitness, c.truth_bits), TableFromBits(c.bitness, c.second_bits),
                    c.targets);
    }
}

TEST(SolverTest, GoldenTargetsMatchReference) {
    for (const SolverGoldenCase& c : kSolverGoldenCases) {
        CheckAgainstReference(c.bitness, TableFromBits(c.bitness, c.truth_bits),
                              TableFromBits(c.bitness, c.second_bits));
    }
}

TEST(SolverTest, SeededGoldenTargets) {
    for (const SolverSeededGoldenCase& c : kSolverSeededGoldenCases) {
        const auto [truth_table, second_table] = SeededPair(c.bitness, c.seed);
        CheckGolden(c.bitness, truth_table, second_table, c.targets);
    }
}

TEST(SolverTest, MatchesReferenceExhaustively) {
    constexpr uint16_t kBitness = 2;
    for (uint64_t truth_bits = 0; truth_bits < 16; ++truth_bits) {
        for (uint64_t second_bits = 0; second_bits < 16; ++second_bits) {
            CheckAgainstReference(kBitness, TableFromBits(kBitness, truth_bits), TableFromBits(kBitness, second_bits));
        }
    }
}

TEST(SolverTest, MatchesReferenceOnSamples) {
    for (uint16_t bitness : {3, 4}) {
        const uint64_t pairs = bitness == 3 ? 100 : 20;
        for (uint64_t seed = 0; seed < pairs; ++seed) {
            const auto [truth_table, second_table] = SeededPair(bitness, seed);
            CheckAgainstReference(bitness, truth_table, second_table);
        }
    }
}

// A helper the tree can already compute for itself buys nothing: f = x0 leaves
// the targets where they were, while f = g collapses them to a single query.
TEST(SolverTest, RedundantAndPerfectHelpers) {
    constexpr uint16_t kBitness = 8;
    for (uint64_t seed = 0; seed < 8; ++seed) {
        const auto [truth_table, ignored] = SeededPair(kBitness, seed);

        std::vector<bool> first_bit(truth_table.size());
        for (size_t values = 0; values < truth_table.size(); ++values) first_bit[values] = values & 1;

        EXPECT_EQ(tools::SolveForDepthGiven(kBitness, truth_table, first_bit),
                  tools::SolveForDepth(kBitness, truth_table));
        EXPECT_EQ(tools::SolveForSizeGiven(kBitness, truth_table, first_bit),
                  tools::SolveForSize(kBitness, truth_table));

        EXPECT_EQ(tools::SolveForDepthGiven(kBitness, truth_table, truth_table), 1u);
        EXPECT_EQ(tools::SolveForSizeGiven(kBitness, truth_table, truth_table), 1u);
    }
}

// The whole domain is no restriction at all, while a subset of one point (or of
// none) leaves nothing to tell apart.
TEST(SolverTest, DegenerateSubsets) {
    constexpr uint16_t kBitness = 8;
    for (uint64_t seed = 0; seed < 8; ++seed) {
        const auto [truth_table, ignored] = SeededPair(kBitness, seed);

        const std::vector<bool> everything(truth_table.size(), true);
        EXPECT_EQ(tools::SolveForDepthRestricted(kBitness, truth_table, everything),
                  tools::SolveForDepth(kBitness, truth_table));
        EXPECT_EQ(tools::SolveForSizeRestricted(kBitness, truth_table, everything),
                  tools::SolveForSize(kBitness, truth_table));

        std::vector<bool> subset(truth_table.size(), false);
        EXPECT_EQ(tools::SolveForDepthRestricted(kBitness, truth_table, subset), 0u);
        EXPECT_EQ(tools::SolveForSizeRestricted(kBitness, truth_table, subset), 0u);

        subset[seed] = true;
        EXPECT_EQ(tools::SolveForDepthRestricted(kBitness, truth_table, subset), 0u);
        EXPECT_EQ(tools::SolveForSizeRestricted(kBitness, truth_table, subset), 0u);
    }
}

// Neither reading of the second table can cost the tree anything: a helper is
// one more query it may ignore, a subset is rows it no longer has to get right.
TEST(SolverTest, SecondTableNeverHurts) {
    constexpr uint16_t kBitness = 8;
    for (uint64_t seed = 0; seed < 8; ++seed) {
        const auto [truth_table, second_table] = SeededPair(kBitness, seed);

        EXPECT_LE(tools::SolveForDepthGiven(kBitness, truth_table, second_table),
                  tools::SolveForDepth(kBitness, truth_table));
        EXPECT_LE(tools::SolveForSizeGiven(kBitness, truth_table, second_table),
                  tools::SolveForSize(kBitness, truth_table));
        EXPECT_LE(tools::SolveForDepthRestricted(kBitness, truth_table, second_table),
                  tools::SolveForDepth(kBitness, truth_table));
        EXPECT_LE(tools::SolveForSizeRestricted(kBitness, truth_table, second_table),
                  tools::SolveForSize(kBitness, truth_table));
    }
}

// The paper's reduction of S_1 through f: rooting the tree at f leaves S_2 on
// f^-1(0) and f^-1(1), so it bounds S_1 from above.
TEST(SolverTest, RootingAtHelperBoundsGiven) {
    constexpr uint16_t kBitness = 8;
    for (uint64_t seed = 0; seed < 8; ++seed) {
        const auto [truth_table, second_table] = SeededPair(kBitness, seed);

        std::vector<bool> complement(second_table.size());
        for (size_t values = 0; values < second_table.size(); ++values) complement[values] = !second_table[values];

        const size_t zero_depth = tools::SolveForDepthRestricted(kBitness, truth_table, complement);
        const size_t one_depth = tools::SolveForDepthRestricted(kBitness, truth_table, second_table);
        EXPECT_LE(tools::SolveForDepthGiven(kBitness, truth_table, second_table),
                  Combine(Kind::kDepth, zero_depth, one_depth));

        const size_t zero_size = tools::SolveForSizeRestricted(kBitness, truth_table, complement);
        const size_t one_size = tools::SolveForSizeRestricted(kBitness, truth_table, second_table);
        EXPECT_LE(tools::SolveForSizeGiven(kBitness, truth_table, second_table),
                  Combine(Kind::kSize, zero_size, one_size));
    }
}
