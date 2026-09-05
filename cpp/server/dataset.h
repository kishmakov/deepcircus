#pragma once

// Turns one offline file of `docs/data_m1.md` or `docs/data_m2.md` into the
// point clouds the model is trained on. An entry is a pair of functions plus a
// target; a case is that pair sampled at `batches * points_in_batch` inputs, so
// the file's entries and the served cases stand one to one.
//
// Both models are served the same way. What their second function means is the
// file's business, not this one's: `M_1` pairs `g` with `f`, `M_2` with the
// indicator of the subset `g` is scored on, and a point carries whichever of
// them the entry holds.
//
// Every epoch samples every case again, at inputs of its own: the epoch id
// enters the seed, so epoch 3 draws different points than epoch 2 for the same
// pair, and asking for epoch 3 twice draws the same ones.
//
// Entries whose target is the unknown marker are sampled too, after the Python
// client reconstructs their targets from the reductions exposed below.

#include <stddef.h>
#include <stdint.h>

#include <string>
#include <vector>

namespace serving {

enum class Model : uint8_t {
    kM1 = 1,
    kM2 = 2,
};

enum class Split : uint8_t {
    kValidation = 0,
    kTrain = 1,
};

// How each entry is expanded into points. `seed` and the epoch decide which
// inputs a case is sampled at, and nothing else does.
struct SamplingShape {
    uint16_t batches;
    uint16_t points_in_batch;
    uint64_t seed;
};

// One point: the input bits, then `g`'s value there followed by its value at
// each single-bit flip, then the same block for the entry's second function --
// `f` for `M_1`, the indicator of `X` for `M_2`, sampled and packed alike.
constexpr uint16_t PointDim(uint16_t bitness) { return 3 * bitness + 2; }

// Two scores per case, matching `tools/score.h`.
inline constexpr size_t kTargetsPerCase = 2;

std::string FilePath(const std::string& directory, Model model, uint16_t bitness, Split split);

// Every solved case of the file, sampled for one epoch: rows of packed point
// bits, little-endian within each byte and padded to a whole number of bytes
// per row, plus the two scores of each case.
struct Cases {
    uint32_t cases;
    uint64_t columns;  // bits per row: points * PointDim(bitness)
    std::vector<uint8_t> values;
    std::vector<float> targets;

    uint64_t RowBytes() const { return (columns + 7) / 8; }
};

// One offline file, read once for what is in it and sampled once per epoch.
class Dataset {
public:
    Dataset(std::string path, Split split, SamplingShape shape);

    uint16_t Bitness() const { return bitness_; }
    uint32_t Entries() const { return entries_; }
    uint32_t KnownCases() const { return known_cases_; }
    uint32_t UnknownCases() const { return static_cast<uint32_t>(unknown_.size()); }

    // Rows are ordered parent, fixed bit, fixed value. Each is the entry with
    // one primary input fixed, represented at bitness - 1 for M_1 or M_2.
    Cases SamplePrimaryReductions(uint32_t first, uint32_t count) const;

    // Two rows per parent, ordered by f's fixed value. They represent
    // M_2[g | f^-1(0)] and M_2[g | f^-1(1)] at the original bitness.
    Cases SampleHelperReductions(uint32_t first, uint32_t count) const;

    // Scores reconstructed by Python, two per unknown entry in UnknownCases()
    // order. Once installed, Sample() serves every file entry.
    void SetUnknownTargets(const std::vector<float>& targets);

    // Every entry at `epoch`'s inputs. Sampled across threads, which changes
    // nothing about the result: a case's inputs follow from its own index.
    Cases Sample(uint32_t epoch) const;

private:
    enum class Reduction {
        kPrimary,
        kHelper,
    };

    Cases SampleReductions(uint32_t first, uint32_t count, Reduction reduction) const;

    std::string path_;
    Split split_;
    SamplingShape shape_;
    uint16_t bitness_ = 0;
    uint32_t entries_ = 0;
    uint32_t known_cases_ = 0;
    std::vector<uint32_t> unknown_;
    std::vector<float> targets_;
    bool targets_ready_ = false;
};

}  // namespace serving
