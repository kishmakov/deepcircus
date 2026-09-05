#include "dataset.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <thread>

#include "func/func.h"
#include "func/table.h"
#include "func/tree.h"
#include "func/tt.h"
#include "offline/read_write.h"
#include "sample.h"
#include "tools/random.h"
#include "tools/score.h"

namespace server {

namespace {

// Keeps the input walk of a case clear of the streams the preparation side
// burned on the same coordinates. The epoch is added to it, so each epoch draws
// from a domain of its own.
constexpr uint64_t kInputDomain = 0x696e707574735f30ull;
constexpr uint64_t kPermutationDomain = 0x7065726d75746530ull;
constexpr uint64_t kPrimaryDomain = 0x7265647563655f70ull;
constexpr uint64_t kHelperDomain = 0x7265647563655f66ull;

struct SampledPair {
    const func::Func& g;
    const func::Func& f;
    uint16_t bitness;
    SamplingShape shape;
};

std::string BitnessTag(uint16_t bitness) {
    return bitness < 10 ? "0" + std::to_string(bitness) : std::to_string(bitness);
}

std::unique_ptr<func::Func> MakeFunc(uint16_t bitness, const offline::Function& function) {
    switch (function.kind) {
        case offline::FunctionKind::kTable:
            return std::make_unique<func::TableFunc>(bitness, function.payload);
        case offline::FunctionKind::kTree:
            return std::make_unique<func::TreeFunc>(bitness, function.payload);
        case offline::FunctionKind::kTreeOverTable:
            return std::make_unique<func::TTFunc>(bitness, function.payload);
    }
    assert(false);
    return nullptr;
}

// The function's value at `point`, then its value at each single-bit flip.
// `point` is restored before returning.
void AppendBlock(std::vector<bool>& row, const func::Func& function, std::vector<bool>& point,
                 const std::vector<uint16_t>& permutation, bool invert = false) {
    row.push_back(function(point) != invert);
    for (uint16_t bit : permutation) {
        point[bit] = !point[bit];
        row.push_back(function(point) != invert);
        point[bit] = !point[bit];
    }
}

std::vector<bool> SamplePair(const SampledPair& pair, const std::vector<uint16_t>& permutation,
                             tools::Random& random, bool invert_f = false) {
    assert(permutation.size() == pair.bitness);
    const tools::InputShape input_shape{pair.shape.batches, pair.shape.points_in_batch};
    const std::vector<bool> inputs =
        tools::SampleInputs(input_shape, pair.bitness, [&random] { return random.Bool(); });
    const size_t points = size_t{pair.shape.batches} * pair.shape.points_in_batch;
    assert(inputs.size() == points * pair.bitness);

    std::vector<bool> row;
    row.reserve(points * PointDim(pair.bitness));
    std::vector<bool> point(pair.bitness);
    for (size_t id = 0; id < points; ++id) {
        const auto begin = inputs.begin() + id * pair.bitness;
        row.insert(row.end(), begin, begin + pair.bitness);
        // Displayed bit i feeds original input permutation[i] in both functions.
        for (uint16_t bit = 0; bit < pair.bitness; ++bit) point[permutation[bit]] = inputs[id * pair.bitness + bit];
        AppendBlock(row, pair.g, point, permutation);
        AppendBlock(row, pair.f, point, permutation, invert_f);
    }
    assert(row.size() == points * PointDim(pair.bitness));
    return row;
}

uint16_t FullBitId(uint16_t free_bit, uint16_t fixed_bit) { return free_bit < fixed_bit ? free_bit : free_bit + 1; }

void AppendRestrictedBlock(std::vector<bool>& row, const func::Func& function, std::vector<bool>& point,
                           uint16_t fixed_bit) {
    row.push_back(function(point));
    for (uint16_t free_bit = 0; free_bit + 1 < point.size(); ++free_bit) {
        const uint16_t full_bit = FullBitId(free_bit, fixed_bit);
        point[full_bit] = !point[full_bit];
        row.push_back(function(point));
        point[full_bit] = !point[full_bit];
    }
}

// Samples the pair with one input bit fixed, omitting that bit from the point layout.
std::vector<bool> SamplePrimaryReduction(const SampledPair& pair, uint16_t fixed_bit, bool fixed_value,
                                         tools::Random& random) {
    const uint16_t free_bits = pair.bitness - 1;
    const tools::InputShape input_shape{pair.shape.batches, pair.shape.points_in_batch};
    const std::vector<bool> inputs = tools::SampleInputs(input_shape, free_bits, [&random] { return random.Bool(); });
    const size_t points = size_t{pair.shape.batches} * pair.shape.points_in_batch;
    assert(inputs.size() == points * free_bits);

    std::vector<bool> row;
    row.reserve(points * PointDim(free_bits));
    std::vector<bool> point(pair.bitness);
    point[fixed_bit] = fixed_value;
    for (size_t id = 0; id < points; ++id) {
        const auto begin = inputs.begin() + id * free_bits;
        row.insert(row.end(), begin, begin + free_bits);
        for (uint16_t free_bit = 0; free_bit < free_bits; ++free_bit) {
            point[FullBitId(free_bit, fixed_bit)] = inputs[id * free_bits + free_bit];
        }
        AppendRestrictedBlock(row, pair.g, point, fixed_bit);
        AppendRestrictedBlock(row, pair.f, point, fixed_bit);
    }
    assert(row.size() == points * PointDim(free_bits));
    return row;
}

// Samples both fixed values of every input bit, ordered by bit then value.
std::vector<std::vector<bool>> SamplePrimaryRows(const SampledPair& pair, tools::Random& random) {
    std::vector<std::vector<bool>> rows;
    rows.reserve(2 * pair.bitness);
    for (uint16_t fixed_bit = 0; fixed_bit < pair.bitness; ++fixed_bit) {
        for (uint16_t fixed_value = 0; fixed_value <= 1; ++fixed_value) {
            rows.push_back(SamplePrimaryReduction(pair, fixed_bit, fixed_value != 0, random));
        }
    }
    return rows;
}

// Samples g paired with the indicators of f = 0 and f = 1, in that order.
std::vector<std::vector<bool>> SampleHelperRows(const SampledPair& pair, tools::Random& random) {
    std::vector<uint16_t> permutation(pair.bitness);
    std::iota(permutation.begin(), permutation.end(), 0);
    return {SamplePair(pair, permutation, random, true), SamplePair(pair, permutation, random, false)};
}

void PackRow(const std::vector<bool>& bits, uint8_t* row) {
    std::memset(row, 0, (bits.size() + 7) / 8);
    for (size_t bit = 0; bit < bits.size(); ++bit) {
        row[bit / 8] |= static_cast<uint8_t>(bits[bit]) << (bit % 8);
    }
}

}  // namespace

std::string FilePath(const std::string& directory, Model model, uint16_t bitness, Split split) {
    const std::string name = model == Model::kM1 ? "m1" : "m2";
    const std::string suffix = split == Split::kTrain ? ".train" : ".val";
    return directory + "/" + name + "_" + BitnessTag(bitness) + suffix;
}

Dataset::Dataset(const std::string& path, Split split, SamplingShape shape) : split_(split), shape_(shape) {
    assert(shape_.batches > 1);
    assert(shape_.points_in_batch > 1);
    assert(std::has_single_bit(shape_.points_in_batch));

    offline::Reader reader(path);
    bitness_ = reader.Bitness();
    entries_.reserve(reader.Entries());
    targets_.resize(kTargetsPerCase * reader.Entries());
    for (uint32_t index = 0; index < reader.Entries(); ++index) {
        const offline::Entry entry = reader.Read(index);
        entries_.push_back({MakeFunc(bitness_, entry.g), MakeFunc(bitness_, entry.f)});
        if (!entry.TargetKnown()) {
            unknown_.push_back(index);
            continue;
        }
        ++known_cases_;
        targets_[kTargetsPerCase * index] = tools::DepthScore(bitness_, entry.min_depth);
        targets_[kTargetsPerCase * index + 1] = tools::SizeScore(bitness_, entry.min_size);
    }
    targets_ready_ = unknown_.empty();

    // stdout is for client, so console goes to stderr.
    std::cerr << kConsolePrefix << " sourcing " << path << std::endl;
    std::cerr << kConsolePrefix << " " << Entries() << " entries, " << known_cases_ << " with targets, "
              << unknown_.size() << " without\n";
    if (targets_ready_) ReportTargetQuantiles();
}

Cases Dataset::SamplePrimaryReductions(uint32_t first, uint32_t count) const {
    return SampleReductions(first, count, Reduction::kPrimary);
}

Cases Dataset::SampleHelperReductions(uint32_t first, uint32_t count) const {
    return SampleReductions(first, count, Reduction::kHelper);
}

Cases Dataset::SampleReductions(uint32_t first, uint32_t count, Reduction reduction) const {
    assert(first <= unknown_.size());
    assert(count > 0 && count <= unknown_.size() - first);

    const bool primary = reduction == Reduction::kPrimary;
    const uint32_t rows_per_parent = primary ? 2 * bitness_ : 2;
    const uint16_t reduced_bitness = primary ? bitness_ - 1 : bitness_;
    const uint64_t rows = uint64_t{count} * rows_per_parent;
    assert(rows <= UINT32_MAX);
    const size_t points = size_t{shape_.batches} * shape_.points_in_batch;
    Cases block{static_cast<uint32_t>(rows), points * PointDim(reduced_bitness), {}, {}};
    block.values.assign(rows * block.RowBytes(), 0);

    const uint64_t domain = tools::DomainSeed(shape_.seed, primary ? kPrimaryDomain : kHelperDomain, bitness_);
    const auto sample_rows = primary ? SamplePrimaryRows : SampleHelperRows;
    const size_t workers = std::min<size_t>(std::max(std::thread::hardware_concurrency(), 1u), count);
    std::vector<std::thread> threads;
    threads.reserve(workers);
    for (size_t worker = 0; worker < workers; ++worker) {
        threads.emplace_back([this, &block, domain, first, count, rows_per_parent, sample_rows, worker, workers] {
            for (size_t position = worker; position < count; position += workers) {
                const uint32_t index = unknown_[first + position];
                const Entry& entry = entries_[index];
                tools::Random random(tools::EntrySeed(domain, 0, bitness_, index));
                const std::vector<std::vector<bool>> rows = sample_rows({*entry.g, *entry.f, bitness_, shape_}, random);
                assert(rows.size() == rows_per_parent);
                for (size_t reduction_id = 0; reduction_id < rows.size(); ++reduction_id) {
                    const size_t row_id = position * rows_per_parent + reduction_id;
                    PackRow(rows[reduction_id], block.values.data() + row_id * block.RowBytes());
                }
            }
        });
    }
    for (std::thread& thread : threads) thread.join();
    return block;
}

void Dataset::SetUnknownTargets(const std::vector<float>& targets) {
    assert(targets.size() == kTargetsPerCase * unknown_.size());
    for (size_t position = 0; position < unknown_.size(); ++position) {
        const uint32_t index = unknown_[position];
        for (size_t target = 0; target < kTargetsPerCase; ++target) {
            const float value = targets[kTargetsPerCase * position + target];
            assert(std::isfinite(value) && value >= 0.0f && value <= bitness_);
            targets_[kTargetsPerCase * index + target] = value;
        }
    }
    targets_ready_ = true;
    ReportTargetQuantiles();
}

void Dataset::ReportTargetQuantiles() const {
    assert(targets_ready_);
    assert(!entries_.empty());
    constexpr const char* names[] = {"depth_score", "size_score"};
    std::vector<float> values(entries_.size());
    for (size_t target = 0; target < kTargetsPerCase; ++target) {
        for (size_t index = 0; index < entries_.size(); ++index) {
            values[index] = targets_[kTargetsPerCase * index + target];
        }
        std::sort(values.begin(), values.end());
        std::cerr << kConsolePrefix << " " << names[target] << " quantiles:";
        for (const int percentile : {1, 25, 50, 75, 99}) {
            const double position = (values.size() - 1) * (percentile / 100.0);
            const size_t lower = static_cast<size_t>(position);
            const size_t upper = std::min(lower + 1, values.size() - 1);
            const double value = std::lerp(double{values[lower]}, double{values[upper]}, position - lower);
            std::cerr << " " << percentile << "%=" << value;
        }
        std::cerr << '\n';
    }
}

Cases Dataset::Sample(uint32_t epoch) const {
    assert(!entries_.empty());
    assert(targets_ready_);

    const size_t points = size_t{shape_.batches} * shape_.points_in_batch;
    Cases block{Entries(), points * PointDim(bitness_), {}, targets_};
    block.values.assign(entries_.size() * block.RowBytes(), 0);

    // Each worker writes separate rows and samples immutable functions.
    const size_t workers = std::min<size_t>(std::max(std::thread::hardware_concurrency(), 1u), entries_.size());
    const uint64_t domain = tools::DomainSeed(shape_.seed, kInputDomain + epoch, bitness_);
    const uint64_t permutation_domain = tools::DomainSeed(shape_.seed, kPermutationDomain + epoch, bitness_);
    std::vector<std::thread> threads;
    threads.reserve(workers);
    for (size_t worker = 0; worker < workers; ++worker) {
        threads.emplace_back([this, &block, domain, permutation_domain, worker, workers] {
            for (size_t index = worker; index < entries_.size(); index += workers) {
                const Entry& entry = entries_[index];

                std::vector<uint16_t> permutation(bitness_);
                std::iota(permutation.begin(), permutation.end(), 0);
                if (split_ == Split::kTrain) {
                    tools::Random permutation_random(tools::EntrySeed(permutation_domain, 0, bitness_, index));
                    for (size_t remaining = permutation.size(); remaining > 1; --remaining) {
                        std::swap(permutation[remaining - 1], permutation[permutation_random.Below(remaining)]);
                    }
                }
                const uint64_t seed = tools::EntrySeed(domain, static_cast<uint16_t>(split_), bitness_, index);
                tools::Random random(seed);
                PackRow(SamplePair({*entry.g, *entry.f, bitness_, shape_}, permutation, random),
                        block.values.data() + index * block.RowBytes());
            }
        });
    }
    for (std::thread& thread : threads) thread.join();
    return block;
}

}  // namespace server
