#include "dataset.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <thread>
#include <utility>

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
void AppendBlock(std::vector<bool>& row, const func::Func& function, std::vector<bool>& point, bool invert = false) {
    row.push_back(function(point) != invert);
    for (size_t bit = 0; bit < point.size(); ++bit) {
        point[bit] = !point[bit];
        row.push_back(function(point) != invert);
        point[bit] = !point[bit];
    }
}

std::vector<bool> SamplePair(const SampledPair& pair, tools::Random& random, bool invert_f = false) {
    const tools::InputShape input_shape{pair.shape.batches, pair.shape.points_in_batch};
    const std::vector<bool> inputs =
        tools::SampleInputs(input_shape, pair.bitness, [&random] { return random.Bool(); });
    const size_t points = size_t{pair.shape.batches} * pair.shape.points_in_batch;
    assert(inputs.size() == points * pair.bitness);

    std::vector<bool> row;
    row.reserve(points * PointDim(pair.bitness));
    std::vector<bool> point(pair.bitness);
    for (size_t id = 0; id < points; ++id) {
        std::copy(inputs.begin() + id * pair.bitness, inputs.begin() + (id + 1) * pair.bitness, point.begin());
        row.insert(row.end(), point.begin(), point.end());
        AppendBlock(row, pair.g, point);
        AppendBlock(row, pair.f, point, invert_f);
    }
    assert(row.size() == points * PointDim(pair.bitness));
    return row;
}

std::vector<bool> SampleCase(const offline::Entry& entry, uint16_t bitness, SamplingShape shape, uint64_t seed) {
    const std::unique_ptr<func::Func> g = MakeFunc(bitness, entry.g);
    const std::unique_ptr<func::Func> f = MakeFunc(bitness, entry.f);
    tools::Random random(seed);
    return SamplePair({*g, *f, bitness, shape}, random);
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

std::vector<std::vector<bool>> SamplePrimaryRows(const offline::Entry& entry, uint16_t bitness, SamplingShape shape,
                                                 tools::Random& random) {
    const std::unique_ptr<func::Func> g = MakeFunc(bitness, entry.g);
    const std::unique_ptr<func::Func> f = MakeFunc(bitness, entry.f);
    const SampledPair pair{*g, *f, bitness, shape};
    std::vector<std::vector<bool>> rows;
    rows.reserve(2 * bitness);
    for (uint16_t fixed_bit = 0; fixed_bit < bitness; ++fixed_bit) {
        for (uint16_t fixed_value = 0; fixed_value <= 1; ++fixed_value) {
            rows.push_back(SamplePrimaryReduction(pair, fixed_bit, fixed_value != 0, random));
        }
    }
    return rows;
}

std::vector<std::vector<bool>> SampleHelperRows(const offline::Entry& entry, uint16_t bitness, SamplingShape shape,
                                                tools::Random& random) {
    const std::unique_ptr<func::Func> g = MakeFunc(bitness, entry.g);
    const std::unique_ptr<func::Func> f = MakeFunc(bitness, entry.f);
    const SampledPair pair{*g, *f, bitness, shape};
    return {SamplePair(pair, random, true), SamplePair(pair, random, false)};
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

Dataset::Dataset(std::string path, Split split, SamplingShape shape)
    : path_(std::move(path)), split_(split), shape_(shape) {
    assert(shape_.batches > 1);
    assert(shape_.points_in_batch > 1);
    assert(std::has_single_bit(shape_.points_in_batch));

    offline::Reader reader(path_);
    bitness_ = reader.Bitness();
    entries_ = reader.Entries();
    targets_.resize(kTargetsPerCase * entries_);
    for (uint32_t index = 0; index < entries_; ++index) {
        const offline::Entry entry = reader.Read(index);
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
    std::cerr << kConsolePrefix << " sourcing " << path_ << std::endl;
    std::cerr << kConsolePrefix << " " << entries_ << " entries, " << known_cases_ << " with targets, "
              << unknown_.size() << " without\n";
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
            offline::Reader reader(path_);
            for (size_t position = worker; position < count; position += workers) {
                const uint32_t index = unknown_[first + position];
                const offline::Entry entry = reader.Read(index);
                assert(!entry.TargetKnown());
                tools::Random random(tools::EntrySeed(domain, 0, bitness_, index));
                const std::vector<std::vector<bool>> rows = sample_rows(entry, bitness_, shape_, random);
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
}

Cases Dataset::Sample(uint32_t epoch) const {
    assert(entries_ > 0);
    assert(targets_ready_);

    const size_t points = size_t{shape_.batches} * shape_.points_in_batch;
    Cases block{entries_, points * PointDim(bitness_), {}, targets_};
    block.values.assign(size_t{entries_} * block.RowBytes(), 0);

    // Each case is written to a row of its own, so the workers share the two
    // buffers without touching each other's bytes. A reader is a file position,
    // though, so each one opens the file for itself.
    const size_t workers = std::min<size_t>(std::max(std::thread::hardware_concurrency(), 1u), entries_);
    const uint64_t domain = tools::DomainSeed(shape_.seed, kInputDomain + epoch, bitness_);
    std::vector<std::thread> threads;
    threads.reserve(workers);
    for (size_t worker = 0; worker < workers; ++worker) {
        threads.emplace_back([this, &block, domain, worker, workers] {
            offline::Reader reader(path_);
            for (size_t index = worker; index < entries_; index += workers) {
                const offline::Entry entry = reader.Read(index);

                const uint64_t seed = tools::EntrySeed(domain, static_cast<uint16_t>(split_), bitness_, index);
                PackRow(SampleCase(entry, bitness_, shape_, seed), block.values.data() + index * block.RowBytes());
            }
        });
    }
    for (std::thread& thread : threads) thread.join();
    return block;
}

}  // namespace server
