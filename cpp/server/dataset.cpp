#include "dataset.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstring>
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

namespace serving {

namespace {

// Keeps the input walk of a case clear of the streams the preparation side
// burned on the same coordinates. The epoch is added to it, so each epoch draws
// from a domain of its own.
constexpr uint64_t kInputDomain = 0x696e707574735f30ull;

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
void AppendBlock(std::vector<bool>& row, const func::Func& function, std::vector<bool>& point) {
    row.push_back(function(point));
    for (size_t bit = 0; bit < point.size(); ++bit) {
        point[bit] = !point[bit];
        row.push_back(function(point));
        point[bit] = !point[bit];
    }
}

std::vector<bool> SampleCase(const offline::Entry& entry, uint16_t bitness, SamplingShape shape, uint64_t seed) {
    const std::unique_ptr<func::Func> g = MakeFunc(bitness, entry.g);
    const std::unique_ptr<func::Func> f = MakeFunc(bitness, entry.f);

    tools::Random random(seed);
    const tools::InputShape input_shape{shape.batches, shape.points_in_batch};
    const std::vector<bool> inputs = tools::SampleInputs(input_shape, bitness, [&random] { return random.Bool(); });
    const size_t points = size_t{shape.batches} * shape.points_in_batch;
    assert(inputs.size() == points * bitness);

    std::vector<bool> row;
    row.reserve(points * PointDim(bitness));
    std::vector<bool> point(bitness);
    for (size_t id = 0; id < points; ++id) {
        std::copy(inputs.begin() + id * bitness, inputs.begin() + (id + 1) * bitness, point.begin());
        row.insert(row.end(), point.begin(), point.end());
        AppendBlock(row, *g, point);
        AppendBlock(row, *f, point);
    }
    assert(row.size() == points * PointDim(bitness));
    return row;
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
    assert(std::has_single_bit(shape_.points_in_batch));

    offline::Reader reader(path_);
    bitness_ = reader.Bitness();
    entries_ = reader.Entries();
    for (uint32_t index = 0; index < entries_; ++index) {
        if (reader.Read(index).TargetKnown()) solved_.push_back(index);
    }
}

Cases Dataset::Sample(uint32_t epoch) const {
    assert(!solved_.empty());

    const size_t points = size_t{shape_.batches} * shape_.points_in_batch;
    Cases block{CaseCount(), points * PointDim(bitness_), {}, {}};
    block.values.assign(solved_.size() * block.RowBytes(), 0);
    block.targets.resize(kTargetsPerCase * solved_.size());

    // Each case is written to a row of its own, so the workers share the two
    // buffers without touching each other's bytes. A reader is a file position,
    // though, so each one opens the file for itself.
    const size_t workers = std::min<size_t>(std::max(std::thread::hardware_concurrency(), 1u), solved_.size());
    const uint64_t domain = tools::DomainSeed(shape_.seed, kInputDomain + epoch, bitness_);
    std::vector<std::thread> threads;
    threads.reserve(workers);
    for (size_t worker = 0; worker < workers; ++worker) {
        threads.emplace_back([this, &block, domain, worker, workers] {
            offline::Reader reader(path_);
            for (size_t position = worker; position < solved_.size(); position += workers) {
                const uint32_t index = solved_[position];
                const offline::Entry entry = reader.Read(index);
                assert(entry.TargetKnown());

                const uint64_t seed = tools::EntrySeed(domain, static_cast<uint16_t>(split_), bitness_, index);
                PackRow(SampleCase(entry, bitness_, shape_, seed), block.values.data() + position * block.RowBytes());
                block.targets[kTargetsPerCase * position] = tools::DepthScore(bitness_, entry.min_depth);
                block.targets[kTargetsPerCase * position + 1] = tools::SizeScore(bitness_, entry.min_size);
            }
        });
    }
    for (std::thread& thread : threads) thread.join();
    return block;
}

}  // namespace serving
