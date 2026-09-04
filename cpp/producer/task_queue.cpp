#include "task_queue.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>

#include "func/tree.h"
#include "generator.h"
#include "sample.h"
#include "tools/random.h"
#include "utils.h"

namespace {

constexpr uint64_t kTreeSelectionDomain = 0x747265655f73656cull;
constexpr uint64_t kTreeInputDomain = 0x747265655f696e70ull;

std::vector<bool> SampleValues(const func::TreeFunc& tree, uint16_t bitness, uint64_t seed, gen::InputShape shape) {
    tools::Random random(tools::DomainSeed(seed, kTreeInputDomain, bitness));
    const std::vector<bool> inputs = tools::SampleInputs(shape, bitness, [&random] { return random.Bool(); });
    const size_t points = static_cast<size_t>(shape.batches) * shape.batch_size;
    const size_t sample_size = 2 * bitness + 1;

    std::vector<bool> samples;
    samples.reserve(points * sample_size);
    for (size_t point_id = 0; point_id < points; ++point_id) {
        std::vector<bool> point(inputs.begin() + point_id * bitness, inputs.begin() + (point_id + 1) * bitness);
        samples.insert(samples.end(), point.begin(), point.end());
        samples.push_back(tree(point));
        for (uint16_t bit = 0; bit < bitness; ++bit) {
            point[bit] = !point[bit];
            samples.push_back(tree(point));
            point[bit] = !point[bit];
        }
    }
    assert(samples.size() == points * sample_size);
    return samples;
}

gen::GeneratedValues TreeValuesForSeeds(uint16_t bitness, const std::vector<uint64_t>& seeds, gen::InputShape shape) {
    assert(!seeds.empty());
    assert(shape.batches > 1);
    assert(std::has_single_bit(shape.batch_size));
    assert(bitness >= func::kMinBitness && bitness <= func::kMaxBitness);

    const size_t columns = static_cast<size_t>(shape.batches) * shape.batch_size * (2 * bitness + 1);
    std::vector<bool> values(seeds.size() * columns);
    std::vector<float> targets(gen::kTargetsPerCase * seeds.size());
    for (size_t case_index = 0; case_index < seeds.size(); ++case_index) {
        const func::TreeFunc tree(bitness, seeds[case_index]);
        const std::vector<bool> samples = SampleValues(tree, bitness, seeds[case_index], shape);
        std::copy(samples.begin(), samples.end(), values.begin() + case_index * columns);
        targets[gen::kTargetsPerCase * case_index] = static_cast<float>(bitness - tree.Depth());
        targets[gen::kTargetsPerCase * case_index + 1] = gen::SizeScore(bitness, tree.Size());
    }
    return gen::GeneratedValues{gen::Values(seeds.size(), columns, std::move(values)), std::move(targets)};
}

// Splits case seeds into pool.WorkerCount() contiguous parallel chunks,
// runs the supplied synchronous generator for each chunk, and returns the
// results in their original case order.
template <typename Generator>
auto GenerateParallel(ThreadPool& pool, const std::vector<uint64_t>& seeds, Generator generator) {
    using Generated = std::invoke_result_t<Generator, const std::vector<uint64_t>&>;
    assert(!seeds.empty());
    const size_t chunk_cases = std::max<size_t>(1, (seeds.size() + pool.WorkerCount() - 1) / pool.WorkerCount());
    const size_t chunk_count = (seeds.size() + chunk_cases - 1) / chunk_cases;
    std::vector<std::optional<Generated>> chunks(chunk_count);

    for (size_t chunk_index = 0; chunk_index < chunk_count; ++chunk_index) {
        const size_t first = chunk_index * chunk_cases;
        const size_t count = std::min(chunk_cases, seeds.size() - first);
        std::vector<uint64_t> chunk(seeds.begin() + first, seeds.begin() + first + count);
        pool.Enqueue(
            [&chunks, chunk_index, generator, chunk = std::move(chunk)] { chunks[chunk_index] = generator(chunk); });
    }

    pool.WaitIdle();

    std::vector<Generated> generated;
    generated.reserve(chunks.size());
    for (std::optional<Generated>& chunk : chunks) {
        assert(chunk.has_value());
        generated.push_back(std::move(*chunk));
    }
    return generated;
}

gen::GeneratedValues ConcatValues(std::vector<gen::GeneratedValues> chunks) {
    assert(!chunks.empty());
    std::vector<gen::Values> values;
    values.reserve(chunks.size());
    std::vector<float> targets;
    for (gen::GeneratedValues& chunk : chunks) {
        values.push_back(std::move(chunk.values));
        targets.insert(targets.end(), chunk.targets.begin(), chunk.targets.end());
    }
    return gen::GeneratedValues{gen::Values::Concat(std::move(values)), std::move(targets)};
}

gen::GeneratedValues GenerateTreeValues(ThreadPool& pool, uint16_t bitness, size_t cases, uint16_t batches,
                                        uint16_t batch_size, uint64_t seed) {
    assert(cases > 0);
    const std::vector<uint64_t> seeds =
        tools::SampleSeeds(cases, tools::DomainSeed(seed, kTreeSelectionDomain, bitness));
    return ConcatValues(
        GenerateParallel(pool, seeds, [bitness, batches, batch_size](const std::vector<uint64_t>& chunk) {
            return TreeValuesForSeeds(bitness, chunk, {batches, batch_size});
        }));
}

// A task's validation set depends only on its bitness, so it is generated
// once (iteration 0) ahead of every training iteration for that bitness,
// rather than being recomputed on each training task.
constexpr uint64_t kValidationIteration = 0;

std::unique_ptr<TaskResult> GenerateTrainTask(const TrainingShape& shape, const Task& task, ThreadPool& pool) {
    auto result = std::make_unique<TaskResult>();
    result->task = task;
    result->values.push_back(GenerateTreeValues(pool, task.bitness, shape.train_samples, shape.sample_batches,
                                                shape.sample_batch_size, task.seed));
    return result;
}

std::unique_ptr<TaskResult> GenerateValidationTask(const TrainingShape& shape, const Task& task, ThreadPool& pool) {
    auto result = std::make_unique<TaskResult>();
    result->task = task;
    result->values.push_back(GenerateTreeValues(pool, task.bitness, shape.validation_samples, shape.sample_batches,
                                                shape.sample_batch_size, task.seed));
    return result;
}

std::unique_ptr<TaskResult> GenerateTask(const TrainingShape& shape, const Task& task, ThreadPool& pool) {
    return task.iteration == kValidationIteration ? GenerateValidationTask(shape, task, pool)
                                                  : GenerateTrainTask(shape, task, pool);
}

}  // namespace

TaskQueue::TaskQueue(TrainingShape shape, size_t workers) : shape_(shape), pool_(workers) {
    assert(shape.bitness_from >= func::kMinBitness);
    assert(shape.bitness_to <= func::kMaxBitness);
    for (uint32_t bitness = shape.bitness_from; bitness <= shape.bitness_to; ++bitness) {
        tasks_.push_back(Task{kValidationIteration, static_cast<uint16_t>(bitness),
                              tools::TaskSeed(shape.seed, static_cast<uint16_t>(bitness), kValidationIteration)});
    }
    for (uint64_t iteration = shape.first_iteration; iteration <= shape.last_iteration; ++iteration) {
        for (uint32_t bitness = shape.bitness_from; bitness <= shape.bitness_to; ++bitness) {
            tasks_.push_back(Task{iteration, static_cast<uint16_t>(bitness),
                                  tools::TaskSeed(shape.seed, static_cast<uint16_t>(bitness), iteration)});
        }
    }
    producer_ = std::thread([this] { Produce(); });
}

TaskQueue::~TaskQueue() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    space_.notify_all();
    producer_.join();
}

void TaskQueue::Produce() {
    for (const Task& task : tasks_) {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            space_.wait(lock, [this] { return buffer_.size() < kPrefetchDepth || stopping_; });
            if (stopping_) {
                return;
            }
        }
        std::unique_ptr<TaskResult> result = GenerateTask(shape_, task, pool_);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            buffer_.push_back(std::move(result));
        }
        ready_.notify_one();
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        produced_all_ = true;
    }
    ready_.notify_all();
}

std::unique_ptr<TaskResult> TaskQueue::Take() {
    std::unique_lock<std::mutex> lock(mutex_);
    ready_.wait(lock, [this] { return !buffer_.empty() || produced_all_; });
    if (buffer_.empty()) {
        return nullptr;
    }
    std::unique_ptr<TaskResult> result = std::move(buffer_.front());
    buffer_.pop_front();
    space_.notify_one();
    return result;
}
