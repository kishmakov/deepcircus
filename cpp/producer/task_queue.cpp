#include "task_queue.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>

#include "func/table.h"
#include "generator.h"
#include "tree.h"
#include "utils.h"

namespace {

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

gen::GeneratedRestrictions ConcatRestrictions(std::vector<gen::GeneratedRestrictions> chunks) {
    assert(!chunks.empty());
    std::vector<gen::Values> values;
    values.reserve(chunks.size());
    std::vector<gen::Restrictions> restrictions;
    restrictions.reserve(chunks.size());
    for (gen::GeneratedRestrictions& chunk : chunks) {
        values.push_back(std::move(chunk.values));
        restrictions.push_back(std::move(chunk.restrictions));
    }
    return gen::GeneratedRestrictions{gen::Values::Concat(std::move(values)),
                                      gen::Restrictions::Concat(std::move(restrictions))};
}

gen::GeneratedValues GenerateTreeValues(ThreadPool& pool, uint16_t bitness, size_t cases, uint16_t batches,
                                        uint16_t batch_size, uint64_t seed) {
    assert(cases > 0);
    const std::vector<uint64_t> seeds = gen::TreeSampleSeeds(bitness, cases, seed);
    return ConcatValues(
        GenerateParallel(pool, seeds, [bitness, batches, batch_size](const std::vector<uint64_t>& chunk) {
            return gen::TreeValuesForSeeds(bitness, chunk, {batches, batch_size});
        }));
}

gen::GeneratedValues GenerateTableValues(ThreadPool& pool, uint16_t bitness, size_t cases, uint16_t batches,
                                         uint16_t batch_size, uint64_t seed) {
    assert(cases > 0);
    assert(bitness <= func::kSolvableTableBitness);
    const std::vector<uint64_t> seeds = func::TableSampleSeeds(bitness, cases, seed);
    return ConcatValues(
        GenerateParallel(pool, seeds, [bitness, batches, batch_size](const std::vector<uint64_t>& chunk) {
            return func::TableValuesForSeeds(bitness, chunk, {batches, batch_size});
        }));
}

gen::GeneratedRestrictions GenerateTableRestrictions(ThreadPool& pool, uint16_t bitness, size_t cases, uint16_t batches,
                                                     uint16_t batch_size, uint64_t seed) {
    assert(cases > 0);
    assert(bitness > func::kSolvableTableBitness);
    const std::vector<uint64_t> seeds = func::TableSampleSeeds(bitness, cases, seed);
    return ConcatRestrictions(
        GenerateParallel(pool, seeds, [bitness, batches, batch_size](const std::vector<uint64_t>& chunk) {
            return func::TableRestrictionsForSeeds(bitness, chunk, {batches, batch_size});
        }));
}

// A task's validation set depends only on its bitness, so it is generated
// once (iteration 0) ahead of every training iteration for that bitness,
// rather than being recomputed on each training task.
constexpr uint64_t kValidationIteration = 0;

// Solvable bitness trains on exact table targets alone. Above it, exact
// tree targets are paired with recursive tables whose targets the client
// approximates from the published restrictions.
std::unique_ptr<TaskResult> GenerateTrainTask(const TrainingShape& shape, const Task& task, ThreadPool& pool) {
    auto result = std::make_unique<TaskResult>();
    result->task = task;
    assert(gen::kMinTreeBitness <= func::kSolvableTableBitness);

    if (task.bitness <= func::kSolvableTableBitness) {
        result->values.push_back(GenerateTableValues(pool, task.bitness, shape.train_samples, shape.sample_batches,
                                                     shape.sample_batch_size, task.seed));
    } else {
        assert(shape.train_samples % 2 == 0);
        result->values.push_back(GenerateTreeValues(pool, task.bitness, shape.train_samples / 2, shape.sample_batches,
                                                    shape.sample_batch_size, task.seed));
        result->restrictions.push_back(GenerateTableRestrictions(
            pool, task.bitness, shape.train_samples / 2, shape.sample_batches, shape.sample_batch_size, task.seed));
    }
    return result;
}

// A validation task has the same shape as a train task: values paired with
// exact targets, read once per bitness before training starts.
std::unique_ptr<TaskResult> GenerateValidationTask(const TrainingShape& shape, const Task& task, ThreadPool& pool) {
    auto result = std::make_unique<TaskResult>();
    result->task = task;
    const uint16_t solvable = func::TableSolvableBitness();

    if (task.bitness <= solvable) {
        result->values.push_back(GenerateTableValues(pool, task.bitness, shape.validation_samples, shape.sample_batches,
                                                     shape.sample_batch_size, task.seed));
    } else {
        result->values.push_back(GenerateTreeValues(pool, task.bitness, shape.validation_samples, shape.sample_batches,
                                                    shape.sample_batch_size, task.seed ^ 0x2002));
    }
    return result;
}

std::unique_ptr<TaskResult> GenerateTask(const TrainingShape& shape, const Task& task, ThreadPool& pool) {
    return task.iteration == kValidationIteration ? GenerateValidationTask(shape, task, pool)
                                                  : GenerateTrainTask(shape, task, pool);
}

}  // namespace

TaskQueue::TaskQueue(TrainingShape shape, size_t workers) : shape_(shape), pool_(workers) {
    for (uint32_t bitness = shape.bitness_from; bitness <= shape.bitness_to; ++bitness) {
        tasks_.push_back(Task{kValidationIteration, static_cast<uint16_t>(bitness),
                              gen::TaskSeed(shape.seed, static_cast<uint16_t>(bitness), kValidationIteration)});
    }
    for (uint64_t iteration = shape.first_iteration; iteration <= shape.last_iteration; ++iteration) {
        for (uint32_t bitness = shape.bitness_from; bitness <= shape.bitness_to; ++bitness) {
            tasks_.push_back(Task{iteration, static_cast<uint16_t>(bitness),
                                  gen::TaskSeed(shape.seed, static_cast<uint16_t>(bitness), iteration)});
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
