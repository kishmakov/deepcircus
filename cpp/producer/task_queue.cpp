#include "task_queue.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>

#include "generator.h"
#include "utils.h"

namespace {

// Splits case_ids into pool.WorkerCount() contiguous parallel chunks,
// runs the supplied synchronous generator for each chunk, and returns the
// results in their original case order.
template <typename Generator>
auto GenerateParallel(ThreadPool& pool, const std::vector<size_t>& case_ids, Generator generator) {
    using Generated = std::invoke_result_t<Generator, const std::vector<size_t>&>;
    assert(!case_ids.empty());
    const size_t chunk_cases = std::max<size_t>(1, (case_ids.size() + pool.WorkerCount() - 1) / pool.WorkerCount());
    const size_t chunk_count = (case_ids.size() + chunk_cases - 1) / chunk_cases;
    std::vector<std::optional<Generated>> chunks(chunk_count);

    for (size_t chunk_index = 0; chunk_index < chunk_count; ++chunk_index) {
        const size_t first = chunk_index * chunk_cases;
        const size_t count = std::min(chunk_cases, case_ids.size() - first);
        std::vector<size_t> ids(case_ids.begin() + first, case_ids.begin() + first + count);
        pool.Enqueue([&chunks, chunk_index, generator, ids = std::move(ids)] {
            chunks[chunk_index] = generator(ids);
        });
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

gen::GeneratedValues GenerateTreeValues(ThreadPool& pool, uint16_t bitness, size_t cases, size_t reps,
                                        uint64_t seed) {
    assert(cases > 0);
    const std::vector<size_t> case_ids = gen::TreeSampleCaseIds(bitness, cases, seed);
    return ConcatValues(GenerateParallel(pool, case_ids, [bitness, reps, seed](const std::vector<size_t>& ids) {
        return gen::TreeValuesForCases(bitness, ids, reps, seed);
    }));
}

gen::GeneratedValues GenerateTableValues(ThreadPool& pool, uint16_t bitness, size_t cases, size_t reps,
                                         uint64_t seed) {
    assert(cases > 0);
    assert(bitness <= gen::TableSolvableBitness());
    const std::vector<size_t> case_ids = gen::TableSampleCaseIds(bitness, cases, seed);
    return ConcatValues(GenerateParallel(pool, case_ids, [bitness, reps, seed](const std::vector<size_t>& ids) {
        return gen::TableValuesForCases(bitness, ids, reps, seed);
    }));
}

gen::GeneratedRestrictions GenerateTableRestrictions(ThreadPool& pool, uint16_t bitness, size_t cases,
                                                     size_t reps, uint64_t seed) {
    assert(cases > 0);
    assert(bitness > gen::TableSolvableBitness());
    const std::vector<size_t> case_ids = gen::TableSampleCaseIds(bitness, cases, seed);
    return ConcatRestrictions(GenerateParallel(pool, case_ids, [bitness, reps, seed](const std::vector<size_t>& ids) {
        return gen::TableRestrictionsForCases(bitness, ids, reps, seed);
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
    const uint16_t solvable = gen::TableSolvableBitness();
    assert(gen::MinTreeBitness() <= solvable);

    if (task.bitness <= solvable) {
        result->values.push_back(GenerateTableValues(pool, task.bitness, shape.train_samples,
                                                     shape.points_per_sample, task.seed ^ 0x1001));
    } else {
        assert(shape.train_samples % 2 == 0);
        result->values.push_back(GenerateTreeValues(pool, task.bitness, shape.train_samples / 2,
                                                    shape.points_per_sample, task.seed));
        result->restrictions.push_back(GenerateTableRestrictions(pool, task.bitness, shape.train_samples / 2,
                                                                 shape.points_per_sample, task.seed));
    }
    return result;
}

// A validation task has the same shape as a train task: values paired with
// exact targets, read once per bitness before training starts.
std::unique_ptr<TaskResult> GenerateValidationTask(const TrainingShape& shape, const Task& task, ThreadPool& pool) {
    auto result = std::make_unique<TaskResult>();
    result->task = task;
    const uint16_t solvable = gen::TableSolvableBitness();

    if (task.bitness <= solvable) {
        result->values.push_back(GenerateTableValues(pool, task.bitness, shape.validation_samples,
                                                     shape.points_per_sample, task.seed ^ 0x2001));
    } else {
        result->values.push_back(GenerateTreeValues(pool, task.bitness, shape.validation_samples,
                                                    shape.points_per_sample, task.seed ^ 0x2002));
    }
    return result;
}

}  // namespace

TaskQueue::TaskQueue(TrainingShape shape, size_t workers) : shape_(shape), pool_(workers) {
    for (uint32_t bitness = shape.bitness_from; bitness <= shape.bitness_to; ++bitness) {
        tasks_.push_back(Task{kValidationIteration, static_cast<uint16_t>(bitness),
                              TaskSeed(shape.seed, static_cast<uint16_t>(bitness), kValidationIteration)});
    }
    for (uint64_t iteration = shape.first_iteration; iteration <= shape.last_iteration; ++iteration) {
        for (uint32_t bitness = shape.bitness_from; bitness <= shape.bitness_to; ++bitness) {
            tasks_.push_back(Task{iteration, static_cast<uint16_t>(bitness),
                                  TaskSeed(shape.seed, static_cast<uint16_t>(bitness), iteration)});
        }
    }
}

std::unique_ptr<TaskResult> TaskQueue::Take() {
    if (next_ == tasks_.size()) {
        return nullptr;
    }
    const Task task = tasks_[next_++];
    return task.iteration == kValidationIteration ? GenerateValidationTask(shape_, task, pool_)
                                                  : GenerateTrainTask(shape_, task, pool_);
}
