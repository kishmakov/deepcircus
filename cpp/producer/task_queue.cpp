#include "task_queue.h"

#include "generator.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <utility>

namespace {

    constexpr uint64_t kSplitMixIncrement = 0x9e3779b97f4a7c15ull;

    uint64_t SplitMix64(uint64_t &state) {
        uint64_t value = (state += kSplitMixIncrement);
        value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
        value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
        return value ^ (value >> 31);
    }

    uint64_t TaskSeed(uint64_t seed, uint16_t bitness, uint64_t iteration) {
        uint64_t state = seed ^ (static_cast<uint64_t>(bitness) << 48) ^ iteration;
        return SplitMix64(state);
    }

    // Chooses which generator produces a tensor's values; this distinction is
    // internal to task generation and never reaches the wire protocol.
    enum class GeneratorSource {
        Tree,
        Table,
    };

    void AddData(std::vector<TaskData> &output, GeneratorSource source, uint16_t bitness, uint64_t cases,
                 uint64_t reps, uint64_t batch_size, uint64_t seed) {
        if (cases == 0) {
            return;
        }
        const bool recursive = source == GeneratorSource::Table && bitness > gen::TableSolvableBitness();
        const uint64_t chunk_cases = recursive ? std::min(cases, batch_size) : 0;
        gen::Data generated = source == GeneratorSource::Tree
                                      ? gen::TreeValueTensor(bitness, cases, reps, seed)
                                      : gen::TableValueTensor(bitness, cases, reps, chunk_cases, seed);
        output.push_back(TaskData{TensorKind::Values, std::move(generated)});
    }

    // A task's validation set depends only on its bitness, so it is generated
    // once (iteration 0) ahead of every training iteration for that bitness,
    // rather than being recomputed on each training task.
    constexpr uint64_t kValidationIteration = 0;

    // Solvable bitness trains on exact table targets alone. Above it, exact
    // tree targets are paired with recursive tables whose targets the client
    // approximates from the published restrictions.
    std::unique_ptr<TaskResult> GenerateTrainTask(const TrainingShape &shape, const Task &task) {
        auto result = std::make_unique<TaskResult>();
        result->task = task;
        const uint16_t solvable = gen::TableSolvableBitness();
        assert(gen::MinTreeBitness() <= solvable);

        if (task.bitness <= solvable) {
            AddData(result->data, GeneratorSource::Table, task.bitness, shape.train_samples, shape.points_per_sample,
                    shape.batch_size, task.seed ^ 0x1001);
        } else {
            assert(shape.train_samples % 2 == 0);
            AddData(result->data, GeneratorSource::Tree, task.bitness, shape.train_samples / 2,
                    shape.points_per_sample, shape.batch_size, task.seed ^ 0x1002);
            AddData(result->data, GeneratorSource::Table, task.bitness, shape.train_samples / 2,
                    shape.points_per_sample, shape.batch_size, task.seed ^ 0x1001);
        }
        return result;
    }

    // A validation task has the same shape as a train task: values paired with
    // exact targets, read once per bitness before training starts.
    std::unique_ptr<TaskResult> GenerateValidationTask(const TrainingShape &shape, const Task &task) {
        auto result = std::make_unique<TaskResult>();
        result->task = task;
        const uint16_t solvable = gen::TableSolvableBitness();

        if (task.bitness <= solvable) {
            AddData(result->data, GeneratorSource::Table, task.bitness, shape.validation_samples,
                    shape.points_per_sample, shape.batch_size, task.seed ^ 0x2001);
        } else {
            AddData(result->data, GeneratorSource::Tree, task.bitness, shape.validation_samples,
                    shape.points_per_sample, shape.batch_size, task.seed ^ 0x2002);
        }
        return result;
    }

} // namespace

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
    results_.resize(tasks_.size());
    // Fixed result indices decouple parallel completion from publication order.
    for (size_t index = 0; index < tasks_.size(); ++index) {
        const Task task = tasks_[index];
        pool_.Enqueue([this, index, task] {
            std::unique_ptr<TaskResult> result = task.iteration == kValidationIteration
                                                          ? GenerateValidationTask(shape_, task)
                                                          : GenerateTrainTask(shape_, task);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                assert(results_[index] == nullptr);
                results_[index] = std::move(result);
            }
            ready_.notify_all();
        });
    }
}

std::unique_ptr<TaskResult> TaskQueue::Take() {
    if (next_ == results_.size()) {
        return nullptr;
    }
    std::unique_lock<std::mutex> lock(mutex_);
    ready_.wait(lock, [this] { return results_[next_] != nullptr; });
    std::unique_ptr<TaskResult> result = std::move(results_[next_]);
    ++next_;
    return result;
}
