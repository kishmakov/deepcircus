#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "generator.h"
#include "thread_pool.h"

struct TrainingShape {
    uint64_t first_iteration;
    uint64_t last_iteration;
    uint16_t bitness_from;
    uint16_t bitness_to;
    uint64_t seed;
    uint64_t train_samples;
    uint64_t validation_samples;
    uint16_t sample_batches;
    uint16_t sample_batch_size;
};

struct Task {
    uint64_t iteration;
    uint16_t bitness;
    uint64_t seed;
};

struct TaskResult {
    Task task;
    // Inputs come with either exact targets (values) or restriction matrices
    // (restrictions); worker chunks are merged back before publication.
    std::vector<gen::GeneratedValues> values;
    std::vector<gen::GeneratedRestrictions> restrictions;
};

// Generates the bitness x iteration task grid up front, then produces each
// coordinate strictly sequentially on a dedicated producer thread: the
// producer samples that coordinate's case ids once, splits them into chunks,
// and fans the chunks out across the FIFO thread pool before merging them
// back into one batch. Finished coordinates are buffered up to kPrefetchDepth
// ahead of consumption, so Take() hands out a ready result and generation of
// the next one is already underway.
class TaskQueue {
public:
    static constexpr size_t kPrefetchDepth = 4;

    TaskQueue(TrainingShape shape, size_t workers);
    ~TaskQueue();

    std::unique_ptr<TaskResult> Take();

private:
    void Produce();

    TrainingShape shape_;
    std::vector<Task> tasks_;
    ThreadPool pool_;

    std::mutex mutex_;
    std::condition_variable ready_;
    std::condition_variable space_;
    std::deque<std::unique_ptr<TaskResult>> buffer_;
    bool produced_all_ = false;
    bool stopping_ = false;
    std::thread producer_;
};
