#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
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
    uint64_t points_per_sample;
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
// coordinate strictly sequentially and on demand: Take() samples that
// coordinate's case ids once, splits them into chunks, and fans the chunks
// out across the FIFO thread pool before merging them back into one batch.
class TaskQueue {
public:
    TaskQueue(TrainingShape shape, size_t workers);

    std::unique_ptr<TaskResult> Take();

private:
    TrainingShape shape_;
    std::vector<Task> tasks_;
    size_t next_ = 0;
    ThreadPool pool_;
};
