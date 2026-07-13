#pragma once

#include "generator.h"
#include "thread_pool.h"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

// Wire-level shape of a tensor. Whether values came from the tree or table
// generator is a `task_queue.cpp` implementation detail and stops there.
enum class TensorKind : uint8_t {
    Values = 1,
    Restrictions = 2,
};

struct TrainingShape {
    uint64_t first_iteration;
    uint64_t last_iteration;
    uint16_t bitness_from;
    uint16_t bitness_to;
    uint64_t seed;
    uint64_t train_samples;
    uint64_t validation_samples;
    uint64_t points_per_sample;
    uint64_t batch_size;
};

struct Task {
    uint64_t iteration;
    uint16_t bitness;
    uint64_t seed;
};

struct TaskData {
    TensorKind kind;
    gen::Data data;
};

struct TaskResult {
    Task task;
    // Worker output remains compact until this coordinate is published.
    std::vector<TaskData> data;
};

// Generates the bitness x iteration task grid and produces each coordinate
// concurrently on a FIFO thread pool, publishing results in iteration-major,
// bitness-major order regardless of completion order.
class TaskQueue {
public:
    TaskQueue(TrainingShape shape, size_t workers);

    std::unique_ptr<TaskResult> Take();

private:
    TrainingShape shape_;
    std::vector<Task> tasks_;
    std::vector<std::unique_ptr<TaskResult>> results_;
    size_t next_ = 0;
    std::mutex mutex_;
    std::condition_variable ready_;
    ThreadPool pool_;
};
