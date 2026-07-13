#pragma once

#include "generator.h"

#include <cstdint>
#include <functional>
#include <memory>
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

// Binds a loopback listener on the port, prints the bound port,
// and returns the accepted client socket.
int AcceptClient(uint16_t port);

// Reads the Initialize command and its training-shape payload; the
// acknowledgement is sent by ServeTasks once the task source is ready.
TrainingShape ReadTrainingShape(int client);

// Acknowledges initialization, then serves Next/Release commands, publishing
// one coordinate at a time through POSIX shared memory. Returns when the
// client hangs up.
void ServeTasks(int client, const std::function<std::unique_ptr<TaskResult>()> &take);
