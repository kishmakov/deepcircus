#pragma once

#include "daemon.h"
#include "thread_pool.h"

#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

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
