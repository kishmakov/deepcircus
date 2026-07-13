#include "thread_pool.h"

#include <cassert>
#include <utility>

ThreadPool::ThreadPool(size_t workers) {
    assert(workers > 0);
    threads_.reserve(workers);
    for (size_t worker = 0; worker < workers; ++worker) {
        threads_.emplace_back([this] { Run(); });
    }
}

ThreadPool::~ThreadPool() {
    WaitIdle();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    work_.notify_all();
    for (std::thread &thread: threads_) {
        thread.join();
    }
}

void ThreadPool::Enqueue(std::function<void()> task) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        assert(!stopping_);
        tasks_.push_back(std::move(task));
    }
    work_.notify_one();
}

void ThreadPool::WaitIdle() {
    std::unique_lock<std::mutex> lock(mutex_);
    idle_.wait(lock, [this] { return tasks_.empty() && active_ == 0; });
}

void ThreadPool::Run() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            work_.wait(lock, [this] { return stopping_ || !tasks_.empty(); });
            if (stopping_ && tasks_.empty()) {
                return;
            }
            task = std::move(tasks_.front());
            tasks_.pop_front();
            ++active_;
        }

        task();

        {
            std::lock_guard<std::mutex> lock(mutex_);
            assert(active_ > 0);
            --active_;
            if (tasks_.empty() && active_ == 0) {
                idle_.notify_all();
            }
        }
    }
}
