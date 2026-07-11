#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

// Minimal FIFO pool; destruction waits for queued and active work to finish.
class ThreadPool {
public:
    explicit ThreadPool(size_t workers);
    ~ThreadPool();

    ThreadPool(const ThreadPool &) = delete;
    ThreadPool &operator=(const ThreadPool &) = delete;

    void Enqueue(std::function<void()> task);
    void WaitIdle();

private:
    void Run();

    std::mutex mutex_;
    std::condition_variable work_;
    std::condition_variable idle_;
    std::deque<std::function<void()>> tasks_;
    std::vector<std::thread> threads_;
    size_t active_ = 0;
    bool stopping_ = false;
};
