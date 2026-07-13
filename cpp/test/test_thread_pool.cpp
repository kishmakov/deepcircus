#include "thread_pool.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <vector>

TEST(ThreadPoolTest, RunsAllEnqueuedTasks) {
    ThreadPool pool(4);
    std::atomic<int> counter{0};
    constexpr int kTasks = 200;
    for (int i = 0; i < kTasks; ++i) {
        pool.Enqueue([&counter] { counter.fetch_add(1, std::memory_order_relaxed); });
    }
    pool.WaitIdle();
    EXPECT_EQ(counter.load(), kTasks);
}

TEST(ThreadPoolTest, SingleWorkerPreservesFifoOrder) {
    ThreadPool pool(1);
    std::mutex mutex;
    std::vector<int> order;
    constexpr int kTasks = 50;
    for (int i = 0; i < kTasks; ++i) {
        pool.Enqueue([&mutex, &order, i] {
            std::lock_guard<std::mutex> lock(mutex);
            order.push_back(i);
        });
    }
    pool.WaitIdle();

    ASSERT_EQ(order.size(), static_cast<size_t>(kTasks));
    for (int i = 0; i < kTasks; ++i) {
        EXPECT_EQ(order[i], i);
    }
}

TEST(ThreadPoolTest, WaitIdleCanBeCalledRepeatedly) {
    ThreadPool pool(3);
    std::atomic<int> counter{0};
    pool.Enqueue([&counter] { counter.fetch_add(1, std::memory_order_relaxed); });
    pool.WaitIdle();
    pool.WaitIdle();
    EXPECT_EQ(counter.load(), 1);
}

TEST(ThreadPoolTest, DestructorWaitsForOutstandingWork) {
    std::atomic<bool> finished{false};
    {
        ThreadPool pool(2);
        pool.Enqueue([&finished] {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            finished.store(true, std::memory_order_relaxed);
        });
    }
    EXPECT_TRUE(finished.load());
}
