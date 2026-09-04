#include <gtest/gtest.h>

#include <set>

#include "func/func.h"
#include "task_queue.h"

namespace {

TrainingShape MakeShape() {
    TrainingShape shape{};
    shape.first_iteration = 1;
    shape.last_iteration = 2;
    shape.bitness_from = func::kMinBitness;
    shape.bitness_to = func::kMinBitness;
    shape.seed = 42;
    shape.train_samples = 4;
    shape.validation_samples = 2;
    shape.sample_batches = 2;
    shape.sample_batch_size = 4;
    return shape;
}

}  // namespace

TEST(TaskQueueTest, ValidationTask) {
    TaskQueue queue(MakeShape(), 2);

    std::unique_ptr<TaskResult> validation = queue.Take();
    ASSERT_NE(validation, nullptr);
    EXPECT_EQ(validation->task.iteration, 0u);
    EXPECT_EQ(validation->task.bitness, func::kMinBitness);
    ASSERT_EQ(validation->values.size(), 1u);
    EXPECT_TRUE(validation->restrictions.empty());
    EXPECT_EQ(validation->values[0].values.Rows(), 2u);
    EXPECT_EQ(validation->values[0].values.Columns(), 2u * 4u * (2 * func::kMinBitness + 1));
    EXPECT_EQ(validation->values[0].targets.size(), gen::kTargetsPerCase * 2u);
}

TEST(TaskQueueTest, TrainingTasks) {
    TaskQueue queue(MakeShape(), 2);
    ASSERT_NE(queue.Take(), nullptr);  // validation task

    for (uint64_t iteration = 1; iteration <= 2; ++iteration) {
        std::unique_ptr<TaskResult> result = queue.Take();
        ASSERT_NE(result, nullptr);
        EXPECT_EQ(result->task.iteration, iteration);
        EXPECT_EQ(result->task.bitness, func::kMinBitness);
        ASSERT_EQ(result->values.size(), 1u);
        EXPECT_TRUE(result->restrictions.empty());
        EXPECT_EQ(result->values[0].values.Rows(), 4u);
        EXPECT_EQ(result->values[0].targets.size(), gen::kTargetsPerCase * 4u);
    }
}

TEST(TaskQueueTest, EndsWithNull) {
    TaskQueue queue(MakeShape(), 2);
    for (int i = 0; i < 3; ++i) {
        ASSERT_NE(queue.Take(), nullptr);
    }
    EXPECT_EQ(queue.Take(), nullptr);
}

TEST(TaskQueueTest, SameSeeds) {
    TaskQueue first(MakeShape(), 2);
    TaskQueue second(MakeShape(), 3);

    for (int i = 0; i < 3; ++i) {
        std::unique_ptr<TaskResult> a = first.Take();
        std::unique_ptr<TaskResult> b = second.Take();
        ASSERT_NE(a, nullptr);
        ASSERT_NE(b, nullptr);
        EXPECT_EQ(a->task.seed, b->task.seed);
    }
}

TEST(TaskQueueTest, SeedsNotRepeated) {
    TaskQueue queue(MakeShape(), 2);
    std::set<uint64_t> seeds;
    for (int i = 0; i < 3; ++i) {
        std::unique_ptr<TaskResult> result = queue.Take();
        ASSERT_NE(result, nullptr);
        seeds.insert(result->task.seed);
    }
    EXPECT_EQ(seeds.size(), 3u);
}

// Different worker counts change how a coordinate's case seeds are chunked
// across the pool internally; the merged batch must come out byte-identical
// regardless, since chunking is an execution detail, not a sampling input.
TEST(TaskQueueTest, WorkersGenerateSameTrees) {
    TaskQueue single(MakeShape(), 1);
    TaskQueue many(MakeShape(), 5);

    std::unique_ptr<TaskResult> from_single = single.Take();
    std::unique_ptr<TaskResult> from_many = many.Take();
    ASSERT_NE(from_single, nullptr);
    ASSERT_NE(from_many, nullptr);
    ASSERT_EQ(from_single->values.size(), 1u);
    ASSERT_EQ(from_many->values.size(), 1u);
    EXPECT_TRUE(from_single->restrictions.empty());
    EXPECT_TRUE(from_many->restrictions.empty());

    const gen::GeneratedValues& single_data = from_single->values[0];
    const gen::GeneratedValues& many_data = from_many->values[0];
    ASSERT_EQ(single_data.values.ValueCount(), many_data.values.ValueCount());
    ASSERT_EQ(single_data.targets.size(), many_data.targets.size());

    std::vector<uint8_t> single_values(single_data.values.ByteCount());
    std::vector<uint8_t> many_values(many_data.values.ByteCount());
    single_data.values.WritePacked(single_values.data());
    many_data.values.WritePacked(many_values.data());
    EXPECT_EQ(single_values, many_values);
    EXPECT_EQ(single_data.targets, many_data.targets);
}
