#include <gtest/gtest.h>

#include <set>

#include "task_queue.h"

namespace {

TrainingShape MakeShape() {
    TrainingShape shape{};
    shape.first_iteration = 1;
    shape.last_iteration = 2;
    shape.bitness_from = 4;
    shape.bitness_to = 4;
    shape.seed = 42;
    shape.train_samples = 4;
    shape.validation_samples = 2;
    shape.points_per_sample = 2;
    return shape;
}

}  // namespace

TEST(TaskQueueTest, PublishesValidationTaskBeforeTrainingTasks) {
    TaskQueue queue(MakeShape(), 2);

    std::unique_ptr<TaskResult> validation = queue.Take();
    ASSERT_NE(validation, nullptr);
    EXPECT_EQ(validation->task.iteration, 0u);
    EXPECT_EQ(validation->task.bitness, 4);
    ASSERT_EQ(validation->values.size(), 1u);
    EXPECT_TRUE(validation->restrictions.empty());
    EXPECT_EQ(validation->values[0].values.Rows(), 2u);
    EXPECT_EQ(validation->values[0].values.Columns(), 2u * (2 * 4 + 1));
    EXPECT_EQ(validation->values[0].targets.size(), 2u);
}

TEST(TaskQueueTest, PublishesTrainingTasksInIterationOrder) {
    TaskQueue queue(MakeShape(), 2);
    ASSERT_NE(queue.Take(), nullptr);  // validation task

    for (uint64_t iteration = 1; iteration <= 2; ++iteration) {
        std::unique_ptr<TaskResult> result = queue.Take();
        ASSERT_NE(result, nullptr);
        EXPECT_EQ(result->task.iteration, iteration);
        EXPECT_EQ(result->task.bitness, 4);
        ASSERT_EQ(result->values.size(), 1u);
        EXPECT_TRUE(result->restrictions.empty());
        EXPECT_EQ(result->values[0].values.Rows(), 4u);
        EXPECT_EQ(result->values[0].targets.size(), 4u);
    }
}

TEST(TaskQueueTest, ReturnsNullAfterExhaustingAllTasks) {
    TaskQueue queue(MakeShape(), 2);
    for (int i = 0; i < 3; ++i) {
        ASSERT_NE(queue.Take(), nullptr);
    }
    EXPECT_EQ(queue.Take(), nullptr);
}

TEST(TaskQueueTest, SeedsAreDeterministicAcrossInstancesRegardlessOfWorkerCount) {
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

TEST(TaskQueueTest, DistinctCoordinatesGetDistinctSeeds) {
    TaskQueue queue(MakeShape(), 2);
    std::set<uint64_t> seeds;
    for (int i = 0; i < 3; ++i) {
        std::unique_ptr<TaskResult> result = queue.Take();
        ASSERT_NE(result, nullptr);
        seeds.insert(result->task.seed);
    }
    EXPECT_EQ(seeds.size(), 3u);
}

// Different worker counts change how a coordinate's case ids are chunked
// across the pool internally; the merged batch must come out byte-identical
// regardless, since chunking is an execution detail, not a sampling input.
TEST(TaskQueueTest, DataIsIdenticalAcrossDifferentWorkerCounts) {
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

    std::vector<float> single_values(single_data.values.ValueCount());
    std::vector<float> many_values(many_data.values.ValueCount());
    single_data.values.WriteValues(single_values.data());
    many_data.values.WriteValues(many_values.data());
    EXPECT_EQ(single_values, many_values);
    EXPECT_EQ(single_data.targets, many_data.targets);
}

namespace {

TrainingShape MakeRecursiveShape() {
    TrainingShape shape{};
    shape.first_iteration = 1;
    shape.last_iteration = 1;
    shape.bitness_from = 17;  // one above TableSolvableBitness(): recursive table path
    shape.bitness_to = 17;
    shape.seed = 7;
    shape.train_samples = 8;
    shape.validation_samples = 6;
    shape.points_per_sample = 2;
    return shape;
}

}  // namespace

// Above TableSolvableBitness, training tasks carry recursive table values
// paired with their restriction matrices; the per-worker chunks are merged
// back into a single pair preserving case order and total case count.
TEST(TaskQueueTest, RecursiveBitnessMergesRestrictionChunks) {
    TaskQueue queue(MakeRecursiveShape(), 2);
    ASSERT_NE(queue.Take(),
              nullptr);  // validation task: tree-only, no restrictions

    std::unique_ptr<TaskResult> train = queue.Take();
    ASSERT_NE(train, nullptr);
    ASSERT_EQ(train->task.iteration, 1u);
    ASSERT_EQ(train->values.size(), 1u);  // exact tree values
    ASSERT_EQ(train->restrictions.size(), 1u);  // merged across worker chunks
    EXPECT_EQ(train->values[0].values.Rows(), 4u);
    EXPECT_EQ(train->values[0].targets.size(), 4u);

    const gen::GeneratedRestrictions& merged = train->restrictions[0];
    EXPECT_EQ(merged.values.Rows(), 4u);
    EXPECT_EQ(merged.values.Rows(), merged.restrictions.Rows());
    EXPECT_EQ(merged.values.Columns(), 2u * (2 * 17 + 1));
    EXPECT_EQ(merged.restrictions.Columns(), 2u * 17 * 2 * (2 * 17 - 1));
}

namespace {

template <typename Tensor>
std::vector<float> TensorFloats(const Tensor& tensor) {
    std::vector<float> values(tensor.ValueCount());
    tensor.WriteValues(values.data());
    return values;
}

}  // namespace

// Different worker counts change how a coordinate's case ids are chunked
// across the pool internally; the merged restriction pair must come out
// byte-identical regardless, since chunking is an execution detail.
TEST(TaskQueueTest, RecursiveDataIsIdenticalAcrossDifferentWorkerCounts) {
    TaskQueue single(MakeRecursiveShape(), 1);
    TaskQueue many(MakeRecursiveShape(), 4);
    ASSERT_NE(single.Take(), nullptr);  // validation task
    ASSERT_NE(many.Take(), nullptr);    // validation task

    std::unique_ptr<TaskResult> from_single = single.Take();
    std::unique_ptr<TaskResult> from_many = many.Take();
    ASSERT_NE(from_single, nullptr);
    ASSERT_NE(from_many, nullptr);
    ASSERT_EQ(from_single->values.size(), 1u);
    ASSERT_EQ(from_many->values.size(), 1u);
    ASSERT_EQ(from_single->restrictions.size(), 1u);
    ASSERT_EQ(from_many->restrictions.size(), 1u);

    EXPECT_EQ(TensorFloats(from_single->restrictions[0].values),
              TensorFloats(from_many->restrictions[0].values));
    EXPECT_EQ(TensorFloats(from_single->restrictions[0].restrictions),
              TensorFloats(from_many->restrictions[0].restrictions));
}
