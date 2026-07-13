#include "task_queue.h"

#include <gtest/gtest.h>

#include <set>

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
        shape.batch_size = 4;
        return shape;
    }

} // namespace

TEST(TaskQueueTest, PublishesValidationTaskBeforeTrainingTasks) {
    TaskQueue queue(MakeShape(), 2);

    std::unique_ptr<TaskResult> validation = queue.Take();
    ASSERT_NE(validation, nullptr);
    EXPECT_EQ(validation->task.iteration, 0u);
    EXPECT_EQ(validation->task.bitness, 4);
    ASSERT_EQ(validation->data.size(), 1u);
    EXPECT_EQ(validation->data[0].kind, TensorKind::Values);
    EXPECT_EQ(validation->data[0].data.Cases(), 2u);
    EXPECT_EQ(validation->data[0].data.Reps(), 2u);
}

TEST(TaskQueueTest, PublishesTrainingTasksInIterationOrder) {
    TaskQueue queue(MakeShape(), 2);
    ASSERT_NE(queue.Take(), nullptr); // validation task

    for (uint64_t iteration = 1; iteration <= 2; ++iteration) {
        std::unique_ptr<TaskResult> result = queue.Take();
        ASSERT_NE(result, nullptr);
        EXPECT_EQ(result->task.iteration, iteration);
        EXPECT_EQ(result->task.bitness, 4);
        ASSERT_EQ(result->data.size(), 1u);
        EXPECT_EQ(result->data[0].data.Cases(), 4u);
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
