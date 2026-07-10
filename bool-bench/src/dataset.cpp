#include "bool_bench.h"
#include "decision_tree.h"
#include "table.h"
#include "tree.h"
#include "utils.h"

#include <algorithm>
#include <cassert>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

constexpr uint64_t kSplitMixIncrement = 0x9e3779b97f4a7c15ull;
constexpr uint64_t kTreeSelectionDomain = 0x747265655f73656cull;
constexpr uint64_t kTableSelectionDomain = 0x7461626c655f7365ull;
constexpr uint64_t kTreeValueDomain = 0x747265655f76616cull;
constexpr uint64_t kTableValueDomain = 0x7461626c655f7661ull;
constexpr uint64_t kRestrictionDomain = 0x7265737472696374ull;

enum class HandleState {
    Queued,
    Generating,
    Ready,
    Acquired,
    Released,
};

enum class DataKind {
    Tree,
    Table,
};

uint64_t SplitMix64(uint64_t& state) {
    uint64_t value = (state += kSplitMixIncrement);
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31);
}

uint64_t DomainSeed(uint64_t seed, uint64_t domain, uint16_t bitness) {
    uint64_t state = seed ^ domain ^ (static_cast<uint64_t>(bitness) << 48);
    return SplitMix64(state);
}

class SplitMixGenerator {
public:
    explicit SplitMixGenerator(uint64_t seed)
        : state_(seed)
    {
    }

    uint64_t Generate() {
        return SplitMix64(state_);
    }

    uint64_t GenerateBelow(uint64_t bound) {
        assert(bound > 0);
        const uint64_t threshold = static_cast<uint64_t>(-bound) % bound;
        while (true) {
            const uint64_t value = Generate();
            if (value >= threshold) {
                return value % bound;
            }
        }
    }

private:
    uint64_t state_;
};

std::vector<size_t> SampleCaseIds(
    size_t population,
    size_t cases,
    uint64_t seed)
{
    assert(cases > 0);
    assert(cases <= population);
    SplitMixGenerator rng(seed);
    std::unordered_set<size_t> selected;
    selected.reserve(cases * 2);
    std::vector<size_t> result;
    result.reserve(cases);

    for (size_t current = population - cases; current < population; ++current) {
        const size_t candidate = static_cast<size_t>(
            rng.GenerateBelow(static_cast<uint64_t>(current) + 1));
        const size_t case_id = selected.contains(candidate) ? current : candidate;
        const bool inserted = selected.insert(case_id).second;
        assert(inserted);
        result.push_back(case_id);
    }
    assert(result.size() == cases);
    return result;
}

class ThreadPool {
public:
    explicit ThreadPool(size_t workers)
        : capacity_(workers * 8)
    {
        assert(workers > 0);
        threads_.reserve(workers);
        for (size_t worker = 0; worker < workers; ++worker) {
            threads_.emplace_back([this] { Run(); });
        }
    }

    ~ThreadPool() {
        WaitIdle();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        work_.notify_all();
        for (std::thread& thread : threads_) {
            thread.join();
        }
    }

    void Enqueue(std::function<void()> task, bool high_priority) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_full_.wait(lock, [this] {
            return stopping_ || Size() < capacity_;
        });
        assert(!stopping_);
        Push(std::move(task), high_priority);
        lock.unlock();
        work_.notify_one();
    }

    bool RunHighPriorityTask() {
        std::function<void()> task;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (high_priority_tasks_.empty()) {
                return false;
            }
            task = std::move(high_priority_tasks_.front());
            high_priority_tasks_.pop_front();
            ++active_;
            not_full_.notify_one();
        }
        task();
        FinishActiveTask();
        return true;
    }

    void WaitIdle() {
        std::unique_lock<std::mutex> lock(mutex_);
        idle_.wait(lock, [this] {
            return Size() == 0 && active_ == 0;
        });
    }

private:
    void Run() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                work_.wait(lock, [this] {
                    return stopping_ || Size() > 0;
                });
                if (stopping_ && Size() == 0) {
                    return;
                }
                if (!high_priority_tasks_.empty()) {
                    task = std::move(high_priority_tasks_.front());
                    high_priority_tasks_.pop_front();
                } else {
                    task = std::move(normal_tasks_.front());
                    normal_tasks_.pop_front();
                }
                ++active_;
                not_full_.notify_one();
            }

            task();

            FinishActiveTask();
        }
    }

    void Push(std::function<void()> task, bool high_priority) {
        if (high_priority) {
            high_priority_tasks_.push_back(std::move(task));
        } else {
            normal_tasks_.push_back(std::move(task));
        }
    }

    size_t Size() const {
        return high_priority_tasks_.size() + normal_tasks_.size();
    }

    void FinishActiveTask() {
        std::lock_guard<std::mutex> lock(mutex_);
        assert(active_ > 0);
        --active_;
        if (Size() == 0 && active_ == 0) {
            idle_.notify_all();
        }
    }

    std::mutex mutex_;
    std::condition_variable work_;
    std::condition_variable idle_;
    std::condition_variable not_full_;
    std::deque<std::function<void()>> high_priority_tasks_;
    std::deque<std::function<void()>> normal_tasks_;
    std::vector<std::thread> threads_;
    size_t active_ = 0;
    size_t capacity_;
    bool stopping_ = false;
};

}  // namespace

struct bb_tensor;

struct bb_data {
    bb_generator* owner;
    DataKind kind;
    uint16_t bitness;
    size_t cases;
    size_t reps;
    uint64_t value_seed;
    uint64_t restriction_seed;
    size_t restriction_chunk_cases;
    std::vector<size_t> case_ids;
    std::vector<std::unique_ptr<DecisionTree>> trees;
    std::vector<std::unique_ptr<TableCase>> tables;
    std::unique_ptr<float[]> values;
    std::unique_ptr<float[]> targets;
    bb_tensor* prefetched_restrictions = nullptr;
    bool prefetched_claimed = false;
    size_t active_tensors = 0;

    std::mutex mutex;
    std::condition_variable ready;
    HandleState state = HandleState::Queued;
    size_t remaining = 0;
    std::atomic<size_t> next_case{0};
};

struct bb_tensor {
    bb_generator* owner;
    bb_data* parent;
    size_t first_case;
    size_t cases;
    std::unique_ptr<float[]> values;

    std::mutex mutex;
    std::condition_variable ready;
    HandleState state = HandleState::Queued;
    size_t remaining = 0;
    std::atomic<size_t> next_case{0};
};

struct bb_generator {
    explicit bb_generator(size_t workers)
        : frontend(std::this_thread::get_id())
        , workers(workers)
        , pool(std::make_unique<ThreadPool>(workers))
    {
    }

    std::thread::id frontend;
    size_t workers;
    std::unique_ptr<ThreadPool> pool;
    std::unordered_set<bb_data*> data;
    std::unordered_set<bb_tensor*> tensors;
};

namespace {

void AssertFrontend(const bb_generator* generator) {
    assert(generator != nullptr);
    assert(generator->frontend == std::this_thread::get_id());
}

void FinishDataTask(bb_data* data) {
    std::lock_guard<std::mutex> lock(data->mutex);
    assert(data->state == HandleState::Generating);
    assert(data->remaining > 0);
    --data->remaining;
    if (data->remaining == 0) {
        data->state = HandleState::Ready;
        data->ready.notify_all();
    }
}

void FinishTensorTask(bb_tensor* tensor) {
    std::lock_guard<std::mutex> lock(tensor->mutex);
    assert(tensor->state == HandleState::Generating);
    assert(tensor->remaining > 0);
    --tensor->remaining;
    if (tensor->remaining == 0) {
        tensor->state = HandleState::Ready;
        tensor->ready.notify_all();
    }
}

void RunRestrictionWorker(
    bb_generator* generator,
    bb_data* data,
    bb_tensor* tensor)
{
    const size_t values_per_case =
        2 * data->bitness * data->reps * (2 * data->bitness - 1);
    while (true) {
        const size_t offset = tensor->next_case.fetch_add(1);
        if (offset >= tensor->cases) {
            break;
        }
        const size_t case_index = tensor->first_case + offset;
        data->tables[case_index]->FillRestrictionsTensor(
            data->reps,
            CaseInputSeed(
                data->restriction_seed,
                data->bitness,
                data->case_ids[case_index]),
            tensor->values.get() + offset * values_per_case);
    }
    FinishTensorTask(tensor);
}

void GenerateDataCase(bb_data* data, size_t case_index) {
    const size_t sample_size = 2 * data->bitness + 1;
    const size_t case_id = data->case_ids[case_index];
    if (data->kind == DataKind::Tree) {
        auto tree = std::make_unique<DecisionTree>(
            BuildTreeCase(data->bitness, case_id));
        tree->FillValueTensor(
            data->reps,
            CaseInputSeed(data->value_seed, data->bitness, case_id),
            data->values.get() + case_index * data->reps * sample_size);
        data->targets[case_index] = static_cast<float>(
            data->bitness - tree->depth);
        data->trees[case_index] = std::move(tree);
    } else {
        auto table = std::make_unique<TableCase>(data->bitness, case_id);
        table->FillValueTensor(
            data->reps,
            CaseInputSeed(data->value_seed, data->bitness, case_id),
            data->values.get() + case_index * data->reps * sample_size);
        if (data->bitness <= kSolvableTableBitness) {
            const size_t depth = SolveForDepth(
                data->bitness,
                table->TruthTable());
            data->targets[case_index] = static_cast<float>(
                data->bitness - depth);
        }
        data->tables[case_index] = std::move(table);

        if (case_index < data->restriction_chunk_cases) {
            bb_tensor* tensor = data->prefetched_restrictions;
            const size_t values_per_case =
                2 * data->bitness * data->reps * (2 * data->bitness - 1);
            data->tables[case_index]->FillRestrictionsTensor(
                data->reps,
                CaseInputSeed(
                    data->restriction_seed,
                    data->bitness,
                    case_id),
                tensor->values.get() + case_index * values_per_case);
            FinishTensorTask(tensor);
        }
    }
}

void RunDataWorker(bb_generator* generator, bb_data* data) {
    while (true) {
        const size_t case_index = data->next_case.fetch_add(1);
        if (case_index >= data->cases) {
            break;
        }
        GenerateDataCase(data, case_index);
        if (case_index % 8 == 0) {
            while (generator->pool->RunHighPriorityTask()) {
            }
        }
    }
    FinishDataTask(data);
}

bb_tensor* MakeRestrictionTensor(
    bb_generator* generator,
    bb_data* data,
    size_t first_case,
    size_t cases,
    bool enqueue)
{
    assert(data->kind == DataKind::Table);
    assert(data->bitness > kSolvableTableBitness);
    assert(cases > 0);
    assert(first_case + cases <= data->cases);

    auto tensor = std::make_unique<bb_tensor>();
    tensor->owner = generator;
    tensor->parent = data;
    tensor->first_case = first_case;
    tensor->cases = cases;
    tensor->remaining = cases;
    tensor->state = HandleState::Generating;
    const size_t values_per_case =
        2 * data->bitness * data->reps * (2 * data->bitness - 1);
    tensor->values = std::make_unique<float[]>(cases * values_per_case);
    bb_tensor* result = tensor.release();
    const bool inserted = generator->tensors.insert(result).second;
    assert(inserted);

    if (enqueue) {
        const size_t workers = std::min(generator->workers, cases);
        result->remaining = workers;
        for (size_t worker = 0; worker < workers; ++worker) {
            generator->pool->Enqueue(
                [generator, data, result] {
                    RunRestrictionWorker(generator, data, result);
                },
                /*high_priority=*/true);
        }
    }
    return result;
}

bb_data* MakeData(
    bb_generator* generator,
    DataKind kind,
    uint16_t bitness,
    size_t cases,
    size_t reps,
    size_t restriction_chunk_cases,
    uint64_t seed)
{
    AssertFrontend(generator);
    assert(cases > 0);
    assert(reps > 0);
    assert(reps % 2 == 0);
    if (kind == DataKind::Tree) {
        assert(bitness >= kMinTreeBitness && bitness <= kMaxTreeBitness);
        assert(restriction_chunk_cases == 0);
    } else {
        assert(bitness >= kMinTableBitness && bitness <= kMaxTableBitness);
        if (bitness <= kSolvableTableBitness) {
            assert(restriction_chunk_cases == 0);
        } else {
            assert(restriction_chunk_cases > 0);
            assert(restriction_chunk_cases <= cases);
        }
    }

    auto data = std::make_unique<bb_data>();
    data->owner = generator;
    data->kind = kind;
    data->bitness = bitness;
    data->cases = cases;
    data->reps = reps;
    data->restriction_chunk_cases = restriction_chunk_cases;
    data->value_seed = DomainSeed(
        seed,
        kind == DataKind::Tree ? kTreeValueDomain : kTableValueDomain,
        bitness);
    data->restriction_seed = DomainSeed(seed, kRestrictionDomain, bitness);
    const size_t population = kind == DataKind::Tree
        ? bb_tree_cases_number(bitness)
        : bb_table_cases_number(bitness);
    data->case_ids = SampleCaseIds(
        population,
        cases,
        DomainSeed(
            seed,
            kind == DataKind::Tree ? kTreeSelectionDomain : kTableSelectionDomain,
            bitness));
    data->values = std::make_unique<float[]>(
        cases * reps * (2 * bitness + 1));
    if (kind == DataKind::Tree || bitness <= kSolvableTableBitness) {
        data->targets = std::make_unique<float[]>(cases);
    }
    if (kind == DataKind::Tree) {
        data->trees.resize(cases);
    } else {
        data->tables.resize(cases);
    }
    data->state = HandleState::Generating;

    bb_data* result = data.release();
    const bool inserted = generator->data.insert(result).second;
    assert(inserted);

    if (kind == DataKind::Table && bitness > kSolvableTableBitness) {
        result->prefetched_restrictions = MakeRestrictionTensor(
            generator,
            result,
            /*first_case=*/0,
            restriction_chunk_cases,
            /*enqueue=*/false);
    }

    const size_t workers = std::min(generator->workers, cases);
    result->remaining = workers;
    for (size_t worker = 0; worker < workers; ++worker) {
        generator->pool->Enqueue(
            [generator, result] {
                RunDataWorker(generator, result);
            },
            /*high_priority=*/false);
    }
    return result;
}

}  // namespace

bb_generator* bb_generator_create(size_t workers) {
    assert(workers > 0);
    return new bb_generator(workers);
}

void bb_generator_destroy(bb_generator* generator) {
    if (generator == nullptr) {
        return;
    }
    AssertFrontend(generator);
    generator->pool->WaitIdle();
    for (bb_tensor* tensor : generator->tensors) {
        delete tensor;
    }
    generator->tensors.clear();
    for (bb_data* data : generator->data) {
        delete data;
    }
    generator->data.clear();
    generator->pool.reset();
    delete generator;
}

bb_data* bb_tree_value_tensor(
    bb_generator* generator,
    uint16_t bitness,
    size_t cases,
    size_t reps,
    uint64_t seed)
{
    return MakeData(
        generator,
        DataKind::Tree,
        bitness,
        cases,
        reps,
        /*restriction_chunk_cases=*/0,
        seed);
}

bb_data* bb_table_value_tensor(
    bb_generator* generator,
    uint16_t bitness,
    size_t cases,
    size_t reps,
    size_t restriction_chunk_cases,
    uint64_t seed)
{
    return MakeData(
        generator,
        DataKind::Table,
        bitness,
        cases,
        reps,
        restriction_chunk_cases,
        seed);
}

void bb_data_acquire(bb_data* data) {
    assert(data != nullptr);
    AssertFrontend(data->owner);
    std::unique_lock<std::mutex> lock(data->mutex);
    assert(
        data->state == HandleState::Generating
        || data->state == HandleState::Ready);
    data->ready.wait(lock, [data] {
        return data->state == HandleState::Ready;
    });
    data->state = HandleState::Acquired;
}

uint16_t bb_data_bitness(const bb_data* data) {
    assert(data != nullptr);
    assert(data->state == HandleState::Acquired);
    return data->bitness;
}

size_t bb_data_cases(const bb_data* data) {
    assert(data != nullptr);
    assert(data->state == HandleState::Acquired);
    return data->cases;
}

size_t bb_data_reps(const bb_data* data) {
    assert(data != nullptr);
    assert(data->state == HandleState::Acquired);
    return data->reps;
}

float* bb_data_take_values(bb_data* data) {
    assert(data != nullptr);
    assert(data->state == HandleState::Acquired);
    assert(data->values != nullptr);
    return data->values.release();
}

float* bb_data_take_targets(bb_data* data) {
    assert(data != nullptr);
    assert(data->state == HandleState::Acquired);
    return data->targets.release();
}

void bb_data_release(bb_generator* generator, bb_data* data) {
    AssertFrontend(generator);
    assert(data != nullptr);
    assert(data->owner == generator);
    assert(data->state == HandleState::Acquired);
    assert(data->active_tensors == 0);

    if (data->prefetched_restrictions != nullptr) {
        bb_tensor* tensor = data->prefetched_restrictions;
        assert(!data->prefetched_claimed);
        assert(tensor->state == HandleState::Ready);
        const size_t erased_tensor = generator->tensors.erase(tensor);
        assert(erased_tensor == 1);
        delete tensor;
        data->prefetched_restrictions = nullptr;
    }
    data->state = HandleState::Released;
    const size_t erased = generator->data.erase(data);
    assert(erased == 1);
    delete data;
}

bb_tensor* bb_table_restrictions_tensor(
    bb_generator* generator,
    bb_data* table_data,
    size_t first_case,
    size_t cases)
{
    AssertFrontend(generator);
    assert(table_data != nullptr);
    assert(table_data->owner == generator);
    assert(table_data->state == HandleState::Acquired);
    assert(table_data->kind == DataKind::Table);
    assert(table_data->bitness > kSolvableTableBitness);
    assert(cases > 0);
    assert(first_case + cases <= table_data->cases);

    if (
        first_case == 0
        && cases == table_data->restriction_chunk_cases
        && !table_data->prefetched_claimed)
    {
        table_data->prefetched_claimed = true;
        ++table_data->active_tensors;
        return table_data->prefetched_restrictions;
    }

    ++table_data->active_tensors;
    return MakeRestrictionTensor(
        generator,
        table_data,
        first_case,
        cases,
        /*enqueue=*/true);
}

void bb_tensor_acquire(bb_tensor* tensor) {
    assert(tensor != nullptr);
    AssertFrontend(tensor->owner);
    std::unique_lock<std::mutex> lock(tensor->mutex);
    assert(
        tensor->state == HandleState::Generating
        || tensor->state == HandleState::Ready);
    tensor->ready.wait(lock, [tensor] {
        return tensor->state == HandleState::Ready;
    });
    tensor->state = HandleState::Acquired;
}

float* bb_tensor_take_values(bb_tensor* tensor) {
    assert(tensor != nullptr);
    assert(tensor->state == HandleState::Acquired);
    assert(tensor->values != nullptr);
    return tensor->values.release();
}

void bb_tensor_release(bb_generator* generator, bb_tensor* tensor) {
    AssertFrontend(generator);
    assert(tensor != nullptr);
    assert(tensor->owner == generator);
    assert(tensor->state == HandleState::Acquired);
    bb_data* parent = tensor->parent;
    assert(parent != nullptr);
    assert(parent->active_tensors > 0);
    --parent->active_tensors;
    if (parent->prefetched_restrictions == tensor) {
        assert(parent->prefetched_claimed);
        parent->prefetched_restrictions = nullptr;
    }
    tensor->state = HandleState::Released;
    const size_t erased = generator->tensors.erase(tensor);
    assert(erased == 1);
    delete tensor;
}

void bb_float_buffer_destroy(float* buffer) {
    delete[] buffer;
}
