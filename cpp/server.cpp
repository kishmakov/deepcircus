#include "generator.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

    constexpr size_t kReadyCapacity = 2;
    constexpr uint64_t kSplitMixIncrement = 0x9e3779b97f4a7c15ull;

    enum class Command : uint32_t {
        Initialize = 1,
        Next = 2,
        Release = 3,
        Shutdown = 4,
    };

    enum class TensorKind : uint8_t {
        Tree = 1,
        Table = 2,
        Restrictions = 3,
    };

    enum class TensorSplit : uint8_t {
        Train = 1,
        Validation = 2,
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

    struct TensorBuffer {
        TensorKind kind;
        TensorSplit split;
        uint16_t bitness;
        uint64_t cases;
        uint64_t reps;
        uint64_t value_count;
        uint64_t target_count;
        std::unique_ptr<float[]> values;
        std::unique_ptr<float[]> targets;
    };

    struct TensorDescriptor {
        TensorKind kind;
        TensorSplit split;
        uint16_t bitness;
        uint64_t cases;
        uint64_t reps;
        uint64_t values_offset;
        uint64_t value_count;
        uint64_t targets_offset;
        uint64_t target_count;
    };

    struct SharedTask {
        ~SharedTask() {
            if (mapping != MAP_FAILED) {
                munmap(mapping, size);
            }
            if (file >= 0) {
                close(file);
            }
            if (!name.empty()) {
                shm_unlink(name.c_str());
            }
        }

        Task task;
        std::string name;
        int file = -1;
        void *mapping = MAP_FAILED;
        uint64_t size = 0;
        std::vector<TensorDescriptor> tensors;
    };

    uint64_t SplitMix64(uint64_t &state) {
        uint64_t value = (state += kSplitMixIncrement);
        value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
        value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
        return value ^ (value >> 31);
    }

    uint64_t TaskSeed(uint64_t seed, uint16_t bitness, uint64_t iteration) {
        uint64_t state = seed ^ (static_cast<uint64_t>(bitness) << 48) ^ iteration;
        return SplitMix64(state);
    }

    bool ReadExact(int socket, void *destination, size_t size) {
        char *output = static_cast<char *>(destination);
        while (size > 0) {
            const ssize_t count = recv(socket, output, size, 0);
            if (count == 0) {
                return false;
            }
            if (count < 0 && errno == EINTR) {
                continue;
            }
            assert(count > 0);
            output += count;
            size -= static_cast<size_t>(count);
        }
        return true;
    }

    bool WriteExact(int socket, const void *source, size_t size) {
        const char *input = static_cast<const char *>(source);
        while (size > 0) {
            const ssize_t count = send(socket, input, size, MSG_NOSIGNAL);
            if (count < 0 && errno == EINTR) {
                continue;
            }
            assert(count > 0);
            input += count;
            size -= static_cast<size_t>(count);
        }
        return true;
    }

    template<typename T>
    T ReadValue(int socket) {
        T value;
        const bool read = ReadExact(socket, &value, sizeof(value));
        assert(read);
        return value;
    }

    template<typename T>
    void Append(std::vector<char> &output, T value) {
        const size_t offset = output.size();
        output.resize(offset + sizeof(value));
        std::memcpy(output.data() + offset, &value, sizeof(value));
    }

    void AppendString(std::vector<char> &output, const std::string &value) {
        Append<uint32_t>(output, value.size());
        output.insert(output.end(), value.begin(), value.end());
    }

    void WriteResponse(int socket, const std::vector<char> &payload = {}) {
        constexpr uint32_t status = 0;
        const uint64_t payload_size = payload.size();
        const bool status_written = WriteExact(socket, &status, sizeof(status));
        assert(status_written);
        const bool size_written = WriteExact(socket, &payload_size, sizeof(payload_size));
        assert(size_written);
        const bool payload_written = WriteExact(socket, payload.data(), payload.size());
        assert(payload_written);
    }

    TrainingShape ReadInitialization(int socket, uint64_t payload_size) {
        constexpr uint64_t expected_size = sizeof(uint64_t) * 7 + sizeof(uint16_t) * 2;
        assert(payload_size == expected_size);

        TrainingShape shape{};
        shape.first_iteration = ReadValue<uint64_t>(socket);
        shape.last_iteration = ReadValue<uint64_t>(socket);
        shape.bitness_from = ReadValue<uint16_t>(socket);
        shape.bitness_to = ReadValue<uint16_t>(socket);
        shape.seed = ReadValue<uint64_t>(socket);
        shape.train_samples = ReadValue<uint64_t>(socket);
        shape.validation_samples = ReadValue<uint64_t>(socket);
        shape.points_per_sample = ReadValue<uint64_t>(socket);
        shape.batch_size = ReadValue<uint64_t>(socket);

        assert(shape.first_iteration <= shape.last_iteration);
        assert(shape.bitness_from > 0);
        assert(shape.bitness_from <= shape.bitness_to);
        assert(shape.train_samples > 0);
        assert(shape.validation_samples > 0);
        assert(shape.points_per_sample > 0);
        assert(shape.points_per_sample % 2 == 0);
        assert(shape.batch_size > 0);
        return shape;
    }

    void AddData(std::vector<TensorBuffer> &output, gen_generator *generator, TensorKind kind, TensorSplit split,
                 uint16_t bitness, uint64_t cases, uint64_t reps, uint64_t batch_size, uint64_t seed) {
        if (cases == 0) {
            return;
        }

        const bool recursive = kind == TensorKind::Table && bitness > gen_table_solvable_bitness();
        const uint64_t chunk_cases = recursive ? std::min(cases, batch_size) : 0;
        gen_data *data = kind == TensorKind::Tree
                                 ? gen_tree_value_tensor(generator, bitness, cases, reps, seed)
                                 : gen_table_value_tensor(generator, bitness, cases, reps, chunk_cases, seed);
        assert(data != nullptr);
        gen_data_acquire(data);

        TensorBuffer values{};
        values.kind = kind;
        values.split = split;
        values.bitness = bitness;
        values.cases = cases;
        values.reps = reps;
        values.value_count = cases * reps * (2 * bitness + 1);
        values.target_count = recursive ? 0 : cases;
        values.values.reset(gen_data_take_values(data));
        values.targets.reset(gen_data_take_targets(data));
        output.push_back(std::move(values));

        if (recursive) {
            for (uint64_t first_case = 0; first_case < cases; first_case += chunk_cases) {
                const uint64_t restriction_cases = std::min(chunk_cases, cases - first_case);
                gen_tensor *tensor = gen_table_restrictions_tensor(generator, data, first_case, restriction_cases);
                assert(tensor != nullptr);
                gen_tensor_acquire(tensor);
                TensorBuffer restrictions{};
                restrictions.kind = TensorKind::Restrictions;
                restrictions.split = split;
                restrictions.bitness = bitness;
                restrictions.cases = restriction_cases;
                restrictions.reps = reps;
                restrictions.value_count = restriction_cases * 2 * bitness * reps * (2 * bitness - 1);
                restrictions.values.reset(gen_tensor_take_values(tensor));
                output.push_back(std::move(restrictions));
                gen_tensor_release(generator, tensor);
            }
        }
        gen_data_release(generator, data);
    }

    std::vector<TensorBuffer> GenerateTask(gen_generator *generator, const TrainingShape &shape, const Task &task) {
        std::vector<TensorBuffer> output;
        const uint16_t min_tree = gen_min_tree_bitness();
        const uint16_t solvable = gen_table_solvable_bitness();
        assert(min_tree <= solvable);

        uint64_t train_table = shape.train_samples;
        uint64_t train_tree = 0;
        if (task.bitness >= min_tree) {
            assert(shape.train_samples % 2 == 0);
            train_table = shape.train_samples / 2;
            train_tree = shape.train_samples / 2;
        }

        uint64_t validation_table = shape.validation_samples;
        uint64_t validation_tree = 0;
        if (task.bitness >= min_tree && task.bitness <= solvable) {
            assert(shape.validation_samples % 2 == 0);
            validation_table = shape.validation_samples / 2;
            validation_tree = shape.validation_samples / 2;
        } else if (task.bitness > solvable) {
            validation_table = 0;
            validation_tree = shape.validation_samples;
        }

        AddData(output, generator, TensorKind::Table, TensorSplit::Train, task.bitness, train_table,
                shape.points_per_sample, shape.batch_size, task.seed ^ 0x1001);
        AddData(output, generator, TensorKind::Tree, TensorSplit::Train, task.bitness, train_tree,
                shape.points_per_sample, shape.batch_size, task.seed ^ 0x1002);
        AddData(output, generator, TensorKind::Table, TensorSplit::Validation, task.bitness, validation_table,
                shape.points_per_sample, shape.batch_size, task.seed ^ 0x2001);
        AddData(output, generator, TensorKind::Tree, TensorSplit::Validation, task.bitness, validation_tree,
                shape.points_per_sample, shape.batch_size, task.seed ^ 0x2002);
        return output;
    }

    std::unique_ptr<SharedTask> ShareTask(const Task &task, std::vector<TensorBuffer> tensors, uint64_t task_id) {
        auto shared = std::make_unique<SharedTask>();
        shared->task = task;
        shared->name = "/deepcircus_" + std::to_string(getpid()) + "_" + std::to_string(task_id);

        uint64_t size = 0;
        for (const TensorBuffer &tensor: tensors) {
            TensorDescriptor descriptor{};
            descriptor.kind = tensor.kind;
            descriptor.split = tensor.split;
            descriptor.bitness = tensor.bitness;
            descriptor.cases = tensor.cases;
            descriptor.reps = tensor.reps;
            descriptor.values_offset = size;
            descriptor.value_count = tensor.value_count;
            size += tensor.value_count * sizeof(float);
            descriptor.targets_offset = std::numeric_limits<uint64_t>::max();
            descriptor.target_count = tensor.target_count;
            if (tensor.target_count > 0) {
                descriptor.targets_offset = size;
                size += tensor.target_count * sizeof(float);
            }
            shared->tensors.push_back(descriptor);
        }
        shared->size = size;
        shared->file = shm_open(shared->name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
        assert(shared->file >= 0);
        const int truncate_result = ftruncate(shared->file, size);
        assert(truncate_result == 0);
        shared->mapping = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, shared->file, 0);
        assert(shared->mapping != MAP_FAILED);

        char *destination = static_cast<char *>(shared->mapping);
        for (size_t index = 0; index < tensors.size(); ++index) {
            const TensorBuffer &tensor = tensors[index];
            const TensorDescriptor &descriptor = shared->tensors[index];
            std::memcpy(destination + descriptor.values_offset, tensor.values.get(),
                        tensor.value_count * sizeof(float));
            if (tensor.target_count > 0) {
                std::memcpy(destination + descriptor.targets_offset, tensor.targets.get(),
                            tensor.target_count * sizeof(float));
            }
        }
        return shared;
    }

    class TaskQueue {
    public:
        TaskQueue(TrainingShape shape, size_t workers) : shape_(shape), workers_(workers) {
            for (uint64_t iteration = shape.first_iteration; iteration <= shape.last_iteration; ++iteration) {
                for (uint32_t bitness = shape.bitness_from; bitness <= shape.bitness_to; ++bitness) {
                    tasks_.push_back(Task{iteration, static_cast<uint16_t>(bitness),
                                          TaskSeed(shape.seed, static_cast<uint16_t>(bitness), iteration)});
                }
            }
            producer_ = std::thread([this] { Produce(); });
        }

        ~TaskQueue() {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                stopping_ = true;
            }
            space_.notify_all();
            if (producer_.joinable()) {
                producer_.join();
            }
        }

        std::unique_ptr<SharedTask> Take() {
            std::unique_lock<std::mutex> lock(mutex_);
            ready_.wait(lock, [this] { return !results_.empty() || finished_; });
            if (results_.empty()) {
                return nullptr;
            }
            std::unique_ptr<SharedTask> result = std::move(results_.front());
            results_.pop_front();
            lock.unlock();
            space_.notify_one();
            return result;
        }

    private:
        void Produce() {
            gen_generator *generator = gen_generator_create(workers_);
            assert(generator != nullptr);
            uint64_t task_id = 0;
            while (!tasks_.empty()) {
                const Task task = tasks_.front();
                tasks_.pop_front();
                auto result = ShareTask(task, GenerateTask(generator, shape_, task), task_id++);

                std::unique_lock<std::mutex> lock(mutex_);
                space_.wait(lock, [this] { return stopping_ || results_.size() < kReadyCapacity; });
                if (stopping_) {
                    break;
                }
                results_.push_back(std::move(result));
                lock.unlock();
                ready_.notify_one();
            }
            gen_generator_destroy(generator);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                finished_ = true;
            }
            ready_.notify_all();
        }

        TrainingShape shape_;
        size_t workers_;
        std::deque<Task> tasks_;
        std::deque<std::unique_ptr<SharedTask>> results_;
        std::thread producer_;
        std::mutex mutex_;
        std::condition_variable ready_;
        std::condition_variable space_;
        bool stopping_ = false;
        bool finished_ = false;
    };

    std::vector<char> DescribeTask(const SharedTask *task) {
        std::vector<char> response;
        Append<uint8_t>(response, task == nullptr ? 1 : 0);
        if (task == nullptr) {
            return response;
        }
        Append<uint64_t>(response, task->task.iteration);
        Append<uint16_t>(response, task->task.bitness);
        Append<uint64_t>(response, task->task.seed);
        Append<uint64_t>(response, task->size);
        AppendString(response, task->name.substr(1));
        Append<uint32_t>(response, task->tensors.size());
        for (const TensorDescriptor &tensor: task->tensors) {
            Append<uint8_t>(response, static_cast<uint8_t>(tensor.kind));
            Append<uint8_t>(response, static_cast<uint8_t>(tensor.split));
            Append<uint16_t>(response, tensor.bitness);
            Append<uint64_t>(response, tensor.cases);
            Append<uint64_t>(response, tensor.reps);
            Append<uint64_t>(response, tensor.values_offset);
            Append<uint64_t>(response, tensor.value_count);
            Append<uint64_t>(response, tensor.targets_offset);
            Append<uint64_t>(response, tensor.target_count);
        }
        return response;
    }

    void RunDaemon(int socket, size_t workers) {
        const Command command = static_cast<Command>(ReadValue<uint32_t>(socket));
        const uint64_t payload_size = ReadValue<uint64_t>(socket);
        assert(command == Command::Initialize);
        const TrainingShape shape = ReadInitialization(socket, payload_size);
        TaskQueue tasks(shape, workers);
        WriteResponse(socket);

        std::unique_ptr<SharedTask> current;
        while (true) {
            const Command next_command = static_cast<Command>(ReadValue<uint32_t>(socket));
            const uint64_t next_payload_size = ReadValue<uint64_t>(socket);
            assert(next_payload_size == 0);
            if (next_command == Command::Next) {
                assert(current == nullptr);
                current = tasks.Take();
                WriteResponse(socket, DescribeTask(current.get()));
            } else if (next_command == Command::Release) {
                assert(current != nullptr);
                current.reset();
                WriteResponse(socket);
            } else {
                assert(next_command == Command::Shutdown);
                assert(current == nullptr);
                WriteResponse(socket);
                return;
            }
        }
    }

    uint16_t ParsePort(const char *value) {
        const unsigned long port = std::strtoul(value, nullptr, 10);
        assert(port <= 65535);
        return static_cast<uint16_t>(port);
    }

    size_t ParseWorkers(const char *value) {
        const unsigned long workers = std::strtoul(value, nullptr, 10);
        assert(workers > 0);
        return workers;
    }

} // namespace

int main(int argc, char **argv) {
    uint16_t port = 7788;
    size_t workers = std::thread::hardware_concurrency();
    assert(workers > 0);
    for (int index = 1; index < argc; index += 2) {
        assert(index + 1 < argc);
        if (std::strcmp(argv[index], "--port") == 0) {
            port = ParsePort(argv[index + 1]);
        } else {
            assert(std::strcmp(argv[index], "--workers") == 0);
            workers = ParseWorkers(argv[index + 1]);
        }
    }

    const int listener = socket(AF_INET, SOCK_STREAM, 0);
    assert(listener >= 0);
    const int reuse = 1;
    const int option_result = setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    assert(option_result == 0);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(port);
    const int bind_result = bind(listener, reinterpret_cast<sockaddr *>(&address), sizeof(address));
    assert(bind_result == 0);
    const int listen_result = listen(listener, 1);
    assert(listen_result == 0);
    socklen_t address_size = sizeof(address);
    const int name_result = getsockname(listener, reinterpret_cast<sockaddr *>(&address), &address_size);
    assert(name_result == 0);
    std::cout << "PORT " << ntohs(address.sin_port) << std::endl;

    int client;
    do {
        client = accept(listener, nullptr, nullptr);
    } while (client < 0 && errno == EINTR);
    assert(client >= 0);
    RunDaemon(client, workers);
    const int client_close_result = close(client);
    assert(client_close_result == 0);
    const int listener_close_result = close(listener);
    assert(listener_close_result == 0);
    return 0;
}
