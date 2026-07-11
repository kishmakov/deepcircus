#include "generator.h"
#include "thread_pool.h"

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

    struct DataDeleter {
        void operator()(gen_data *data) const { gen_data_destroy(data); }
    };

    struct TaskData {
        TensorKind kind;
        TensorSplit split;
        std::unique_ptr<gen_data, DataDeleter> data;
    };

    struct TaskResult {
        Task task;
        // Worker output remains compact until this coordinate is published.
        std::vector<TaskData> data;
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

    struct TensorSource {
        const gen_data *data = nullptr;
        const gen_tensor *tensor = nullptr;
    };

    struct SharedTask {
        // At most the coordinate currently borrowed by Python has this form.
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

    void AppendParams(const TrainingShape &shape) {
        std::filesystem::create_directories("/tmp/circus");
        std::ofstream output("/tmp/circus/params.txt", std::ios::app);
        assert(output.is_open());
        output << "first_iteration=" << shape.first_iteration << " last_iteration=" << shape.last_iteration
               << " bitness_from=" << shape.bitness_from << " bitness_to=" << shape.bitness_to << " seed=" << shape.seed
               << " train_samples=" << shape.train_samples << " validation_samples=" << shape.validation_samples
               << " points_per_sample=" << shape.points_per_sample << " batch_size=" << shape.batch_size << '\n';
        assert(output.good());
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
        AppendParams(shape);
        return shape;
    }

    void AddData(std::vector<TaskData> &output, TensorKind kind, TensorSplit split, uint16_t bitness, uint64_t cases,
                 uint64_t reps, uint64_t batch_size, uint64_t seed) {
        if (cases == 0) {
            return;
        }
        const bool recursive = kind == TensorKind::Table && bitness > gen_table_solvable_bitness();
        const uint64_t chunk_cases = recursive ? std::min(cases, batch_size) : 0;
        gen_data *generated = kind == TensorKind::Tree
                                      ? gen_tree_value_tensor(bitness, cases, reps, seed)
                                      : gen_table_value_tensor(bitness, cases, reps, chunk_cases, seed);
        assert(generated != nullptr);
        output.push_back(TaskData{kind, split, std::unique_ptr<gen_data, DataDeleter>(generated)});
    }

    std::unique_ptr<TaskResult> GenerateTask(const TrainingShape &shape, const Task &task) {
        auto result = std::make_unique<TaskResult>();
        result->task = task;
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

        AddData(result->data, TensorKind::Table, TensorSplit::Train, task.bitness, train_table, shape.points_per_sample,
                shape.batch_size, task.seed ^ 0x1001);
        AddData(result->data, TensorKind::Tree, TensorSplit::Train, task.bitness, train_tree, shape.points_per_sample,
                shape.batch_size, task.seed ^ 0x1002);
        AddData(result->data, TensorKind::Table, TensorSplit::Validation, task.bitness, validation_table,
                shape.points_per_sample, shape.batch_size, task.seed ^ 0x2001);
        AddData(result->data, TensorKind::Tree, TensorSplit::Validation, task.bitness, validation_tree,
                shape.points_per_sample, shape.batch_size, task.seed ^ 0x2002);
        return result;
    }

    // Expand one compact coordinate directly into its final shared-memory layout.
    std::unique_ptr<SharedTask> ShareTask(const TaskResult &result, uint64_t task_id) {
        auto shared = std::make_unique<SharedTask>();
        shared->task = result.task;
        shared->name = "/deepcircus_" + std::to_string(getpid()) + "_" + std::to_string(task_id);
        std::vector<TensorSource> sources;

        uint64_t size = 0;
        for (const TaskData &task_data: result.data) {
            const gen_data *data = task_data.data.get();
            TensorDescriptor descriptor{};
            descriptor.kind = task_data.kind;
            descriptor.split = task_data.split;
            descriptor.bitness = gen_data_bitness(data);
            descriptor.cases = gen_data_cases(data);
            descriptor.reps = gen_data_reps(data);
            descriptor.values_offset = size;
            descriptor.value_count = gen_data_value_count(data);
            size += descriptor.value_count * sizeof(float);
            descriptor.targets_offset = std::numeric_limits<uint64_t>::max();
            descriptor.target_count = gen_data_target_count(data);
            if (descriptor.target_count > 0) {
                descriptor.targets_offset = size;
                size += descriptor.target_count * sizeof(float);
            }
            shared->tensors.push_back(descriptor);
            sources.push_back(TensorSource{data, nullptr});

            const size_t restriction_count = gen_data_restriction_count(data);
            for (size_t index = 0; index < restriction_count; ++index) {
                const gen_tensor *tensor = gen_data_restriction(data, index);
                TensorDescriptor restriction{};
                restriction.kind = TensorKind::Restrictions;
                restriction.split = task_data.split;
                restriction.bitness = gen_tensor_bitness(tensor);
                restriction.cases = gen_tensor_cases(tensor);
                restriction.reps = gen_tensor_reps(tensor);
                restriction.values_offset = size;
                restriction.value_count = gen_tensor_value_count(tensor);
                size += restriction.value_count * sizeof(float);
                restriction.targets_offset = std::numeric_limits<uint64_t>::max();
                shared->tensors.push_back(restriction);
                sources.push_back(TensorSource{nullptr, tensor});
            }
        }

        shared->size = size;
        shared->file = shm_open(shared->name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
        assert(shared->file >= 0);
        const int truncate_result = ftruncate(shared->file, size);
        assert(truncate_result == 0);
        shared->mapping = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, shared->file, 0);
        assert(shared->mapping != MAP_FAILED);

        char *destination = static_cast<char *>(shared->mapping);
        for (size_t index = 0; index < sources.size(); ++index) {
            const TensorSource &source = sources[index];
            const TensorDescriptor &descriptor = shared->tensors[index];
            float *values = reinterpret_cast<float *>(destination + descriptor.values_offset);
            if (source.data != nullptr) {
                gen_data_write_values(source.data, values);
                if (descriptor.target_count > 0) {
                    float *targets = reinterpret_cast<float *>(destination + descriptor.targets_offset);
                    gen_data_write_targets(source.data, targets);
                }
            } else {
                gen_tensor_write_values(source.tensor, values);
            }
        }
        return shared;
    }

    class TaskQueue {
    public:
        TaskQueue(TrainingShape shape, size_t workers) : shape_(shape), pool_(workers) {
            for (uint64_t iteration = shape.first_iteration; iteration <= shape.last_iteration; ++iteration) {
                for (uint32_t bitness = shape.bitness_from; bitness <= shape.bitness_to; ++bitness) {
                    tasks_.push_back(Task{iteration, static_cast<uint16_t>(bitness),
                                          TaskSeed(shape.seed, static_cast<uint16_t>(bitness), iteration)});
                }
            }
            results_.resize(tasks_.size());
            // Fixed result indices decouple parallel completion from publication order.
            for (size_t index = 0; index < tasks_.size(); ++index) {
                const Task task = tasks_[index];
                pool_.Enqueue([this, index, task] {
                    std::unique_ptr<TaskResult> result = GenerateTask(shape_, task);
                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        assert(results_[index] == nullptr);
                        results_[index] = std::move(result);
                    }
                    ready_.notify_all();
                });
            }
        }

        std::unique_ptr<TaskResult> Take() {
            if (next_ == results_.size()) {
                return nullptr;
            }
            std::unique_lock<std::mutex> lock(mutex_);
            ready_.wait(lock, [this] { return results_[next_] != nullptr; });
            std::unique_ptr<TaskResult> result = std::move(results_[next_]);
            ++next_;
            return result;
        }

    private:
        TrainingShape shape_;
        std::vector<Task> tasks_;
        std::vector<std::unique_ptr<TaskResult>> results_;
        size_t next_ = 0;
        std::mutex mutex_;
        std::condition_variable ready_;
        ThreadPool pool_;
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
        uint64_t task_id = 0;
        while (true) {
            const Command next_command = static_cast<Command>(ReadValue<uint32_t>(socket));
            const uint64_t next_payload_size = ReadValue<uint64_t>(socket);
            assert(next_payload_size == 0);
            if (next_command == Command::Next) {
                assert(current == nullptr);
                std::unique_ptr<TaskResult> result = tasks.Take();
                current = result == nullptr ? nullptr : ShareTask(*result, task_id++);
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
