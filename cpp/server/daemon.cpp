#include "daemon.h"

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
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

namespace {

enum class Command : uint32_t {
    Initialize = 1,
    Next = 2,
    Release = 3,
};

enum class TensorKind : uint8_t {
    Values = 1,
    Restrictions = 2,
};

struct TensorDescriptor {
    TensorKind kind;
    uint16_t bitness;
    uint64_t cases;
    uint64_t reps;
    uint64_t values_offset;
    uint64_t value_count;
    uint64_t targets_offset;
    uint64_t target_count;
};

struct TensorSource {
    const gen::Values* values = nullptr;
    const gen::Restrictions* restrictions = nullptr;
    const std::vector<float>* targets = nullptr;
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
    void* mapping = MAP_FAILED;
    uint64_t size = 0;
    std::vector<TensorDescriptor> tensors;
};

bool ReadExact(int socket, void* destination, size_t size) {
    char* output = static_cast<char*>(destination);
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

bool WriteExact(int socket, const void* source, size_t size) {
    const char* input = static_cast<const char*>(source);
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

template <typename T>
T ReadValue(int socket) {
    T value;
    const bool read = ReadExact(socket, &value, sizeof(value));
    assert(read);
    return value;
}

template <typename T>
void Append(std::vector<char>& output, T value) {
    const size_t offset = output.size();
    output.resize(offset + sizeof(value));
    std::memcpy(output.data() + offset, &value, sizeof(value));
}

void AppendString(std::vector<char>& output, const std::string& value) {
    Append<uint32_t>(output, value.size());
    output.insert(output.end(), value.begin(), value.end());
}

void WriteResponse(int socket, const std::vector<char>& payload = {}) {
    constexpr uint32_t status = 0;
    const uint64_t payload_size = payload.size();
    const bool status_written = WriteExact(socket, &status, sizeof(status));
    assert(status_written);
    const bool size_written = WriteExact(socket, &payload_size, sizeof(payload_size));
    assert(size_written);
    const bool payload_written = WriteExact(socket, payload.data(), payload.size());
    assert(payload_written);
}

// Publish one compact coordinate into its shared-memory layout: value and
// restriction bits stay bit-packed (rows padded to whole bytes, little-endian
// bit order; the client unpacks bit b to the value 2*b - 1), while exact
// targets are written as float32.
std::unique_ptr<SharedTask> ShareTask(const TaskResult& result, uint64_t task_id) {
    auto shared = std::make_unique<SharedTask>();
    shared->task = result.task;
    shared->name = "/deepcircus_" + std::to_string(getpid()) + "_" + std::to_string(task_id);
    std::vector<TensorSource> sources;

    uint64_t size = 0;
    const auto align_floats = [&size] { size = (size + alignof(float) - 1) / alignof(float) * alignof(float); };
    const auto append_values = [&](const gen::Values& values, const std::vector<float>* targets) {
        const size_t point_size = 2 * result.task.bitness + 1;
        assert(values.Columns() % point_size == 0);
        TensorDescriptor descriptor{};
        descriptor.kind = TensorKind::Values;
        descriptor.bitness = result.task.bitness;
        descriptor.cases = values.Rows();
        descriptor.reps = values.Columns() / point_size;
        descriptor.values_offset = size;
        descriptor.value_count = values.ValueCount();
        size += values.ByteCount();
        descriptor.targets_offset = std::numeric_limits<uint64_t>::max();
        if (targets != nullptr) {
            assert(targets->size() == gen::kTargetsPerCase * values.Rows());
            descriptor.target_count = targets->size();
            align_floats();
            descriptor.targets_offset = size;
            size += descriptor.target_count * sizeof(float);
        }
        shared->tensors.push_back(descriptor);
        sources.push_back(TensorSource{&values, nullptr, targets});
    };
    const auto append_restrictions = [&](const gen::Restrictions& restrictions) {
        const size_t point_size = 2 * result.task.bitness * (2 * result.task.bitness - 1);
        assert(restrictions.Columns() % point_size == 0);
        TensorDescriptor descriptor{};
        descriptor.kind = TensorKind::Restrictions;
        descriptor.bitness = result.task.bitness;
        descriptor.cases = restrictions.Rows();
        descriptor.reps = restrictions.Columns() / point_size;
        descriptor.values_offset = size;
        descriptor.value_count = restrictions.ValueCount();
        size += restrictions.ByteCount();
        descriptor.targets_offset = std::numeric_limits<uint64_t>::max();
        shared->tensors.push_back(descriptor);
        sources.push_back(TensorSource{{}, &restrictions, nullptr});
    };
    for (const gen::GeneratedValues& generated : result.values) {
        append_values(generated.values, &generated.targets);
    }
    for (const gen::GeneratedRestrictions& generated : result.restrictions) {
        append_values(generated.values, nullptr);
        append_restrictions(generated.restrictions);
    }

    shared->size = size;
    shared->file = shm_open(shared->name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    assert(shared->file >= 0);
    const int truncate_result = ftruncate(shared->file, size);
    assert(truncate_result == 0);
    shared->mapping = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, shared->file, 0);
    assert(shared->mapping != MAP_FAILED);

    char* destination = static_cast<char*>(shared->mapping);
    for (size_t index = 0; index < sources.size(); ++index) {
        const TensorSource& source = sources[index];
        const TensorDescriptor& descriptor = shared->tensors[index];
        uint8_t* values = reinterpret_cast<uint8_t*>(destination + descriptor.values_offset);
        if (source.restrictions != nullptr) {
            source.restrictions->WritePacked(values);
            continue;
        }
        assert(source.values != nullptr);
        source.values->WritePacked(values);
        if (descriptor.target_count > 0) {
            assert(source.targets != nullptr);
            float* targets = reinterpret_cast<float*>(destination + descriptor.targets_offset);
            std::copy(source.targets->begin(), source.targets->end(), targets);
        }
    }
    return shared;
}

std::vector<char> DescribeTask(const SharedTask* task) {
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
    for (const TensorDescriptor& tensor : task->tensors) {
        Append<uint8_t>(response, static_cast<uint8_t>(tensor.kind));
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

}  // namespace

int AcceptClient(uint16_t port) {
    const int listener = socket(AF_INET, SOCK_STREAM, 0);
    assert(listener >= 0);
    const int reuse = 1;
    const int option_result = setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    assert(option_result == 0);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(port);
    const int bind_result = bind(listener, reinterpret_cast<sockaddr*>(&address), sizeof(address));
    assert(bind_result == 0);
    const int listen_result = listen(listener, 1);
    assert(listen_result == 0);
    socklen_t address_size = sizeof(address);
    const int name_result = getsockname(listener, reinterpret_cast<sockaddr*>(&address), &address_size);
    assert(name_result == 0);
    std::cout << "PORT " << ntohs(address.sin_port) << std::endl;

    int client;
    do {
        client = accept(listener, nullptr, nullptr);
    } while (client < 0 && errno == EINTR);
    assert(client >= 0);
    const int listener_close_result = close(listener);
    assert(listener_close_result == 0);
    return client;
}

TrainingShape ReadTrainingShape(int client) {
    uint32_t command_value;
    const bool command_read = ReadExact(client, &command_value, sizeof(command_value));
    assert(command_read);
    const Command command = static_cast<Command>(command_value);
    const uint64_t payload_size = ReadValue<uint64_t>(client);
    assert(command == Command::Initialize);
    constexpr uint64_t expected_size = sizeof(uint64_t) * 6 + sizeof(uint16_t) * 2;
    assert(payload_size == expected_size);

    TrainingShape shape{};
    shape.first_iteration = ReadValue<uint64_t>(client);
    shape.last_iteration = ReadValue<uint64_t>(client);
    shape.bitness_from = ReadValue<uint16_t>(client);
    shape.bitness_to = ReadValue<uint16_t>(client);
    shape.seed = ReadValue<uint64_t>(client);
    shape.train_samples = ReadValue<uint64_t>(client);
    shape.validation_samples = ReadValue<uint64_t>(client);
    shape.points_per_sample = ReadValue<uint64_t>(client);

    assert(shape.first_iteration <= shape.last_iteration);
    assert(shape.bitness_from > 0);
    assert(shape.bitness_from <= shape.bitness_to);
    assert(shape.train_samples > 0);
    assert(shape.validation_samples > 0);
    assert(shape.points_per_sample > 0);
    assert(shape.points_per_sample % 2 == 0);
    return shape;
}

void ServeTasks(int client, const std::function<std::unique_ptr<TaskResult>()>& take) {
    WriteResponse(client);

    std::unique_ptr<SharedTask> current;
    uint64_t task_id = 0;
    while (true) {
        // The client hangs up when it is done with its tensors.
        uint32_t command_value;
        if (!ReadExact(client, &command_value, sizeof(command_value))) {
            return;
        }
        const Command command = static_cast<Command>(command_value);
        const uint64_t payload_size = ReadValue<uint64_t>(client);
        assert(payload_size == 0);
        if (command == Command::Next) {
            assert(current == nullptr);
            std::unique_ptr<TaskResult> result = take();
            current = result == nullptr ? nullptr : ShareTask(*result, task_id++);
            WriteResponse(client, DescribeTask(current.get()));
        } else {
            assert(command == Command::Release);
            assert(current != nullptr);
            current.reset();
            WriteResponse(client);
        }
    }
}
