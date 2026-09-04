#include "daemon.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cassert>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

namespace {

enum class Command : uint32_t {
    Initialize = 1,
    Epoch = 2,
};

// False once the client has hung up; anything else that goes wrong asserts.
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

void WriteExact(int socket, const void* source, size_t size) {
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
}

template <typename T>
T ReadValue(int socket) {
    T value;
    const bool read = ReadExact(socket, &value, sizeof(value));
    assert(read);
    return value;
}

std::string ReadString(int socket) {
    const uint32_t size = ReadValue<uint32_t>(socket);
    std::string value(size, '\0');
    const bool read = ReadExact(socket, value.data(), size);
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

void WriteResponse(int socket, const std::vector<char>& payload) {
    constexpr uint32_t status = 0;
    const uint64_t payload_size = payload.size();
    WriteExact(socket, &status, sizeof(status));
    WriteExact(socket, &payload_size, sizeof(payload_size));
    WriteExact(socket, payload.data(), payload.size());
}

// One epoch's cases, mapped where the client can read them: the packed rows,
// then the case targets as float32.
struct SharedCases {
    ~SharedCases() {
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

    std::string name;
    int file = -1;
    void* mapping = MAP_FAILED;
    uint64_t size = 0;
    uint64_t targets_offset = 0;
};

std::unique_ptr<SharedCases> Share(const serving::Cases& cases, uint32_t epoch) {
    auto shared = std::make_unique<SharedCases>();
    shared->name = "/deepcircus_" + std::to_string(getpid()) + "_" + std::to_string(epoch);
    const uint64_t values_bytes = cases.values.size();
    shared->targets_offset = (values_bytes + alignof(float) - 1) / alignof(float) * alignof(float);
    shared->size = shared->targets_offset + cases.targets.size() * sizeof(float);

    shared->file = shm_open(shared->name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    assert(shared->file >= 0);
    const int truncate_result = ftruncate(shared->file, static_cast<off_t>(shared->size));
    assert(truncate_result == 0);
    shared->mapping = mmap(nullptr, shared->size, PROT_READ | PROT_WRITE, MAP_SHARED, shared->file, 0);
    assert(shared->mapping != MAP_FAILED);

    char* destination = static_cast<char*>(shared->mapping);
    std::memcpy(destination, cases.values.data(), values_bytes);
    std::memcpy(destination + shared->targets_offset, cases.targets.data(), cases.targets.size() * sizeof(float));
    return shared;
}

std::vector<char> Describe(const serving::Cases& cases, const SharedCases& shared) {
    std::vector<char> response;
    Append<uint32_t>(response, cases.cases);
    Append<uint64_t>(response, cases.columns);
    Append<uint64_t>(response, shared.size);
    AppendString(response, shared.name.substr(1));
    Append<uint64_t>(response, shared.targets_offset);
    Append<uint64_t>(response, cases.targets.size());
    return response;
}

std::vector<char> DescribeDataset(uint16_t bitness, const serving::Dataset& validation, const serving::Dataset& train) {
    std::vector<char> response;
    Append<uint16_t>(response, bitness);
    Append<uint16_t>(response, serving::PointDim(bitness));
    Append<uint32_t>(response, validation.CaseCount());
    Append<uint32_t>(response, validation.Entries());
    Append<uint32_t>(response, train.CaseCount());
    Append<uint32_t>(response, train.Entries());
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

ServingShape ReadServingShape(int client) {
    const auto command = static_cast<Command>(ReadValue<uint32_t>(client));
    const uint64_t payload_size = ReadValue<uint64_t>(client);
    assert(command == Command::Initialize);

    ServingShape shape{};
    const uint16_t model = ReadValue<uint16_t>(client);
    assert(model == 1 || model == 2);
    shape.model = static_cast<serving::Model>(model);
    shape.bitness = ReadValue<uint16_t>(client);
    shape.sampling.batches = ReadValue<uint16_t>(client);
    shape.sampling.points_in_batch = ReadValue<uint16_t>(client);
    shape.sampling.seed = ReadValue<uint64_t>(client);
    shape.data_dir = ReadString(client);

    const uint64_t expected_size = sizeof(uint16_t) * 4 + sizeof(uint64_t) + sizeof(uint32_t) + shape.data_dir.size();
    assert(payload_size == expected_size);
    return shape;
}

void ServeEpochs(int client, const ServingShape& shape) {
    const serving::Dataset validation(
        serving::FilePath(shape.data_dir, shape.model, shape.bitness, serving::Split::kValidation),
        serving::Split::kValidation, shape.sampling);
    const serving::Dataset train(serving::FilePath(shape.data_dir, shape.model, shape.bitness, serving::Split::kTrain),
                                 serving::Split::kTrain, shape.sampling);
    assert(validation.Bitness() == shape.bitness);
    assert(train.Bitness() == shape.bitness);
    assert(validation.CaseCount() > 0);
    assert(train.CaseCount() > 0);
    WriteResponse(client, DescribeDataset(shape.bitness, validation, train));

    std::unique_ptr<SharedCases> current;
    while (true) {
        // The client hangs up when it has trained its last epoch.
        uint32_t command_value = 0;
        if (!ReadExact(client, &command_value, sizeof(command_value))) {
            return;
        }
        const uint64_t payload_size = ReadValue<uint64_t>(client);
        assert(static_cast<Command>(command_value) == Command::Epoch);
        assert(payload_size == sizeof(uint32_t));
        const uint32_t epoch = ReadValue<uint32_t>(client);

        // The client copies a segment out before asking for the next one, so
        // the previous one has no reader left by now.
        current.reset();
        const serving::Cases cases = epoch == 0 ? validation.Sample(epoch) : train.Sample(epoch);
        current = Share(cases, epoch);
        WriteResponse(client, Describe(cases, *current));
    }
}
