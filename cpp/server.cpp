#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cassert>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace {

    enum class Command : uint32_t {
        Initialize = 1,
        Shutdown = 2,
    };

    struct TrainingShape {
        uint64_t first_iteration;
        uint64_t last_iteration;
        uint16_t bitness_from;
        uint16_t bitness_to;
        uint64_t seed;
    };

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

    void WriteResponse(int socket) {
        constexpr uint32_t status = 0;
        constexpr uint64_t payload_size = 0;
        const bool status_written = WriteExact(socket, &status, sizeof(status));
        assert(status_written);
        const bool size_written = WriteExact(socket, &payload_size, sizeof(payload_size));
        assert(size_written);
    }

    TrainingShape ReadInitialization(int socket, uint64_t payload_size) {
        constexpr uint64_t expected_size = sizeof(uint64_t) * 3 + sizeof(uint16_t) * 2;
        assert(payload_size == expected_size);

        TrainingShape shape{};
        shape.first_iteration = ReadValue<uint64_t>(socket);
        shape.last_iteration = ReadValue<uint64_t>(socket);
        shape.bitness_from = ReadValue<uint16_t>(socket);
        shape.bitness_to = ReadValue<uint16_t>(socket);
        shape.seed = ReadValue<uint64_t>(socket);

        assert(shape.first_iteration <= shape.last_iteration);
        assert(shape.bitness_from > 0);
        assert(shape.bitness_from <= shape.bitness_to);
        return shape;
    }

    void RunDaemon(int socket) {
        const Command command = static_cast<Command>(ReadValue<uint32_t>(socket));
        const uint64_t payload_size = ReadValue<uint64_t>(socket);
        assert(command == Command::Initialize);
        const TrainingShape shape = ReadInitialization(socket, payload_size);
        WriteResponse(socket);

        // Eager generation and shared-memory slot setup will start here.
        static_cast<void>(shape);

        const Command final_command = static_cast<Command>(ReadValue<uint32_t>(socket));
        const uint64_t final_payload_size = ReadValue<uint64_t>(socket);
        assert(final_command == Command::Shutdown);
        assert(final_payload_size == 0);
        WriteResponse(socket);
    }

    uint16_t ParsePort(const char *value) {
        const unsigned long port = std::strtoul(value, nullptr, 10);
        assert(port <= 65535);
        return static_cast<uint16_t>(port);
    }

} // namespace

int main(int argc, char **argv) {
    uint16_t port = 7788;
    for (int index = 1; index < argc; index += 2) {
        assert(index + 1 < argc);
        assert(std::strcmp(argv[index], "--port") == 0);
        port = ParsePort(argv[index + 1]);
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

    RunDaemon(client);
    const int client_close_result = close(client);
    assert(client_close_result == 0);
    const int listener_close_result = close(listener);
    assert(listener_close_result == 0);
    return 0;
}
