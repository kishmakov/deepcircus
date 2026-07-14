#include <unistd.h>

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <thread>

#include "daemon.h"
#include "task_queue.h"

inline uint16_t ParsePort(const char* value) {
    const unsigned long port = std::strtoul(value, nullptr, 10);
    assert(port <= 65535);
    return static_cast<uint16_t>(port);
}

int main(int argc, char** argv) {
    uint16_t port = 7788;
    size_t workers = 16;
    for (int index = 1; index < argc; index += 2) {
        assert(index + 1 < argc);
        if (std::strcmp(argv[index], "--port") == 0) {
            port = ParsePort(argv[index + 1]);
        } else if (std::strcmp(argv[index], "--workers") == 0) {
            workers = std::strtoull(argv[index + 1], nullptr, 10);
            assert(workers > 0);
        } else {
            assert(false);
        }
    }

    const int client = AcceptClient(port);
    TrainingShape shape = ReadTrainingShape(client);
    TaskQueue tasks(shape, workers);
    ServeTasks(client, [&tasks] { return tasks.Take(); });
    const int client_close_result = close(client);
    assert(client_close_result == 0);
    return 0;
}
