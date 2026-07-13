#include "daemon.h"
#include "task_queue.h"

#include <unistd.h>

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <thread>

inline uint16_t ParsePort(const char *value) {
    const unsigned long port = std::strtoul(value, nullptr, 10);
    assert(port <= 65535);
    return static_cast<uint16_t>(port);
}

int main(int argc, char **argv) {
    uint16_t port = 7788;
    size_t workers = 16;
    assert(workers > 0);
    for (int index = 1; index < argc; index += 2) {
        assert(index + 1 < argc);
        if (std::strcmp(argv[index], "--port") == 0) {
            port = ParsePort(argv[index + 1]);
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
