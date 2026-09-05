// The `offline_server` daemon: see `daemon.h` for what it serves and how it is
// asked. Started by `src/daemon/client.py`, which passes `--port 0` and reads
// the bound port off the first line of its output.

#include <unistd.h>

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "daemon.h"

namespace {

uint16_t ParsePort(const char* value) {
    const unsigned long port = std::strtoul(value, nullptr, 10);
    assert(port <= 65535);
    return static_cast<uint16_t>(port);
}

}  // namespace

int main(int argc, char** argv) {
    uint16_t port = 0;
    for (int index = 1; index < argc; index += 2) {
        assert(index + 1 < argc);
        assert(std::strcmp(argv[index], "--port") == 0);
        port = ParsePort(argv[index + 1]);
    }

    const int client = server::AcceptClient(port);
    const server::ServingShape shape = server::ReadServingShape(client);
    server::ServeEpochs(client, shape);
    const int client_close_result = close(client);
    assert(client_close_result == 0);
    return 0;
}
