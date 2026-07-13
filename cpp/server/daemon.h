#pragma once

#include "task_queue.h"

#include <cstdint>
#include <functional>
#include <memory>

// Binds a loopback listener on the port, prints the bound port,
// and returns the accepted client socket.
int AcceptClient(uint16_t port);

// Reads the Initialize command and its training-shape payload; the
// acknowledgement is sent by ServeTasks once the task source is ready.
TrainingShape ReadTrainingShape(int client);

// Acknowledges initialization, then serves Next/Release commands, publishing
// one coordinate at a time through POSIX shared memory. Returns when the
// client hangs up.
void ServeTasks(int client, const std::function<std::unique_ptr<TaskResult>()> &take);
