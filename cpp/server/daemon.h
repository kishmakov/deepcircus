#pragma once

// The daemon that feeds Python its training data. One client, accepted once:
// `src/daemon/client.py` starts it, says what to serve, and then asks for
// bootstrap reductions or epochs until it hangs up, at which point the daemon
// exits.
//
// Commands are a `uint32_t` id and a `uint64_t` payload size followed by the
// payload; each answer is a `uint32_t` status of 0 and its own size-prefixed
// payload.
//
//   Initialize | model, bitness, batches, points_in_batch, seed, data directory
//              -> bitness, point dim, and the case and entry counts of both files
//   Primary reductions | first unknown entry, number of entries
//              -> packed lower-arity rows for both values of every fixed input
//   Helper reductions | first unknown entry, number of entries
//              -> packed M_2 rows for the two values of M_1's helper
//   Targets    | two reconstructed float32 scores per unknown training entry
//   Epoch      | the epoch id, 0 being the validation file and anything above it
//              | the training one
//              -> the shared-memory segment that epoch's cases were written to
//
// One segment exists at a time: the next response carrying cases frees the one
// before it, which the client has copied out by then.

#include <stdint.h>

#include <string>

#include "dataset.h"

// What the client asked for, read off the Initialize command.
struct ServingShape {
    serving::Model model;
    uint16_t bitness;
    serving::SamplingShape sampling;
    std::string data_dir;
};

// Binds a loopback listener on `port`, prints the bound port so a client that
// asked for port 0 learns it, and returns the one accepted client.
int AcceptClient(uint16_t port);

// Reads the Initialize command; the answer is sent by ServeEpochs, once both
// files are open and their solved entries counted.
ServingShape ReadServingShape(int client);

// Answers Initialize, then serves one epoch per command until the client hangs
// up.
void ServeEpochs(int client, const ServingShape& shape);
