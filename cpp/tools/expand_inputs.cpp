// Thin stdin/stdout bridge over tools::ExpandInputs: expands per-batch base
// sequences with the same block-and-random walk the case sampling uses.
// Usage: expand_inputs <batches> <batch_size>; stdin carries one 0/1 line per
// base sequence, grouped `batches` lines per request; each request answers
// with one line of batches * batch_size * dims concatenated 0/1 points.

#include "sample.h"

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    assert(argc == 3);
    tools::InputShape shape{};
    shape.batches = static_cast<uint16_t>(std::stoul(argv[1]));
    shape.batch_size = static_cast<uint16_t>(std::stoul(argv[2]));

    std::string line;
    std::vector<std::vector<bool>> sequences;
    while (std::getline(std::cin, line)) {
        sequences.push_back(tools::BitsFromChars(line));
        if (sequences.size() < shape.batches) {
            continue;
        }

        const std::vector<bool> points = tools::ExpandInputs(shape, sequences);
        std::string output;
        output.reserve(points.size());
        for (const bool bit : points) {
            output.push_back(bit ? '1' : '0');
        }
        std::cout << output << '\n';
        sequences.clear();
    }
    assert(sequences.empty());
    return 0;
}
