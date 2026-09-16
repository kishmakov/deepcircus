#pragma once

#include <cstdint>
#include <vector>

#include "graph.h"

namespace tools {

struct DenseGraph {
    explicit DenseGraph(const Graph& graph);

    struct Layer {
        // Node p owns questions [offsets[p], offsets[p + 1]); size is nodes + 1.
        std::vector<uint32_t> offsets;
        // Paired child indices in the next layer; index 0 is the empty node.
        std::vector<uint32_t> zero;
        std::vector<uint32_t> one;
        // Restriction IDs in cells; UINT32_MAX for the empty node.
        std::vector<uint32_t> cell;
    };

    uint16_t bitness;
    std::vector<Graph::Cell> cells;
    // Layer k fixes k coordinates. Root: layer 0, node 1; node 0 is reserved.
    std::vector<Layer> layers;
};

}  // namespace tools
