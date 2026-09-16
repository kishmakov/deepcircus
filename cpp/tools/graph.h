#pragma once

#include <bitset>
#include <cstdint>
#include <memory>
#include <vector>

#include "func/func.h"

namespace tools {

// Layers own nodes; questions don't own.
struct Graph {
    using Bits = std::bitset<func::kMaxBitness>;

    struct Cell {
        Bits fixed;   // Queried axes.
        Bits values;  // Assignments; zero outside fixed.
        bool operator==(const Cell&) const = default;
    };

    struct GraphNode;

    struct NodeQuestion {
        uint16_t axis;
        GraphNode* zero;
        GraphNode* one;
    };

    struct GraphNode {
        uint32_t cell_id;
        std::vector<NodeQuestion> questions;
    };

    using GraphNodeUPtr = std::unique_ptr<GraphNode>;

    // Lays the cells out by the number of fixed axes and links every available split.
    Graph(uint16_t bitness, std::vector<Cell> cells);

    uint16_t bitness;
    // One restriction per vertex, including the root; helper coordinate is bitness.
    std::vector<Cell> cells;
    std::vector<std::vector<GraphNodeUPtr>> layers;
    GraphNode* root = nullptr;
};

// Minimum vertex count, including the root; complete the last random path.
Graph BuildGraph(uint16_t bitness, uint64_t seed, uint32_t cells_number);

// Packed primary inputs covering every cell's primary projection, low bit first.
// Asserts if points cannot hold the covering sample; extra rows may repeat.
std::vector<uint8_t> SampleGraphInputs(const Graph& graph, uint64_t seed, uint32_t points);

}  // namespace tools
