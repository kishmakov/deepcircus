#pragma once

#include <bitset>
#include <cstdint>
#include <memory>
#include <vector>

#include "func/func.h"

namespace tools {

using BitsSet = std::bitset<func::kMaxBitness>;

// Layers own nodes; questions don't own.
struct Graph {
    struct Cell {
        BitsSet fixed;   // Queried axes.
        BitsSet values;  // Assignments; zero outside fixed.
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

    Graph(uint16_t bitness, std::vector<Cell> cells);

    uint16_t bitness;
    // One restriction per vertex, including the root; helper coordinate is bitness.
    std::vector<Cell> cells;
    std::vector<std::vector<GraphNodeUPtr>> layers;
    GraphNode* root = nullptr;
};

/**
 * Generates a graph with guaranteed cells_number.
 */
Graph BuildGraph(uint16_t bitness, uint64_t seed, uint32_t cells_number);

// Distinct primary-input bitsets covering every cell's primary projection.
// Asserts if points cannot hold the covering sample or exceeds the cube.
std::vector<BitsSet> SampleGraphInputs(const Graph& graph, uint64_t seed, uint32_t points);

}  // namespace tools
