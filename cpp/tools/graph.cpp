#include "graph.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "tools/random.h"

namespace tools {

namespace {

constexpr uint64_t kGraphInputs = 0x67726170685f696eull;
constexpr uint64_t kGraphOrders = 0x67726170685f6f72ull;
constexpr uint64_t kGraphValues = 0x67726170685f7661ull;

struct CellHash {
    size_t operator()(const Graph::Cell& cell) const {
        return Mix64(std::hash<Graph::Bits>{}(cell.fixed)) ^ std::hash<Graph::Bits>{}(cell.values);
    }
};

using CellSet = std::unordered_set<Graph::Cell, CellHash>;
using Order = std::vector<uint16_t>;

Order GenerateOrder(uint16_t bitness, Random& random) {
    Order order(bitness + 1);
    std::iota(order.begin(), order.end(), 0);
    for (size_t remaining = order.size(); remaining > 1; --remaining) {
        std::swap(order[remaining - 1], order[random.Below(remaining)]);
    }
    return order;
}

void InsertAllOnes(uint16_t bitness, CellSet& cells) {
    for (uint16_t axis = 0; axis <= bitness; ++axis) {
        Graph::Cell cell;
        cell.fixed.set(axis);
        cells.insert(cell);
        cell.values.set(axis);
        cells.insert(cell);
    }
}

void InsertAllTwos(uint16_t bitness, CellSet& cells) {
    for (uint16_t first = 0; first <= bitness; ++first) {
        for (uint16_t second = first + 1; second <= bitness; ++second) {
            Graph::Cell cell;
            cell.fixed.set(first);
            cell.fixed.set(second);
            for (unsigned values = 0; values < 4; ++values) {
                cell.values.set(first, (values & 1) != 0);
                cell.values.set(second, (values & 2) != 0);
                cells.insert(cell);
            }
        }
    }
}

// Singles, pairs, and prefixes with up to two holes.
std::vector<Graph::Bits> GeneratePatterns(const Order& order) {
    std::unordered_set<Graph::Bits> patterns{Graph::Bits{}};

    for (size_t first = 0; first < order.size(); ++first) {
        Graph::Bits fixed;
        fixed.set(order[first]);
        patterns.insert(fixed);
        for (size_t second = first + 1; second < order.size(); ++second) {
            fixed.set(order[second]);
            patterns.insert(fixed);
            fixed.reset(order[second]);
        }
    }

    Graph::Bits prefix;
    for (size_t end = 0; end < order.size(); ++end) {
        prefix.set(order[end]);
        patterns.insert(prefix);
        for (size_t first = 0; first < end; ++first) {
            auto fixed = prefix;
            fixed.reset(order[first]);
            patterns.insert(fixed);
            for (size_t second = 0; second < first; ++second) {
                fixed.reset(order[second]);
                patterns.insert(fixed);
                fixed.set(order[second]);
            }
        }
    }

    return {patterns.begin(), patterns.end()};
}

// Build a path through selected axes, keeping both branches at each step.
CellSet BuildAssignedCells(Random& random, const Graph::Bits& pattern, const Order& order) {
    CellSet cells;
    Graph::Cell cell;
    for (uint16_t axis : order) {
        if (!pattern[axis]) continue;
        cell.fixed.set(axis);
        cells.insert(cell);
        cell.values.set(axis);
        cells.insert(cell);
        cell.values.set(axis, random.NextBool());
    }
    cells.insert(cell);  // The root, when nothing is fixed.
    return cells;
}

// Estimate new cells from path prefixes, before deduplication.
uint64_t CellsPerOrder(uint16_t bitness) {
    const uint16_t axes = bitness + 1;
    uint64_t cells = 0;
    for (uint16_t fixed = 3; fixed <= axes; ++fixed) {
        uint64_t patterns = 1;
        if (fixed < axes) patterns += fixed;
        if (fixed + 1 < axes) patterns += uint64_t{fixed} * (fixed + 1) / 2;
        cells += 2 * (fixed - 2) * patterns;
    }
    return cells;
}

Graph::Bits RandomInput(Random& random, uint16_t bitness) {
    Graph::Bits input;
    for (uint16_t axis = 0; axis < bitness; ++axis) input[axis] = random.NextBool();
    return input;
}

}  // namespace

Graph::Graph(uint16_t bitness, std::vector<Cell> cells) : bitness(bitness), cells(std::move(cells)) {
    layers.resize(this->bitness + 2);
    std::unordered_map<Cell, GraphNode*, CellHash> nodes;
    for (size_t index = 0; index < this->cells.size(); ++index) {
        const auto& cell = this->cells[index];
        auto node = std::make_unique<GraphNode>();
        node->cell_id = static_cast<uint32_t>(index);
        const bool inserted = nodes.emplace(cell, node.get()).second;
        assert(inserted);
        layers[cell.fixed.count()].push_back(std::move(node));
    }
    root = nodes.at(Cell{});
    for (const auto& layer : layers) {
        for (const auto& node : layer) {
            const auto& parent = this->cells[node->cell_id];
            for (uint16_t axis = 0; axis <= this->bitness; ++axis) {
                if (parent.fixed[axis]) continue;
                Cell child = parent;
                child.fixed.set(axis);
                const auto zero = nodes.find(child);
                child.values.set(axis);
                const auto one = nodes.find(child);
                if (zero == nodes.end() || one == nodes.end()) continue;
                node->questions.push_back({axis, zero->second, one->second});
            }
        }
    }
}

Graph BuildGraph(uint16_t bitness, uint64_t seed, uint32_t cells_number) {
    assert(bitness > 0 && bitness <= 255);
    assert(cells_number >= 1 + 2 * (uint32_t{bitness} + 1));
    assert(cells_number < UINT32_MAX);
    uint64_t maximum = 1;
    for (uint16_t axis = 0; axis <= bitness && maximum < cells_number; ++axis) maximum *= 3;
    assert(cells_number <= maximum);
    Random orders_random(DomainSeed(seed, kGraphOrders, bitness));
    Random values_random(DomainSeed(seed, kGraphValues, bitness));

    CellSet cells{Graph::Cell{}};
    InsertAllOnes(bitness, cells);
    InsertAllTwos(bitness, cells);

    const uint64_t per_order = CellsPerOrder(bitness);
    assert(per_order > 0);
    while (cells.size() < cells_number) {
        Order order = GenerateOrder(bitness, orders_random);
        for (const auto& pattern : GeneratePatterns(order)) {
            cells.merge(BuildAssignedCells(values_random, pattern, order));
        }
    }

    assert(cells.size() < UINT32_MAX);

    std::cerr << "Real size: " << cells.size() << "\n";

    std::vector<size_t> counts(bitness + 2);
    for (const auto& cell : cells) ++counts[cell.fixed.count()];
    for (size_t bits_assigned = 0; bits_assigned < counts.size(); ++bits_assigned) {
        std::cerr << bits_assigned << ": " << counts[bits_assigned] << '\n';
    }

    return Graph(bitness, {cells.begin(), cells.end()});
}

std::vector<uint8_t> SampleGraphInputs(const Graph& graph, uint64_t seed, uint32_t points) {
    assert(points > 0);
    Random random(DomainSeed(seed, kGraphInputs, graph.bitness));
    std::vector<Graph::Bits> inputs;
    inputs.reserve(points);
    // Cover the most restricted cells first; each input may cover many others.
    for (size_t depth = graph.layers.size(); depth-- > 0;) {
        for (const auto& node : graph.layers[depth]) {
            const auto& cell = graph.cells[node->cell_id];
            Graph::Bits fixed = cell.fixed;
            fixed.reset(graph.bitness);
            const Graph::Bits values = cell.values & fixed;
            const bool covered =
                std::any_of(inputs.begin(), inputs.end(), [&](const auto& input) { return (input & fixed) == values; });
            if (covered) continue;
            assert(inputs.size() < points && "points cannot cover the graph cells");
            inputs.push_back((RandomInput(random, graph.bitness) & ~fixed) | values);
        }
    }
    while (inputs.size() < points) inputs.push_back(RandomInput(random, graph.bitness));
    const size_t row_bytes = (graph.bitness + 7) / 8;
    std::vector<uint8_t> packed(size_t{points} * row_bytes, 0);
    for (size_t row = 0; row < inputs.size(); ++row) {
        for (uint16_t axis = 0; axis < graph.bitness; ++axis) {
            if (inputs[row][axis]) packed[row * row_bytes + axis / 8] |= 1u << (axis % 8);
        }
    }
    return packed;
}

}  // namespace tools
