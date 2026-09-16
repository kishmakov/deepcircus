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
constexpr uint32_t kAssignmentsPerBitSquared = 4;

struct CellHash {
    size_t operator()(const Graph::Cell& cell) const {
        return Mix64(std::hash<Graph::Bits>{}(cell.fixed)) ^ std::hash<Graph::Bits>{}(cell.values);
    }
};

using CellSet = std::unordered_set<Graph::Cell, CellHash>;

void PickRandomAxesOrder(Random& random, std::vector<uint16_t>& order) {
    std::iota(order.begin(), order.end(), 0);
    for (size_t remaining = order.size(); remaining > 1; --remaining) {
        std::swap(order[remaining - 1], order[random.Below(remaining)]);
    }
}

std::vector<Graph::Bits> GeneratePatterns(const std::vector<uint16_t>& order) {
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
    std::vector<Graph::Bits> result(patterns.begin(), patterns.end());
    std::stable_partition(result.begin(), result.end(), [](const auto& fixed) { return fixed.count() <= 2; });
    return result;
}

std::vector<Graph::Cell> GenerateCells(uint16_t bitness, uint64_t seed, uint32_t cells_number) {
    Random orders_random(DomainSeed(seed, kGraphOrders, bitness));
    Random values_random(DomainSeed(seed, kGraphValues, bitness));

    CellSet cells;

    std::vector<uint16_t> order(bitness + 1);

    while (cells.size() < cells_number) {
        PickRandomAxesOrder(orders_random, order);
        const auto patterns = GeneratePatterns(order);
        for (const auto& fixed : patterns) {
            const bool exhaustive = fixed.count() <= 2;
            if (!exhaustive && cells.size() >= cells_number) break;
            const uint32_t assignments =
                exhaustive ? (1u << fixed.count()) : kAssignmentsPerBitSquared * bitness * bitness;
            for (uint32_t draw = 0; draw < assignments; ++draw) {
                if (!exhaustive && cells.size() >= cells_number) break;
                Graph::Cell cell;
                uint16_t assigned = 0;
                for (uint16_t axis : order) {
                    if (!fixed[axis]) continue;
                    cell.fixed.set(axis);
                    cells.insert(cell);
                    cell.values.set(axis);
                    cells.insert(cell);
                    const bool value = exhaustive ? ((draw >> assigned) & 1u) != 0 : values_random.NextBool();
                    cell.values.set(axis, value);
                    ++assigned;
                }
                cells.insert(cell);
            }
        }
    }
    assert(cells.size() < UINT32_MAX);
    std::vector<size_t> counts(bitness + 2);
    for (const auto& cell : cells) ++counts[cell.fixed.count()];
    // for (size_t bits_assigned = 0; bits_assigned < counts.size(); ++bits_assigned) {
    //     std::cerr << bits_assigned << ": " << counts[bits_assigned] << '\n';
    // }
    return {cells.begin(), cells.end()};
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
    assert(cells_number <= UINT32_MAX - 2 * (uint32_t{bitness} + 1));
    uint64_t maximum = 1;
    for (uint16_t axis = 0; axis <= bitness && maximum < cells_number; ++axis) maximum *= 3;
    assert(cells_number <= maximum); // is requested number of vertices possible?

    return Graph(bitness, GenerateCells(bitness, seed, cells_number));
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
