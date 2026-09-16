#include "dense_graph.h"

#include <cassert>
#include <cstddef>
#include <unordered_map>

namespace tools {

namespace {

std::vector<DenseGraph::Layer> CompactGraph(const Graph& graph) {
    std::unordered_map<const Graph::GraphNode*, uint32_t> indices{{nullptr, 0}};
    for (const auto& nodes : graph.layers) {
        assert(nodes.size() < UINT32_MAX);
        for (size_t index = 0; index < nodes.size(); ++index) {
            indices.emplace(nodes[index].get(), static_cast<uint32_t>(index + 1));
        }
    }
    assert(!graph.layers.empty());
    assert(!graph.layers.front().empty() && graph.layers.front().front().get() == graph.root);
    std::vector<DenseGraph::Layer> result(graph.layers.size());
    for (size_t depth = 0; depth < graph.layers.size(); ++depth) {
        auto& layer = result[depth];
        layer.offsets = {0, 0};
        layer.cell = {UINT32_MAX};
        for (const auto& node : graph.layers[depth]) {
            assert(node->questions.size() <= UINT32_MAX - layer.zero.size());
            for (const auto& question : node->questions) {
                layer.zero.push_back(indices.at(question.zero));
                layer.one.push_back(indices.at(question.one));
            }
            layer.offsets.push_back(static_cast<uint32_t>(layer.zero.size()));
            layer.cell.push_back(node->cell_id);
        }
    }
    return result;
}

}  // namespace

DenseGraph::DenseGraph(const Graph& graph) : bitness(graph.bitness), cells(graph.cells), layers(CompactGraph(graph)) {}

}  // namespace tools
