#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <unordered_map>

#include "cli.h"
#include "graph.h"

namespace {

constexpr size_t kSeparatorWidth = 60;

// `#` rule carrying the number of axes the upcoming layer fixes.
void LayerSeparator(std::ostream& output, size_t depth) {
    std::string block(30, '#');
    output << block << " " << depth << " " << block << "\n";;
}

void WriteGraph(std::ostream& output, const tools::Graph& graph) {
    output << "Nodes: " << graph.cells.size() << std::endl;
    std::unordered_map<const tools::Graph::GraphNode*, std::string> labels{
        {graph.root, std::string(graph.bitness + 1, '*')}};
    for (size_t depth = 0; depth < graph.layers.size(); ++depth) {
        LayerSeparator(output, depth);

        for (const auto& node : graph.layers[depth]) {
            const std::string label = labels.at(node.get());
            output << label << ":\n";
            for (const auto& question : node->questions) {
                for (unsigned answer = 0; answer < 2; ++answer) {
                    const tools::Graph::GraphNode* child = answer ? question.one : question.zero;
                    if (child == nullptr) continue;
                    std::string child_label = label;
                    assert(child_label[question.axis] == '*');
                    child_label[question.axis] = '0' + answer;
                    const auto [position, inserted] = labels.emplace(child, child_label);
                    assert(inserted || position->second == child_label);
                    output << "  -" << question.axis << "-> " << child_label << '\n';
                }
            }
        }
    }
    assert(output.good());
}

}  // namespace

int main(int argc, char** argv) {
    assert(argc == 4 && "usage: graph_vis bitness seed cells_number");
    const uint16_t bitness = tools::Parse16(argv[1], 1, 255);
    const uint64_t seed = tools::Parse64(argv[2]);
    const uint32_t cells_number = tools::Parse32(argv[3]);
    const tools::Graph graph = tools::BuildGraph(bitness, seed, cells_number);
    WriteGraph(std::cout, graph);
}
