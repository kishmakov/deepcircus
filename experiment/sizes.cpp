#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "function.h"
#include "solver.h"

namespace {

void PrintFunction(const func::Function& function) {
    for (uint8_t level = 0; level < function.kDepth; ++level) {
        const func::OperationElement& element = function.OperationAt(level);
        std::cout << "    s" << int(element.output_id) << " = " << element.kName << "(";
        for (size_t id = 0; id < element.input_ids.size(); ++id) {
            std::cout << (id == 0 ? "" : ", ") << "s" << int(element.input_ids[id]);
        }
        std::cout << ")\n";
    }
    // `func::Function::Compute` returns the last slot.
    std::cout << "    output = s" << (function.kSlots - 1) << '\n';
}

// The function a level defines: `Compute` from that level over every assignment
// of the slots it takes from the outside.
std::vector<bool> LevelTruthTable(const func::Function& function, uint8_t level) {
    const size_t row_count = size_t{1} << function.InputCount(level);

    std::vector<bool> truth_table(row_count);
    for (size_t row = 0; row < row_count; ++row) {
        truth_table[row] = function.Compute(static_cast<func::FunctionInput>(row), level);
    }
    return truth_table;
}

}  // namespace

int main(int argc, char** argv) {
    assert(argc <= 3);

    const std::string bitness_str = argc >= 2 ? argv[1] : "12";
    const std::string seed_str = argc >= 3 ? argv[2] : "239";

    const size_t bitness = std::stoul(bitness_str);
    const uint64_t seed = std::stoull(seed_str);

    assert(func::kMinBitness <= bitness && bitness <= func::kMaxBitness);

    const func::FunctionBuilder builder(bitness, seed);
    const func::Function function = builder.Build();

    std::cout << "Random function over " << bitness << " inputs (seed " << seed << "), " << int(function.kDepth)
              << " gates:\n";
    PrintFunction(function);
    for (uint8_t level = 0; level < function.kDepth; ++level) {
        const uint16_t inputs = static_cast<uint16_t>(function.InputCount(level));
        assert(inputs <= tools::kMaxSolvableBitness);

        const std::vector<bool> truth_table = LevelTruthTable(function, level);

        int depth_delta = inputs - tools::SolveForDepth(inputs, truth_table);

        std::cout << "    level: " << std::setw(2) << int(level) << " inputs (dd): " << std::setw(2) << inputs
                  << " (" << std::setw(2) << depth_delta << ") min-size: " << std::setw(3)
                  << tools::SolveForSize(inputs, truth_table) << '\n';
    }

    return 0;
}
