#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "function.h"
#include "solver.h"

namespace {

// A single-input function has no gates at all, and `func::Function` needs at
// least one level to be evaluated.
constexpr size_t kMinBitness = 2;
constexpr size_t kMaxBitness = 12;

const func::op::Operation& OperationAt(size_t operation_id) {
    assert(operation_id < func::op::kOperations.size());
    return func::op::kOperations[operation_id];
}

struct Gate {
    // Index into `func::op::kOperations`.
    size_t operation_id;
    // Signal ids the operation reads.
    std::vector<size_t> inputs;
    // Signals still unbound when the gate was added, in ascending order: what
    // evaluation started at this level has to be given from the outside, in the
    // order `func::Function::Compute` reads the input bits in.
    std::vector<size_t> live_in;
};

// Builds a random boolean function bottom-up: each operation consumes unbound
// signals (not yet consumed by any other) and produces one, so construction
// ends with a single unbound signal, the output.
//
// A signal id is also its `func::Function` slot: inputs come first, gate
// `level` writes slot `kBitness + level`, and the output is the last slot.
class FunctionBuilder {
public:
    FunctionBuilder(size_t bitness, uint64_t seed) : kBitness(bitness), rng_(seed) {
        assert(kMinBitness <= kBitness && kBitness <= kMaxBitness);

        for (size_t input_id = 0; input_id < kBitness; ++input_id) {
            unbound_.push_back(input_id);
        }

        while (unbound_.size() > 1) {
            const size_t operation_id = PickOperation();

            std::vector<size_t> live_in = unbound_;
            std::sort(live_in.begin(), live_in.end());

            std::vector<size_t> inputs = TakeInputs(OperationAt(operation_id).kArity);
            gates_.push_back({operation_id, std::move(inputs), std::move(live_in)});
            unbound_.push_back(kBitness + gates_.size() - 1);
        }
    }

    size_t GateCount() const { return gates_.size(); }

    const Gate& GateAt(size_t level) const {
        assert(level < gates_.size());
        return gates_[level];
    }

    size_t OutputId() const {
        assert(unbound_.size() == 1);
        return unbound_.front();
    }

    func::Function Build() const {
        const size_t slot_count = kBitness + gates_.size();
        assert(slot_count <= std::numeric_limits<uint8_t>::max());
        // `func::Function::Compute` returns the last slot.
        assert(OutputId() == slot_count - 1);

        std::vector<func::SlotMask> masks;
        std::vector<func::OperationElement> operations;
        for (size_t level = 0; level < gates_.size(); ++level) {
            const Gate& gate = gates_[level];
            assert(gate.live_in.size() <= std::numeric_limits<func::FunctionInput>::digits);

            func::SlotMask mask(slot_count, false);
            for (size_t slot : gate.live_in) {
                mask[slot] = true;
            }
            masks.push_back(std::move(mask));

            std::vector<uint8_t> input_ids;
            for (size_t input_id : gate.inputs) {
                input_ids.push_back(static_cast<uint8_t>(input_id));
            }
            operations.push_back(
                {OperationAt(gate.operation_id), std::move(input_ids), static_cast<uint8_t>(kBitness + level)});
        }

        return func::Function(static_cast<uint8_t>(slot_count), std::move(masks), std::move(operations));
    }

    const size_t kBitness;

private:
    // Takes a random operation and walks on until there is a arity conflict.
    size_t PickOperation() {
        assert(unbound_.size() > 1);
        std::uniform_int_distribution<size_t> pick(0, func::op::kOperations.size() - 1);
        size_t operation_id = pick(rng_);
        while (OperationAt(operation_id).kArity > unbound_.size()) {
            operation_id = (operation_id + 1) % func::op::kOperations.size();
        }
        return operation_id;
    }

    // Picks `arity` distinct unbound signals and binds them.
    std::vector<size_t> TakeInputs(size_t arity) {
        assert(arity <= unbound_.size());

        std::vector<size_t> input_ids;
        while (input_ids.size() < arity) {
            std::uniform_int_distribution<size_t> pick(0, unbound_.size() - 1);
            const size_t slot = pick(rng_);
            input_ids.push_back(unbound_[slot]);
            unbound_[slot] = unbound_.back();
            unbound_.pop_back();
        }
        return input_ids;
    }

    // One gate per level; gate `level` writes signal `kBitness + level`.
    std::vector<Gate> gates_;
    std::vector<size_t> unbound_;
    std::mt19937_64 rng_;
};

void PrintFunction(const FunctionBuilder& builder) {
    for (size_t level = 0; level < builder.GateCount(); ++level) {
        const Gate& gate = builder.GateAt(level);
        std::cout << "    s" << (builder.kBitness + level) << " = " << OperationAt(gate.operation_id).kName << "(";
        for (size_t id = 0; id < gate.inputs.size(); ++id) {
            std::cout << (id == 0 ? "" : ", ") << "s" << gate.inputs[id];
        }
        std::cout << ")\n";
    }
    std::cout << "    output = s" << builder.OutputId() << '\n';
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

    assert(kMinBitness <= bitness && bitness <= kMaxBitness);

    const FunctionBuilder builder(bitness, seed);
    const func::Function function = builder.Build();

    std::cout << "Random function over " << bitness << " inputs (seed " << seed << "), " << builder.GateCount()
              << " gates:\n";
    PrintFunction(builder);
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
