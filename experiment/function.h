#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace func {

namespace op {

typedef uint8_t OperationInput;

struct Operation {
    static constexpr uint8_t kMaxArity = 6;  // 2^6 == 64

    Operation(const char* name, uint8_t arity, uint64_t table) : kName(name), kArity(arity), table_(table) {
        assert(arity <= kMaxArity);
    }

    bool operator()(OperationInput input) const {
        assert(input < (OperationInput{1} << kArity));
        return (table_ >> input) & 1;
    }

    const std::string kName;
    uint8_t kArity;

private:
    uint64_t table_ = 0;
};

extern std::vector<Operation> kOperations;

const Operation& OperationAt(size_t operation_id);

}  // namespace op

typedef uint16_t FunctionInput;
typedef std::vector<bool> SlotMask;
typedef std::vector<bool> Slots;

struct OperationElement : op::Operation {
    std::vector<uint8_t> input_ids;
    uint8_t output_id;

    bool operator()(const Slots& slots) const;
};

struct Function {
    // `masks[level]` marks the slots level `level` has to be given from the
    // outside; `operations[level]` is the single operation applied at it.
    Function(uint8_t slots, std::vector<SlotMask> masks, std::vector<OperationElement> operations)
        : kDepth(operations.size()), kSlots(slots), masks_(std::move(masks)), operations_(std::move(operations)) {
        assert(masks_.size() == kDepth);
    }

    uint8_t kDepth;
    uint8_t kSlots;

    bool Compute(FunctionInput input, uint8_t level = 0) const;

    // Number of slots evaluation started at `level` has to be given.
    size_t InputCount(uint8_t level) const;

    // The single operation applied at `level`.
    const OperationElement& OperationAt(uint8_t level) const;

private:
    std::vector<SlotMask> masks_;
    std::vector<OperationElement> operations_;
};

// A single-input function has no gates at all, and `Function` needs at least one
// level to be evaluated.
constexpr size_t kMinBitness = 2;
constexpr size_t kMaxBitness = 12;

struct Gate {
    // Index into `op::kOperations`.
    size_t operation_id;
    // Signal ids the operation reads.
    std::vector<size_t> inputs;
    // Signals still unbound when the gate was added, in ascending order: what
    // evaluation started at this level has to be given from the outside, in the
    // order `Function::Compute` reads the input bits in.
    std::vector<size_t> live_in;
};

// Builds a random boolean function bottom-up: each operation consumes unbound
// signals (not yet consumed by any other) and produces one, so construction
// ends with a single unbound signal, the output.
//
// A signal id is also its `Function` slot: inputs come first, gate `level`
// writes slot `kBitness + level`, and the output is the last slot.
class FunctionBuilder {
public:
    FunctionBuilder(size_t bitness, uint64_t seed);

    size_t GateCount() const { return gates_.size(); }

    const Gate& GateAt(size_t level) const {
        assert(level < gates_.size());
        return gates_[level];
    }

    size_t OutputId() const {
        assert(unbound_.size() == 1);
        return unbound_.front();
    }

    Function Build() const;

    const size_t kBitness;

private:
    // Takes a random operation and walks on until there is a arity conflict.
    size_t PickOperation();

    // Picks `arity` distinct unbound signals and binds them.
    std::vector<size_t> TakeInputs(size_t arity);

    // One gate per level; gate `level` writes signal `kBitness + level`.
    std::vector<Gate> gates_;
    std::vector<size_t> unbound_;
    std::mt19937_64 rng_;
};

}  // namespace func