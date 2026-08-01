#include "function.h"

#include <algorithm>
#include <limits>

namespace func {

namespace op {

std::vector<Operation> kOperations = {
    {"NOT", 1, 0b01},    {"AND", 2, 0b1000}, {"OR", 2, 0b1110},      {"XOR", 2, 0b0110},
    {"NAND", 2, 0b0111}, {"NOR", 2, 0b0001}, {"MAJ", 3, 0b11101000},
};

const Operation& OperationAt(size_t operation_id) {
    assert(operation_id < kOperations.size());
    return kOperations[operation_id];
}

}  // namespace op

bool OperationElement::operator()(const Slots& slots) const {
    assert(kArity == input_ids.size());
    op::OperationInput input = 0;

    for (int id = 0; id < kArity; id++) {
        input |= static_cast<op::OperationInput>(slots[input_ids[id]]) << id;
    }

    return op::Operation::operator()(input);
}

bool Function::Compute(FunctionInput input, uint8_t level) const {
    assert(level < kDepth);

    std::vector<bool> slots(kSlots);

    auto mask = masks_[level];

    for (uint8_t id = 0; id < kSlots; id++) {
        if (mask[id]) {
            slots[id] = (input & 1) != 0;
            input >>= 1;
        }
    }

    for (uint8_t lev_id = level; lev_id < kDepth; ++lev_id) {
        const auto& element = operations_[lev_id];
        slots[element.output_id] = element(slots);
    }

    return slots[kSlots - 1];
}

size_t Function::InputCount(uint8_t level) const {
    assert(level < kDepth);

    size_t count = 0;
    for (bool needed : masks_[level]) {
        count += needed ? 1 : 0;
    }
    return count;
}

const OperationElement& Function::OperationAt(uint8_t level) const {
    assert(level < kDepth);
    return operations_[level];
}

FunctionBuilder::FunctionBuilder(size_t bitness, uint64_t seed) : kBitness(bitness), rng_(seed) {
    assert(kMinBitness <= kBitness && kBitness <= kMaxBitness);

    for (size_t input_id = 0; input_id < kBitness; ++input_id) {
        unbound_.push_back(input_id);
    }

    while (unbound_.size() > 1) {
        const size_t operation_id = PickOperation();

        std::vector<size_t> live_in = unbound_;
        std::sort(live_in.begin(), live_in.end());

        std::vector<size_t> inputs = TakeInputs(op::OperationAt(operation_id).kArity);
        gates_.push_back({operation_id, std::move(inputs), std::move(live_in)});
        unbound_.push_back(kBitness + gates_.size() - 1);
    }
}

Function FunctionBuilder::Build() const {
    const size_t slot_count = kBitness + gates_.size();
    assert(slot_count <= std::numeric_limits<uint8_t>::max());
    // `Function::Compute` returns the last slot.
    assert(OutputId() == slot_count - 1);

    std::vector<SlotMask> masks;
    std::vector<OperationElement> operations;
    for (size_t level = 0; level < gates_.size(); ++level) {
        const Gate& gate = gates_[level];
        assert(gate.live_in.size() <= std::numeric_limits<FunctionInput>::digits);

        SlotMask mask(slot_count, false);
        for (size_t slot : gate.live_in) {
            mask[slot] = true;
        }
        masks.push_back(std::move(mask));

        std::vector<uint8_t> input_ids;
        for (size_t input_id : gate.inputs) {
            input_ids.push_back(static_cast<uint8_t>(input_id));
        }
        operations.push_back(
            {op::OperationAt(gate.operation_id), std::move(input_ids), static_cast<uint8_t>(kBitness + level)});
    }

    return Function(static_cast<uint8_t>(slot_count), std::move(masks), std::move(operations));
}

size_t FunctionBuilder::PickOperation() {
    assert(unbound_.size() > 1);
    std::uniform_int_distribution<size_t> pick(0, op::kOperations.size() - 1);
    size_t operation_id = pick(rng_);
    while (op::OperationAt(operation_id).kArity > unbound_.size()) {
        operation_id = (operation_id + 1) % op::kOperations.size();
    }
    return operation_id;
}

std::vector<size_t> FunctionBuilder::TakeInputs(size_t arity) {
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

}  // namespace func