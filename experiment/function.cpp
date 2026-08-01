#include "function.h"

namespace func {

namespace op {

std::vector<Operation> kOperations = {
    {"NOT", 1, 0b01},    {"AND", 2, 0b1000}, {"OR", 2, 0b1110},      {"XOR", 2, 0b0110},
    {"NAND", 2, 0b0111}, {"NOR", 2, 0b0001}, {"MAJ", 3, 0b11101000},
};

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

}  // namespace func