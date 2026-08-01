#include "scheme.h"

#include <algorithm>
#include <limits>
#include <ostream>
#include <random>

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

Scheme::Scheme(size_t bitness) : slots(static_cast<uint8_t>(bitness)) {
    assert(kMinBitness <= bitness && bitness <= kMaxBitness);

    SlotIds unbound;
    for (uint8_t input_id = 0; input_id < bitness; ++input_id) {
        unbound.push_back(input_id);
    }
    live_ins_.push_back(std::move(unbound));
}

void Scheme::AddOperation(const op::Operation& operation, const std::vector<size_t>& input_ids) {
    assert(operation.kArity == input_ids.size());
    assert(slots < std::numeric_limits<uint8_t>::max());
    // Distinctness is what the `unbound` lookup below checks, so sorted here
    // means increasing.
    assert(std::is_sorted(input_ids.begin(), input_ids.end()));

    SlotIds unbound = Unbound();
    std::vector<uint8_t> slot_ids;
    for (size_t input_id : input_ids) {
        const auto bound = std::find(unbound.begin(), unbound.end(), static_cast<uint8_t>(input_id));
        assert(bound != unbound.end());
        unbound.erase(bound);
        slot_ids.push_back(static_cast<uint8_t>(input_id));
    }

    operations_.push_back({operation, std::move(slot_ids), slots});
    // The written slot is the largest one, so `unbound` stays ascending.
    unbound.push_back(slots);
    live_ins_.push_back(std::move(unbound));

    ++slots;
    ++depth;
}

Slots Scheme::Compute(SchemeInput input, uint8_t level) const {
    const SlotIds& live_in = LiveIn(level);
    assert(live_in.size() <= std::numeric_limits<SchemeInput>::digits);

    Slots values(slots);

    for (uint8_t slot : live_in) {
        values[slot] = (input & 1) != 0;
        input >>= 1;
    }

    for (uint8_t lev_id = level; lev_id < depth; ++lev_id) {
        const OperationElement& element = operations_[lev_id];
        values[element.output_id] = element(values);
    }

    return values;
}

const SlotIds& Scheme::LiveIn(uint8_t level) const {
    assert(level <= depth);
    return live_ins_[level];
}

const OperationElement& Scheme::OperationAt(uint8_t level) const {
    assert(level < depth);
    return operations_[level];
}

namespace {

// Picks a random operation and walks on until there is no arity conflict.
size_t PickOperation(std::mt19937_64& rng, size_t unbound_count) {
    assert(unbound_count > 1);
    std::uniform_int_distribution<size_t> pick(0, op::kOperations.size() - 1);
    size_t operation_id = pick(rng);
    while (op::OperationAt(operation_id).kArity > unbound_count) {
        operation_id = (operation_id + 1) % op::kOperations.size();
    }
    return operation_id;
}

// Picks `arity` distinct slots, increasing as `AddOperation` wants them;
// `unbound` is taken by value to draw from.
std::vector<size_t> PickInputs(std::mt19937_64& rng, SlotIds unbound, size_t arity) {
    assert(arity <= unbound.size());

    std::vector<size_t> input_ids;
    while (input_ids.size() < arity) {
        std::uniform_int_distribution<size_t> pick(0, unbound.size() - 1);
        const size_t slot = pick(rng);
        input_ids.push_back(unbound[slot]);
        unbound[slot] = unbound.back();
        unbound.pop_back();
    }

    std::sort(input_ids.begin(), input_ids.end());
    return input_ids;
}

}  // namespace

Scheme RandomScheme(size_t bitness, uint64_t seed) {
    std::mt19937_64 rng(seed);
    Scheme scheme(bitness);

    while (!scheme.IsCompleted()) {
        size_t operation_id = PickOperation(rng, scheme.Unbound().size());
        const op::Operation& operation = op::OperationAt(operation_id);
        const std::vector<size_t>& inputs = PickInputs(rng, scheme.Unbound(), operation.kArity);
        scheme.AddOperation(operation, inputs);
    }

    return scheme;
}

std::ostream& operator<<(std::ostream& out, const OperationElement& element) {
    out << element.kName << "(";
    for (size_t id = 0; id < element.input_ids.size(); ++id) {
        out << (id == 0 ? "" : ", ") << "s" << int(element.input_ids[id]);
    }
    return out << ") -> s" << int(element.output_id);
}

std::ostream& operator<<(std::ostream& out, const Scheme& scheme) {
    for (uint8_t level = 0; level < scheme.depth; ++level) {
        out << "    " << scheme.OperationAt(level) << '\n';
    }
    return out << "    output = s" << scheme.OutputId() << '\n';
}

}  // namespace func
