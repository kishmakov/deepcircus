#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
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

private:
    std::vector<SlotMask> masks_;
    std::vector<OperationElement> operations_;
};

}  // namespace func