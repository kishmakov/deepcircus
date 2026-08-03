#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace func {

namespace op {

typedef uint8_t OperationInput;

struct Operation {
    static constexpr uint8_t kMaxArity = 6;  // 2^6 == 64

    Operation(const char* name, uint8_t arity, uint64_t table) : kName(name), kArity(arity), table_(table) {
        assert(arity <= kMaxArity);

        for (OperationInput input = 0; input < (OperationInput{1} << kArity); ++input) {
            OperationInput& preimage = preimages_[(*this)(input)];
            preimage = preimage == kNoInput ? input : preimage;
        }
        // Every operation here is non-constant, so it takes both values.
        assert(preimages_[0] != kNoInput && preimages_[1] != kNoInput);
    }

    bool operator()(OperationInput input) const {
        assert(input < (OperationInput{1} << kArity));
        return (table_ >> input) & 1;
    }

    // Smallest input the operation takes to `value`, so inverting is deterministic.
    OperationInput Preimage(bool value) const { return preimages_[value]; }

    const std::string kName;
    uint8_t kArity;

private:
    // Out of the input range of any operation, so it stands for "not seen yet".
    static constexpr OperationInput kNoInput = OperationInput{1} << kMaxArity;

    uint64_t table_ = 0;
    OperationInput preimages_[2] = {kNoInput, kNoInput};
};

extern std::vector<Operation> kOperations;

const Operation& OperationAt(size_t operation_id);

}  // namespace op

typedef uint16_t SchemeInput;
typedef std::vector<bool> Slots;
typedef std::vector<uint8_t> SlotIds;

struct OperationElement : op::Operation {
    std::vector<uint8_t> input_ids;
    uint8_t output_id;

    bool operator()(const Slots& slots) const;
};

// A single-input function has no gates, and `Scheme` needs at least one level.
constexpr size_t kMinBitness = 2;
constexpr size_t kMaxBitness = 12;

// A boolean function grown one operation at a time: starting from `bitness`
// unbound input slots, each operation binds unbound slots and leaves one new
// unbound slot in their place, so growing ends with one unbound slot, the
// output. A slot id is also its evaluation order: inputs first, then level
// `level` writes slot `bitness + level`.
class Scheme {
public:
    explicit Scheme(size_t bitness);

    // Binds `input_ids` -- unbound and increasing, since a symmetric operation
    // would otherwise have a second spelling -- and appends the slot the
    // operation writes, so both `depth` and `slots` grow by one.
    void AddOperation(const op::Operation& operation, const std::vector<size_t>& input_ids);

    // Whether growing is over: the scheme computes a single value.
    bool IsCompleted() const { return Unbound().size() == 1; }

    // The slot a completed scheme computes.
    size_t OutputId() const {
        assert(IsCompleted());
        return Unbound().front();
    }

    uint8_t depth = 0;
    uint8_t slots = 0;

    // Slot values after evaluation started at `level`: its live-in slots take
    // the bits of `input` (lowest bit first, ascending slot id), then
    // operations `level`..`depth` run in order. Untouched slots stay false.
    Slots ComputeAll(SchemeInput input, uint8_t level = 0) const;

    // The value of target slot for given input.
    bool ComputeValue(SchemeInput input) const { return ComputeAll(input)[OutputId()]; }

    // The value of unbound slots for given input, packed.
    size_t ComputeUnbound(SchemeInput input) const;

    // Slots evaluation started at `level` needs given; `level == depth` is `Unbound()`.
    const SlotIds& LiveIn(uint8_t level) const;

    // Number of slots evaluation started at `level` has to be given.
    size_t InputCount(uint8_t level) const { return LiveIn(level).size(); }

    // Slot ids no operation reads yet.
    const SlotIds& Unbound() const { return live_ins_.back(); }

    // The single operation applied at `level`.
    const OperationElement& OperationAt(uint8_t level) const;

private:
    // One entry per level plus the current unbound set: `live_ins_[level]` is what
    // was unbound just before operation `level` was added.
    std::vector<SlotIds> live_ins_;
    std::vector<OperationElement> operations_;
};

// Grows a random scheme until it is completed.
Scheme RandomScheme(size_t bitness, uint64_t seed);

// "MAJ(s0, s1, s4) -> s5".
std::ostream& operator<<(std::ostream& out, const OperationElement& element);

// One indented line per level, then the output slot.
std::ostream& operator<<(std::ostream& out, const Scheme& scheme);

}  // namespace func