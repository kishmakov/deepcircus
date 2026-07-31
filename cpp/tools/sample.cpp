#include "sample.h"

#include <bit>
#include <cassert>
#include <vector>

namespace tools {

std::vector<bool> BitsFromChars(std::string_view input) {
    std::vector<bool> bits(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        assert(input[i] == '0' || input[i] == '1');
        bits[i] = input[i] == '1';
    }
    return bits;
}

std::vector<uint16_t> SplitBitsInGroups(uint16_t bitness, uint16_t groups, uint16_t way) {
    assert(groups > 0);
    assert(groups <= bitness);

    const size_t shift = way % bitness;
    std::vector<uint16_t> group_ids(bitness);
    for (size_t bit_id = 0; bit_id < bitness; ++bit_id) {
        group_ids[(bit_id + shift) % bitness] = bit_id * groups / bitness;
    }
    return group_ids;
}

std::vector<bool> NextSequence(const std::vector<bool>& sequence) {
    // Reflected binary Gray-code successor: exactly one bit changes per step and
    // the walk cycles through all 2^n sequences (the last one wraps to all zeros).
    std::vector<bool> next = sequence;
    assert(!next.empty());

    size_t ones = 0;
    for (bool bit : next) {
        ones += bit ? 1 : 0;
    }

    if (ones % 2 == 0) {
        next[0] = !next[0];
        return next;
    }

    size_t lowest = 0;
    while (!next[lowest]) {
        ++lowest;
    }
    const size_t flip = lowest + 1 < next.size() ? lowest + 1 : lowest;
    next[flip] = !next[flip];
    return next;
}

std::vector<bool> GenerateSequence(uint16_t length, const BitSource& next_bit) {
    assert(length > 0);

    std::vector<bool> sequence(length, false);

    for (uint16_t bit = 0; bit < length; ++bit) {
        sequence[bit] = next_bit();
    }

    return sequence;
}

std::vector<bool> ExpandInputs(InputShape shape, const std::vector<std::vector<bool>>& sequences) {
    assert(shape.batches > 1);
    assert(std::has_single_bit(shape.batch_size));
    assert(sequences.size() == shape.batches);
    const uint16_t dims = static_cast<uint16_t>(sequences[0].size());
    assert(dims > 0);

    std::vector<bool> result;
    result.reserve(static_cast<size_t>(shape.batches) * shape.batch_size * dims);

    // random sampling

    std::vector<bool> sequence = sequences[0];

    for (size_t id = 0; id < shape.batch_size; id++) {
        if (id > 0) {
            sequence = NextSequence(NextSequence(sequence));
        }

        result.insert(result.end(), sequence.begin(), sequence.end());
    }

    // block inverting

    for (uint16_t batch = 1; batch < shape.batches; ++batch) {
        const std::vector<bool>& base = sequences[batch];
        assert(base.size() == dims);

        const uint16_t groups = std::countr_zero(shape.batch_size);
        const std::vector<uint16_t> group_ids = SplitBitsInGroups(dims, groups, batch);

        for (uint16_t id = 0; id < shape.batch_size; ++id) {
            std::vector<bool> point = base;
            for (uint16_t bit = 0; bit < dims; ++bit) {
                if (((id >> group_ids[bit]) & 1u) != 0) {
                    point[bit] = !point[bit];
                }
            }

            result.insert(result.end(), point.begin(), point.end());
        }
    }

    return result;
}

std::vector<bool> SampleInputs(InputShape shape, uint16_t dims, const BitSource& next_bit) {
    assert(dims > 0);

    std::vector<std::vector<bool>> sequences;
    sequences.reserve(shape.batches);
    for (uint16_t batch = 0; batch < shape.batches; ++batch) {
        sequences.push_back(GenerateSequence(dims, next_bit));
    }

    return ExpandInputs(shape, sequences);
}

}  // namespace tools
