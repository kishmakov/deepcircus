#include "case.h"

#include "utils.h"

#include <bit>
#include <cassert>
#include <string>

namespace gen {

Case::Case(uint16_t bitness, size_t case_id) : bitness_(bitness), case_id_(case_id) {
    std::seed_seq seq{
        static_cast<uint32_t>(bitness),
        static_cast<uint32_t>(case_id),
        static_cast<uint32_t>(case_id >> 32),
    };
    rng_.seed(seq);
}

bool Case::GenerateBool() {
    if (bits_remaining_ == 0) {
        bit_buffer_ = static_cast<uint32_t>(rng_());
        bits_remaining_ = 32;
    }

    const bool result = (bit_buffer_ & 1u) != 0;
    bit_buffer_ >>= 1;
    --bits_remaining_;
    return result;
}

std::vector<bool> Case::GenerateSequence(uint16_t subtract) {
    assert(subtract < bitness_);
    const uint16_t length = bitness_ - subtract;

    std::vector<bool> sequence(length, false);

    for (uint16_t bit = 0; bit < length; ++bit) {
        sequence[bit] = GenerateBool();
    }

    return sequence;
}

std::vector<bool> Case::ComputeAt(const std::vector<bool>& input) const {
    assert(input.size() == bitness_);

    std::vector<bool> point = input;
    std::vector<bool> result(bitness_ + 1);
    result[0] = Evaluate(point);
    for (uint16_t bit = 0; bit < bitness_; ++bit) {
        point[bit] = !point[bit];
        result[bit + 1] = Evaluate(point);
        point[bit] = !point[bit];
    }
    return result;
}

std::vector<bool> Case::Sample(InputShape shape, uint16_t dims, const ComputeBlock& compute) {
    assert(shape.batches > 1);
    assert(std::has_single_bit(shape.batch_size));
    assert(dims > 0);

    const size_t sample_size = 2 * dims + 1;
    const size_t result_size = static_cast<size_t>(shape.batches) * shape.batch_size * sample_size;

    std::vector<bool> result;
    result.reserve(result_size);

    // random sampling

    std::vector<bool> sequence = GenerateSequence(bitness_ - dims);

    for (size_t id = 0; id < shape.batch_size; id++) {
        if (id > 0) {
            sequence = NextSequence(NextSequence(sequence));
        }

        const std::vector<bool> computed = compute(sequence);

        result.insert(result.end(), sequence.begin(), sequence.end());
        result.insert(result.end(), computed.begin(), computed.end());
    }

    // block inverting

    for (uint16_t batch = 1; batch < shape.batches; ++batch) {
        sequence = GenerateSequence(bitness_ - dims);

        const uint16_t groups = std::countr_zero(shape.batch_size);
        const std::vector<uint16_t> group_ids = SplitBitsInGroups(dims, groups, batch);

        for (uint16_t id = 0; id < shape.batch_size; ++id) {
            std::vector<bool> point = sequence;
            for (uint16_t bit = 0; bit < dims; ++bit) {
                if (((id >> group_ids[bit]) & 1u) != 0) {
                    point[bit] = !point[bit];
                }
            }

            const std::vector<bool> computed = compute(point);

            result.insert(result.end(), point.begin(), point.end());
            result.insert(result.end(), computed.begin(), computed.end());
        }
    }

    assert(result.size() == result_size);
    return result;
}

std::vector<bool> Case::SampleValues(InputShape shape) {
    return Sample(shape, bitness_, [this](const std::vector<bool>& point) { return ComputeAt(point); });
}

std::vector<bool> Case::SampleRestrictions(InputShape shape) {
    assert(bitness_ > 1);

    const uint16_t free_bits = bitness_ - 1;
    const size_t restriction_size = static_cast<size_t>(shape.batches) * shape.batch_size * (2 * free_bits + 1);

    std::vector<bool> result;
    result.reserve(size_t{2} * bitness_ * restriction_size);

    for (uint16_t fixed_bit_id = 0; fixed_bit_id < bitness_; ++fixed_bit_id) {
        for (uint16_t fixed_value = 0; fixed_value <= 1; ++fixed_value) {
            // Restricted function of the free bits: fix fixed_bit_id, then
            // evaluate at the free point and at each of its single-bit flips.
            const bool fixed = fixed_value != 0;
            const auto compute = [this, free_bits, fixed_bit_id, fixed](const std::vector<bool>& free_point) {
                std::vector<bool> input(bitness_);
                input[fixed_bit_id] = fixed;
                for (uint16_t coord = 0; coord < free_bits; ++coord) {
                    input[FullBitId(coord, fixed_bit_id)] = free_point[coord];
                }

                std::vector<bool> values(free_bits + 1);
                values[0] = Evaluate(input);
                for (uint16_t coord = 0; coord < free_bits; ++coord) {
                    const size_t bit_id = FullBitId(coord, fixed_bit_id);
                    input[bit_id] = !input[bit_id];
                    values[coord + 1] = Evaluate(input);
                    input[bit_id] = !input[bit_id];
                }
                return values;
            };

            const std::vector<bool> block = Sample(shape, free_bits, compute);
            assert(block.size() == restriction_size);
            result.insert(result.end(), block.begin(), block.end());
        }
    }

    return result;
}

std::string Case::SampledValueString(const std::vector<bool>& input) const {
    assert(input.size() == bitness_);

    const std::vector<bool> computed = ComputeAt(input);
    std::string value;
    value.reserve(2 * bitness_ + 1);
    for (bool bit : input) {
        value.push_back(bit ? '1' : '0');
    }
    for (bool bit : computed) {
        value.push_back(bit ? '1' : '0');
    }
    return value;
}

}  // namespace gen
