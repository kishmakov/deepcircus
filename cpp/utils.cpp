#include "utils.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <utility>

namespace {

constexpr uint64_t kSplitMixIncrement = 0x9e3779b97f4a7c15ull;

uint64_t SplitMix64(uint64_t& state) {
    uint64_t value = (state += kSplitMixIncrement);
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31);
}

}  // namespace

size_t FullBitId(size_t bit_id, size_t fixed_id) {
    return bit_id < fixed_id ? bit_id : bit_id + 1;
}

float SignedBit(char bit) {
    assert(bit == '0' || bit == '1');
    return bit == '1' ? 1.0f : -1.0f;
}

std::mt19937 PrepRNG(uint16_t bitness, size_t case_id) {
    std::seed_seq seed{
        static_cast<uint32_t>(bitness),
        static_cast<uint32_t>(case_id),
    };
    return std::mt19937(seed);
}

RandomBoolGenerator::RandomBoolGenerator(std::mt19937 rng)
    : rng_(std::move(rng))
{
}

bool RandomBoolGenerator::Generate() {
    if (bits_remaining_ == 0) {
        bit_buffer_ = static_cast<uint32_t>(rng_());
        bits_remaining_ = 32;
    }

    const bool result = (bit_buffer_ & 1u) != 0;
    bit_buffer_ >>= 1;
    --bits_remaining_;
    return result;
}

std::mt19937& RandomBoolGenerator::RNG() {
    return rng_;
}

uint64_t CaseInputSeed(uint64_t seed, uint16_t bitness, size_t case_id) {
    uint64_t state = seed;
    state ^= static_cast<uint64_t>(bitness) << 48;
    state ^= static_cast<uint64_t>(case_id);
    return SplitMix64(state);
}

InputGenerator::InputGenerator(
    uint16_t bitness,
    size_t reps,
    uint64_t seed)
    : bitness_(bitness)
    , reps_(reps)
    , state_(seed)
    , input_(bitness, '0')
    , next_rep_(reps)
{
    assert(bitness_ > 0);
    assert(reps_ > 0);
    assert(reps_ % 2 == 0);
    base_input_.assign(bitness_, '0');
}

void InputGenerator::StartSample() {
    assert(next_rep_ == reps_);
    FillRandom(base_input_);
    next_rep_ = 0;
}

std::string_view InputGenerator::Generate(size_t rep) {
    assert(rep == next_rep_);
    ++next_rep_;

    const size_t block_reps = reps_ / 2;
    if (rep >= block_reps) {
        FillRandom(input_);
        return input_;
    }

    input_ = base_input_;
    size_t blocks = 0;
    for (size_t masks = block_reps - 1; masks > 0; masks >>= 1) {
        ++blocks;
    }

    size_t block_start = 0;
    for (size_t block_id = 0; block_id < blocks; ++block_id) {
        const size_t block_size =
            bitness_ / blocks + (block_id < bitness_ % blocks ? 1 : 0);
        if (((rep >> block_id) & 1) != 0) {
            for (size_t bit_id = block_start; bit_id < block_start + block_size; ++bit_id) {
                input_[bit_id] = input_[bit_id] == '1' ? '0' : '1';
            }
        }
        block_start += block_size;
    }
    return input_;
}

void InputGenerator::FillRandom(std::string& output) {
    size_t offset = 0;
    while (offset < output.size()) {
        if (bits_remaining_ == 0) {
            bit_buffer_ = SplitMix64(state_);
            bits_remaining_ = 64;
        }

        const size_t chunk_size = std::min<size_t>(
            output.size() - offset,
            bits_remaining_);
        uint64_t chunk = bit_buffer_;
        for (size_t bit = 0; bit < chunk_size; ++bit) {
            output[offset + bit] = (chunk & 1) != 0 ? '1' : '0';
            chunk >>= 1;
        }

        bit_buffer_ = chunk_size == 64 ? 0 : bit_buffer_ >> chunk_size;
        bits_remaining_ = static_cast<uint8_t>(bits_remaining_ - chunk_size);
        offset += chunk_size;
    }
}

void FillGeneratedValueTensor(
    uint16_t bitness,
    size_t reps,
    uint64_t seed,
    float* out,
    const std::function<bool(std::string_view)>& evaluate)
{
    assert(out != nullptr);

    const size_t sample_size = 2 * bitness + 1;
    InputGenerator inputs(bitness, reps, seed);
    inputs.StartSample();
    FlippingSampler sampler;
    for (size_t rep = 0; rep < reps; ++rep) {
        sampler.Reset(bitness, inputs.Generate(rep));
        sampler.Fill(
            out,
            rep * sample_size,
            bitness,
            evaluate);
    }
}

void FillGeneratedRestrictionsTensor(
    uint16_t bitness,
    size_t reps,
    uint64_t seed,
    float* out,
    const std::function<bool(std::string_view)>& evaluate)
{
    assert(out != nullptr);
    assert(bitness > 1);
    assert(reps > 0);
    assert(reps % 2 == 0);

    const size_t free_bits = bitness - 1;
    const size_t sample_size = 2 * free_bits + 1;
    const size_t restrictions = bitness * 2;
    std::string sample_input(bitness, '0');
    InputGenerator inputs(free_bits, reps, seed);
    FlippingSampler sampler;

    for (size_t fixed_bit_id = 0; fixed_bit_id < bitness; ++fixed_bit_id) {
        for (size_t fixed_bit_value = 0; fixed_bit_value <= 1; ++fixed_bit_value) {
            const size_t restriction_id = fixed_bit_id * 2 + fixed_bit_value;
            sample_input[fixed_bit_id] = static_cast<char>('0' + fixed_bit_value);
            inputs.StartSample();

            for (size_t rep = 0; rep < reps; ++rep) {
                const std::string_view free_input = inputs.Generate(rep);
                for (size_t coord = 0; coord < free_bits; ++coord) {
                    sample_input[FullBitId(coord, fixed_bit_id)] =
                        free_input[coord];
                }

                sampler.Reset(bitness, sample_input);
                sampler.Fill(
                    out,
                    (restriction_id * reps + rep) * sample_size,
                    fixed_bit_id,
                    evaluate);
            }
        }
    }
}

FlippingSampler::FlippingSampler(uint16_t bitness, std::string_view input) {
    Reset(bitness, input);
}

void FlippingSampler::Reset(uint16_t bitness, std::string_view input) {
    bitness_ = bitness;
    this->input.assign(input.data(), input.size());
}

void FlippingSampler::Fill(
    std::string& value,
    size_t sample_offset,
    size_t fixed_bit_id,
    const std::function<bool(std::string_view)>& evaluate)
{
    const size_t free_bits = fixed_bit_id < bitness_ ? bitness_ - 1 : bitness_;

    for (size_t coord = 0; coord < free_bits; ++coord) {
        value[sample_offset + coord] = input[FullBitId(coord, fixed_bit_id)];
    }
    value[sample_offset + free_bits] = evaluate({input.data(), bitness_}) ? '1' : '0';
    for (size_t coord = 0; coord < free_bits; ++coord) {
        char& bit = input[FullBitId(coord, fixed_bit_id)];
        bit = bit == '1' ? '0' : '1';
        value[sample_offset + free_bits + 1 + coord] =
            evaluate({input.data(), bitness_}) ? '1' : '0';
        bit = bit == '1' ? '0' : '1';
    }
}

void FlippingSampler::Fill(
    float* value,
    size_t sample_offset,
    size_t fixed_bit_id,
    const std::function<bool(std::string_view)>& evaluate)
{
    assert(value != nullptr);
    const size_t free_bits = fixed_bit_id < bitness_ ? bitness_ - 1 : bitness_;

    for (size_t coord = 0; coord < free_bits; ++coord) {
        value[sample_offset + coord] = SignedBit(input[FullBitId(coord, fixed_bit_id)]);
    }
    value[sample_offset + free_bits] =
        evaluate({input.data(), bitness_}) ? 1.0f : -1.0f;
    for (size_t coord = 0; coord < free_bits; ++coord) {
        char& bit = input[FullBitId(coord, fixed_bit_id)];
        bit = bit == '1' ? '0' : '1';
        value[sample_offset + free_bits + 1 + coord] =
            evaluate({input.data(), bitness_}) ? 1.0f : -1.0f;
        bit = bit == '1' ? '0' : '1';
    }
}
