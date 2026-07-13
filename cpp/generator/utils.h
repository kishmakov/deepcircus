#pragma once

#include <stddef.h>
#include <stdint.h>

#include <functional>
#include <random>
#include <string>
#include <string_view>
#include <vector>

std::mt19937 PrepRNG(uint16_t bitness, size_t case_id);

// SplitMix64 output finalizer over an already-advanced state.
uint64_t Mix64(uint64_t value);
uint64_t SplitMix64(uint64_t &state);
uint64_t TaskSeed(uint64_t seed, uint16_t bitness, uint64_t iteration);
uint64_t DomainSeed(uint64_t seed, uint64_t domain, uint16_t bitness);

// Deterministic, chunk-order-independent sample of `cases` distinct ids from
// [0, population).
std::vector<size_t> SampleCaseIds(size_t population, size_t cases, uint64_t seed);

// Generated +/-1 values stay bit-packed until a caller requests float output.
void StoreBits(std::vector<bool> &destination, const std::vector<float> &source);

class RandomBoolGenerator {
public:
    explicit RandomBoolGenerator(std::mt19937 rng);

    bool Generate();
    std::mt19937 &RNG();

private:
    std::mt19937 rng_;
    uint32_t bit_buffer_ = 0;
    uint8_t bits_remaining_ = 0;
};

size_t FullBitId(size_t bit_id, size_t fixed_id);

uint64_t CaseInputSeed(uint64_t seed, uint16_t bitness, size_t case_id);

class InputGenerator {
public:
    InputGenerator(uint16_t bitness, size_t reps, uint64_t seed);

    void StartSample();
    std::string_view Generate(size_t rep);

private:
    void FillRandom(std::string &output);

    uint16_t bitness_;
    size_t reps_;
    uint64_t state_;
    uint64_t bit_buffer_ = 0;
    uint8_t bits_remaining_ = 0;
    std::string base_input_;
    std::string input_;
    size_t next_rep_;
};

void FillGeneratedValueTensor(uint16_t bitness, size_t reps, uint64_t seed, float *out,
                              const std::function<bool(std::string_view)> &evaluate);

void FillGeneratedRestrictionsTensor(uint16_t bitness, size_t reps, uint64_t seed, float *out,
                                     const std::function<bool(std::string_view)> &evaluate);

class FlippingSampler {
public:
    FlippingSampler() = default;
    FlippingSampler(uint16_t bitness, std::string_view input);

    void Reset(uint16_t bitness, std::string_view input);
    std::string input;

    void Fill(std::string &value, size_t sample_offset, size_t fixed_bit_id,
              const std::function<bool(std::string_view)> &evaluate);

    void Fill(float *value, size_t sample_offset, size_t fixed_bit_id,
              const std::function<bool(std::string_view)> &evaluate);

private:
    uint16_t bitness_ = 0;
};

// One full flip sample as a 0/1 string of length 2 * bitness + 1.
std::string SampledValueString(uint16_t bitness, std::string_view input,
                               const std::function<bool(std::string_view)> &evaluate);
