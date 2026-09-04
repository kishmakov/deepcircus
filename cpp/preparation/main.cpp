// Producer for the offline data staged into `data/` by
// `scripts/prep/generate_train_data.sh`. Invoked as
//
//     data_generator <output_dir> <m1|m2> <bitness> <seed> \
//         <train_solved> <train_unsolved> <val_solved>

#include <cassert>
#include <cstdint>
#include <limits>
#include <string>

#include "offline/read_write.h"
#include "sampler.h"
#include "tools/random.h"
#include "tools/solver.h"

namespace {

constexpr uint64_t kTrainSolvedDomain = 0x747261696e5f736full;
constexpr uint64_t kTrainUnsolvedDomain = 0x747261696e5f756eull;
constexpr uint64_t kValidationDomain = 0x76616c5f736f6c76ull;

std::string BitnessTag(uint16_t bitness) {
    assert(bitness <= offline::kMaxBitness);
    return bitness < 10 ? "0" + std::to_string(bitness) : std::to_string(bitness);
}

preparation::Model ParseModel(const char* argument) {
    const std::string value(argument);
    if (value == "m1") return preparation::Model::kM1;
    assert(value == "m2");
    return preparation::Model::kM2;
}

uint16_t ParseBitness(const char* argument) {
    const unsigned long value = std::stoul(argument);
    assert(value >= offline::kMinBitness && value <= offline::kMaxBitness);
    return static_cast<uint16_t>(value);
}

uint32_t ParseEntries(const char* argument) {
    const unsigned long value = std::stoul(argument);
    assert(value <= std::numeric_limits<uint32_t>::max());
    return static_cast<uint32_t>(value);
}

uint32_t TotalEntries(uint32_t solved, uint32_t unsolved) {
    assert(uint64_t{solved} + unsolved <= std::numeric_limits<uint32_t>::max());
    return solved + unsolved;
}

void WriteFile(const std::string& path, preparation::Model model, uint16_t bitness, uint64_t seed, uint32_t solved,
               uint32_t unsolved) {
    offline::Writer writer(path, TotalEntries(solved, unsolved), bitness);

    const preparation::Parameters solved_parameters{tools::DomainSeed(seed, kTrainSolvedDomain, bitness)};
    for (uint32_t index = 0; index < solved; ++index) {
        writer.Write(preparation::SolvedEntry(solved_parameters, model, bitness, index));
    }

    const preparation::Parameters unsolved_parameters{tools::DomainSeed(seed, kTrainUnsolvedDomain, bitness)};
    for (uint32_t index = 0; index < unsolved; ++index) {
        writer.Write(preparation::UnsolvedEntry(unsolved_parameters, model, bitness, index));
    }
}

void WriteValidation(const std::string& path, preparation::Model model, uint16_t bitness, uint64_t seed,
                     uint32_t entries) {
    offline::Writer writer(path, entries, bitness);
    const preparation::Parameters parameters{tools::DomainSeed(seed, kValidationDomain, bitness)};
    for (uint32_t index = 0; index < entries; ++index) {
        writer.Write(preparation::SolvedEntry(parameters, model, bitness, index));
    }
}

}  // namespace

int main(int argc, char** argv) {
    assert(argc == 8);

    const std::string directory = argv[1];
    const std::string model_name = argv[2];
    const preparation::Model model = ParseModel(argv[2]);
    const uint16_t bitness = ParseBitness(argv[3]);
    const uint64_t seed = std::stoull(argv[4]);
    const uint32_t train_solved = ParseEntries(argv[5]);
    const uint32_t train_unsolved = ParseEntries(argv[6]);
    const uint32_t val_solved = ParseEntries(argv[7]);
    if (bitness <= tools::kMaxSolvableBitness) assert(train_unsolved == 0);

    const std::string prefix = directory + "/" + model_name + "_" + BitnessTag(bitness);
    WriteFile(prefix + ".train", model, bitness, seed, train_solved, train_unsolved);
    WriteValidation(prefix + ".val", model, bitness, seed, val_solved);
    return 0;
}
