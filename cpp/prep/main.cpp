// Producer for the offline data staged into `data/` by
// `scripts/prep/generate_train_data.sh`. Invoked as
//
//     data_generator <output_dir> <m1|m2> <bitness> <seed> \
//         <train_tt> <train_general> <val_tt>

#include <cassert>
#include <cstdint>
#include <limits>
#include <string>

#include "offline/read_write.h"
#include "sampler.h"
#include "tools/random.h"

namespace {

constexpr uint64_t kTrainTTDomain = 0x747261696e5f736full;
constexpr uint64_t kTrainGeneralDomain = 0x747261696e5f756eull;
constexpr uint64_t kValidationDomain = 0x76616c5f736f6c76ull;

std::string BitnessTag(uint16_t bitness) {
    assert(bitness <= offline::kMaxBitness);
    return bitness < 10 ? "0" + std::to_string(bitness) : std::to_string(bitness);
}

prep::Model ParseModel(const char* argument) {
    const std::string value(argument);
    if (value == "m1") return prep::Model::kM1;
    assert(value == "m2");
    return prep::Model::kM2;
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

uint32_t TotalEntries(uint32_t tt, uint32_t general) {
    assert(uint64_t{tt} + general <= std::numeric_limits<uint32_t>::max());
    return tt + general;
}

void WriteFile(const std::string& path, prep::Model model, uint16_t bitness, uint64_t seed, uint32_t tt,
               uint32_t general) {
    offline::Writer writer(path, TotalEntries(tt, general), bitness);

    const prep::Parameters tt_parameters{tools::DomainSeed(seed, kTrainTTDomain, bitness)};
    for (uint32_t index = 0; index < tt; ++index) {
        writer.Write(prep::TTEntry(tt_parameters, model, bitness, index));
    }

    const prep::Parameters general_parameters{tools::DomainSeed(seed, kTrainGeneralDomain, bitness)};
    for (uint32_t index = 0; index < general; ++index) {
        writer.Write(prep::GeneralEntry(general_parameters, model, bitness, index));
    }
}

void WriteValidation(const std::string& path, prep::Model model, uint16_t bitness, uint64_t seed, uint32_t entries) {
    offline::Writer writer(path, entries, bitness);
    const prep::Parameters parameters{tools::DomainSeed(seed, kValidationDomain, bitness)};
    for (uint32_t index = 0; index < entries; ++index) {
        writer.Write(prep::TTEntry(parameters, model, bitness, index));
    }
}

}  // namespace

int main(int argc, char** argv) {
    assert(argc == 8);

    const std::string directory = argv[1];
    const std::string model_name = argv[2];
    const prep::Model model = ParseModel(argv[2]);
    const uint16_t bitness = ParseBitness(argv[3]);
    const uint64_t seed = std::stoull(argv[4]);
    const uint32_t train_tt = ParseEntries(argv[5]);
    const uint32_t train_general = ParseEntries(argv[6]);
    const uint32_t val_tt = ParseEntries(argv[7]);

    const std::string prefix = directory + "/" + model_name + "_" + BitnessTag(bitness);
    WriteFile(prefix + ".train", model, bitness, seed, train_tt, train_general);
    WriteValidation(prefix + ".val", model, bitness, seed, val_tt);
    return 0;
}
