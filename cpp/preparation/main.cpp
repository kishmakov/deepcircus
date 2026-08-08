// Producer for the offline training data that
// `scripts/preparation/prepare_offline_train_data.sh` stages into `data/`.
// Invoked as
//
//     offline_train_data_generator <output_dir> <bitness> <seed> <entries> \
//         <small_size_from> <small_size_to>
//
// it writes `s<series>_<bitness>_<source>.bin` into <output_dir> for both
// series and both sources -- four files per call. Every argument past the
// output directory comes from `conf/preparation.conf`.
// The bitness is zero-padded, so a plain listing of `data/` stays in bitness
// order instead of putting `s1_10_rand.bin` ahead of `s1_08_rand.bin`.

#include <cassert>
#include <cstdint>
#include <limits>
#include <string>

#include "offline/read_write.h"
#include "sampler.h"

namespace {

constexpr uint16_t kSeries[] = {1, 2};

// The file-name tag of a source and the sampler behind it; see `sampler.h` for
// what separates the two.
struct Source {
    const char* tag;
    offline::Entry (*entry)(const preparation::Parameters& parameters, uint16_t series, uint16_t bitness,
                            uint32_t index);
};

constexpr Source kSources[] = {
    {"rand", &preparation::RandomEntry},
    {"small", &preparation::SmallEntry},
};

std::string BitnessTag(uint16_t bitness) {
    assert(bitness < 100);
    const char digits[] = {char('0' + bitness / 10), char('0' + bitness % 10), '\0'};
    return digits;
}

void WriteSeries(const std::string& directory, const Source& source, uint16_t series, uint16_t bitness,
                 const preparation::Parameters& parameters, uint32_t entries) {
    const std::string path =
        directory + "/s" + std::to_string(series) + "_" + BitnessTag(bitness) + "_" + source.tag + ".bin";
    offline::Writer writer(path, entries, bitness);
    for (uint32_t index = 0; index < entries; ++index) {
        writer.Write(source.entry(parameters, series, bitness, index));
    }
}

uint16_t ParseSize(const char* argument) {
    const unsigned long value = std::stoul(argument);
    assert(value >= 1 && value <= std::numeric_limits<uint16_t>::max());
    return static_cast<uint16_t>(value);
}

}  // namespace

int main(int argc, char** argv) {
    assert(argc == 7);

    const std::string directory = argv[1];
    const unsigned long bitness_argument = std::stoul(argv[2]);
    const uint64_t seed = std::stoull(argv[3]);
    const unsigned long entries_argument = std::stoul(argv[4]);
    assert(bitness_argument <= std::numeric_limits<uint16_t>::max());
    assert(entries_argument <= std::numeric_limits<uint32_t>::max());
    const uint16_t bitness = bitness_argument;
    const uint32_t entries = entries_argument;
    assert(bitness >= offline::kMinBitness && bitness <= offline::kMaxBitness);

    const preparation::Parameters parameters{seed, ParseSize(argv[5]), ParseSize(argv[6])};
    assert(parameters.small_size_from <= parameters.small_size_to);

    for (const Source& source : kSources) {
        for (const uint16_t series : kSeries) {
            WriteSeries(directory, source, series, bitness, parameters, entries);
        }
    }

    return 0;
}
