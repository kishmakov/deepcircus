// Producer for the offline training data that
// `scripts/preparation/prepare_offline_train_data.sh` stages into `data/`.
// Invoked as
//
//     offline_train_data_generator <output_dir> <bitness> <seed> <entries>
//
// it writes `s1_<bitness>.bin` and `s2_<bitness>.bin` into <output_dir>.
// The bitness is zero-padded, so a plain listing of `data/` stays in bitness
// order instead of putting `s1_10.bin` ahead of `s1_8.bin`.
//
// The payload is still a placeholder: deterministic filler keyed by
// `(seed, series, bitness)` so the staging script can be exercised end to end.
// Only `SeriesEntry` changes when the real solver-backed content lands; the
// file layout and the CLI are meant to stay.

#include <cassert>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "offline_data.h"

namespace {

constexpr uint16_t kSeries[] = {1, 2};

// SplitMix64 finalizer, same mixer the generator uses to key randomness off a
// case's coordinates.
uint64_t Mix(uint64_t value) {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

offline::Entry SeriesEntry(uint64_t seed, uint16_t series, uint16_t bitness, uint32_t index) {
    uint64_t state = Mix(seed);
    state = Mix(state ^ series);
    state = Mix(state ^ bitness);
    state = Mix(state ^ index);

    const size_t table_bytes = offline::TableBytes(bitness);
    std::vector<uint8_t> g(table_bytes);
    std::vector<uint8_t> fx(table_bytes);
    for (std::vector<uint8_t>* table : {&g, &fx}) {
        for (uint8_t& byte : *table) {
            state = Mix(state);
            byte = static_cast<uint8_t>(state);
        }
    }

    state = Mix(state);
    const uint8_t min_depth = static_cast<uint8_t>(state % (bitness + 1));
    state = Mix(state);
    const uint16_t min_size = static_cast<uint16_t>(state % (uint32_t{1} << bitness));
    return offline::Entry{std::move(g), std::move(fx), min_depth, min_size};
}

std::string BitnessTag(uint16_t bitness) {
    assert(bitness < 100);
    const char digits[] = {char('0' + bitness / 10), char('0' + bitness % 10), '\0'};
    return digits;
}

void WriteSeries(const std::string& directory, uint16_t series, uint16_t bitness, uint64_t seed, uint32_t entries) {
    const std::string path = directory + "/s" + std::to_string(series) + "_" + BitnessTag(bitness) + ".bin";
    offline::Writer writer(path, entries, bitness);
    for (uint32_t index = 0; index < entries; ++index) {
        writer.Write(SeriesEntry(seed, series, bitness, index));
    }
}

}  // namespace

int main(int argc, char** argv) {
    assert(argc == 5);

    const std::string directory = argv[1];
    const unsigned long bitness_argument = std::stoul(argv[2]);
    const uint64_t seed = std::stoull(argv[3]);
    const unsigned long entries_argument = std::stoul(argv[4]);
    assert(bitness_argument <= std::numeric_limits<uint16_t>::max());
    assert(entries_argument <= std::numeric_limits<uint32_t>::max());
    const uint16_t bitness = bitness_argument;
    const uint32_t entries = entries_argument;
    assert(bitness >= offline::kMinBitness && bitness <= offline::kMaxBitness);

    for (const uint16_t series : kSeries) {
        WriteSeries(directory, series, bitness, seed, entries);
    }

    return 0;
}
