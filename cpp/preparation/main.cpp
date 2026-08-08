// Producer for the offline training data that
// `scripts/preparation/prepare_offline_train_data.sh` stages into `data/`.
// Invoked as
//
//     offline_train_data_generator <output_dir> <bitness> <seed>
//
// it writes `s1_<bitness>.bin` and `s2_<bitness>.bin` into <output_dir>.
// The bitness is zero-padded, so a plain listing of `data/` stays in bitness
// order instead of putting `s1_10.bin` ahead of `s1_8.bin`. The seed reaches
// the file's header as well as its payload, so what produced a file can be read
// back off it -- the name carries no trace of the seed.
//
// The payload is still a placeholder: deterministic filler keyed by
// `(seed, series, bitness)` so the staging script can be exercised end to end.
// Only `SeriesRecord` changes when the real solver-backed content lands; the
// file layout and the CLI are meant to stay.

#include <cassert>
#include <cstdint>
#include <fstream>
#include <string>

namespace {

constexpr uint16_t kMinBitness = 8;
constexpr uint16_t kMaxBitness = 12;

// "CRCS" in little-endian byte order, so `xxd` on a produced file is readable.
constexpr uint32_t kMagic = 0x53435243;
constexpr uint16_t kSeries[] = {1, 2};
constexpr uint32_t kPlaceholderRecords = 16;

// SplitMix64 finalizer, same mixer the generator uses to key randomness off a
// case's coordinates.
uint64_t Mix(uint64_t value) {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

// Deterministic from `(seed, series, bitness, index)` alone -- nothing here may
// reach for a clock, a global RNG, or the filesystem.
uint64_t SeriesRecord(uint64_t seed, uint16_t series, uint16_t bitness, uint32_t index) {
    return Mix(seed ^ (uint64_t(series) << 48) ^ (uint64_t(bitness) << 32) ^ index);
}

template <typename T>
void Write(std::ofstream& out, T value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

std::string BitnessTag(uint16_t bitness) {
    assert(bitness < 100);
    const char digits[] = {char('0' + bitness / 10), char('0' + bitness % 10), '\0'};
    return digits;
}

void WriteSeries(const std::string& directory, uint16_t series, uint16_t bitness, uint64_t seed) {
    const std::string path = directory + "/s" + std::to_string(series) + "_" + BitnessTag(bitness) + ".bin";
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    assert(out.is_open());

    Write(out, kMagic);
    Write(out, series);
    Write(out, bitness);
    Write(out, kPlaceholderRecords);
    Write(out, seed);
    for (uint32_t index = 0; index < kPlaceholderRecords; ++index) {
        Write(out, SeriesRecord(seed, series, bitness, index));
    }

    out.flush();
    assert(out.good());
}

}  // namespace

int main(int argc, char** argv) {
    assert(argc == 4);

    const std::string directory = argv[1];
    const uint16_t bitness = static_cast<uint16_t>(std::stoul(argv[2]));
    const uint64_t seed = std::stoull(argv[3]);
    assert(bitness >= kMinBitness && bitness <= kMaxBitness);

    for (const uint16_t series : kSeries) {
        WriteSeries(directory, series, bitness, seed);
    }

    return 0;
}
