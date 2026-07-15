#include "case.h"

namespace gen {

Case::Case(uint16_t bitness, size_t case_id, uint64_t seed) : bitness_(bitness), case_id_(case_id) {
    std::seed_seq seq{
        static_cast<uint32_t>(bitness),
        static_cast<uint32_t>(case_id),
        static_cast<uint32_t>(seed),
        static_cast<uint32_t>(seed >> 32),
    };
    rng_.seed(seq);
}

uint16_t Case::Bitness() const { return bitness_; }

size_t Case::CaseId() const { return case_id_; }

bool Case::Generate() {
    if (bits_remaining_ == 0) {
        bit_buffer_ = static_cast<uint32_t>(rng_());
        bits_remaining_ = 32;
    }

    const bool result = (bit_buffer_ & 1u) != 0;
    bit_buffer_ >>= 1;
    --bits_remaining_;
    return result;
}

std::mt19937& Case::RNG() { return rng_; }

}  // namespace gen
