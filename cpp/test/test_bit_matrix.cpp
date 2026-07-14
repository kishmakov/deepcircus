#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "generator.h"

// The packed export is the daemon's publication format: each row padded to
// whole bytes, little-endian bit order, bit b standing for the value
// 2*b - 1. The byte layout is pinned here because the Python client decodes
// it independently.
TEST(BitMatrixTest, PackedLayoutIsRowPaddedLittleEndian) {
    const std::vector<std::vector<bool>> data = {
        {true, false, true, true, false, false, true, false, true, false},
        {false, true, false, false, true, true, false, true, false, true},
        {true, true, true, false, false, false, false, false, true, true},
    };
    const gen::Values matrix(data);
    EXPECT_EQ(matrix.RowBytes(), 2u);
    EXPECT_EQ(matrix.ByteCount(), 6u);

    std::vector<uint8_t> packed(matrix.ByteCount());
    matrix.WritePacked(packed.data());
    const std::vector<uint8_t> expected = {0x4d, 0x01, 0xb2, 0x02, 0x07, 0x03};
    EXPECT_EQ(packed, expected);
}

// UnpackRows is the client-side expansion of the packed publication format;
// packing and unpacking must round-trip to the values 2*b - 1.
TEST(BitMatrixTest, UnpackRowsRoundTripsWritePacked) {
    const std::vector<std::vector<bool>> data = {
        {true, false, true, true, false, false, true, false, true},
        {false, true, false, false, true, true, false, true, false},
        {true, true, false, true, false, true, false, false, true},
    };
    const gen::Values matrix(data);

    std::vector<uint8_t> packed(matrix.ByteCount());
    matrix.WritePacked(packed.data());
    std::vector<float> unpacked(matrix.ValueCount());
    gen::UnpackRows(packed.data(), matrix.Rows(), matrix.Columns(), unpacked.data());

    std::vector<float> expected;
    for (const std::vector<bool>& row : data) {
        for (bool value : row) {
            expected.push_back(value ? 1.0f : -1.0f);
        }
    }
    EXPECT_EQ(unpacked, expected);
}

TEST(BitMatrixTest, PackedRowsAreByteAligned) {
    const std::vector<std::vector<bool>> data = {
        {true, true, true, true, true, true, true, true},
        {true, false, false, false, false, false, false, false},
    };
    const gen::Values matrix(data);
    EXPECT_EQ(matrix.RowBytes(), 1u);

    std::vector<uint8_t> packed(matrix.ByteCount());
    matrix.WritePacked(packed.data());
    EXPECT_EQ(packed[0], 0xff);
    EXPECT_EQ(packed[1], 0x01);
}
