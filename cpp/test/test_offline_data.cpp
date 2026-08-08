#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "offline/read_write.h"

namespace {

offline::Entry MakeEntry(uint16_t bitness, uint8_t first, uint8_t second, uint8_t min_depth, uint16_t min_size) {
    return offline::Entry{std::vector<uint8_t>(offline::TableBytes(bitness), first),
                          std::vector<uint8_t>(offline::TableBytes(bitness), second), min_depth, min_size};
}

void ExpectEntry(const offline::Entry& actual, const offline::Entry& expected) {
    EXPECT_EQ(actual.g, expected.g);
    EXPECT_EQ(actual.fx, expected.fx);
    EXPECT_EQ(actual.min_depth, expected.min_depth);
    EXPECT_EQ(actual.min_size, expected.min_size);
}

}  // namespace

TEST(OfflineDataTest, WritesDocumentedLayout) {
    const std::string path = testing::TempDir() + "deepcircus_offline_data_layout.bin";
    const offline::Entry entry = MakeEntry(12, 0x12, 0x34, 7, 0x0a34);
    {
        offline::Writer writer(path, 1, 12);
        writer.Write(entry);
    }

    std::ifstream input(path, std::ios::binary);
    const std::vector<uint8_t> bytes{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    ASSERT_EQ(bytes.size(), 5 + offline::EntryBytes(12));
    EXPECT_EQ(std::vector<uint8_t>(bytes.begin(), bytes.begin() + 5), (std::vector<uint8_t>{1, 0, 0, 0, 12}));
    EXPECT_EQ(bytes[5], 0x12);
    EXPECT_EQ(bytes[5 + offline::TableBytes(12)], 0x34);
    EXPECT_EQ(bytes[5 + 2 * offline::TableBytes(12)], 7);
    EXPECT_EQ(bytes[5 + 2 * offline::TableBytes(12) + 1], 0x34);
    EXPECT_EQ(bytes[5 + 2 * offline::TableBytes(12) + 2], 0x0a);

    std::filesystem::remove(path);
}

TEST(OfflineDataTest, ReadsEntriesByIndex) {
    const std::string path = testing::TempDir() + "deepcircus_offline_data_reader.bin";
    const offline::Entry first = MakeEntry(8, 0xaa, 0xbb, 3, 17);
    const offline::Entry second = MakeEntry(8, 0xcc, 0xdd, 8, 255);
    {
        offline::Writer writer(path, 2, 8);
        writer.Write(first);
        writer.Write(second);
    }

    {
        offline::Reader reader(path);
        EXPECT_EQ(reader.Entries(), 2);
        EXPECT_EQ(reader.Bitness(), 8);
        ExpectEntry(reader.Read(1), second);
        ExpectEntry(reader.Read(0), first);
    }

    std::filesystem::remove(path);
}
