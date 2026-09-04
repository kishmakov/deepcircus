#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "offline/read_write.h"

namespace {

offline::Entry MakeEntry(offline::FunctionKind g_kind, std::vector<uint8_t> g, offline::FunctionKind f_kind,
                         std::vector<uint8_t> f, uint8_t min_depth, uint16_t min_size) {
    return offline::Entry{{g_kind, std::move(g)}, {f_kind, std::move(f)}, min_depth, min_size};
}

void ExpectEntry(const offline::Entry& actual, const offline::Entry& expected) {
    EXPECT_EQ(actual.g.kind, expected.g.kind);
    EXPECT_EQ(actual.g.payload, expected.g.payload);
    EXPECT_EQ(actual.f.kind, expected.f.kind);
    EXPECT_EQ(actual.f.payload, expected.f.payload);
    EXPECT_EQ(actual.min_depth, expected.min_depth);
    EXPECT_EQ(actual.min_size, expected.min_size);
}

}  // namespace

TEST(OfflineDataTest, WritesDocumentedLayout) {
    const std::string path = testing::TempDir() + "deepcircus_offline_data_layout.bin";
    const offline::Entry entry = MakeEntry(offline::FunctionKind::kTreeOverTable, {0xaa, 0xbb},
                                           offline::FunctionKind::kTable, {0x11, 0x22, 0x33}, 7, 0x0a34);
    {
        offline::Writer writer(path, 1, 12);
        writer.Write(entry);
    }

    std::ifstream input(path, std::ios::binary);
    const std::vector<uint8_t> bytes{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    const std::vector<uint8_t> expected{
        1,  0,    0,    0, 12,                    // count, bitness
        13, 0,    0,    0, 0,  0,    0,    0,     // first entry offset
        2,  2,    0,    0, 0,  0xaa, 0xbb,        // g
        0,  3,    0,    0, 0,  0x11, 0x22, 0x33,  // f
        7,  0x34, 0x0a,                           // target
    };
    EXPECT_EQ(bytes, expected);
    EXPECT_EQ(offline::EntryBytes(entry), 18u);

    std::filesystem::remove(path);
}

TEST(OfflineDataTest, ReadsVariableEntriesByIndex) {
    const std::string path = testing::TempDir() + "deepcircus_offline_data_reader.bin";
    const offline::Entry first =
        MakeEntry(offline::FunctionKind::kTree, {0xaa}, offline::FunctionKind::kTable, {0xbb, 0xcc}, 3, 17);
    const offline::Entry second =
        MakeEntry(offline::FunctionKind::kTable, {0x12, 0x34, 0x56}, offline::FunctionKind::kTable, {0x78},
                  offline::kUnknownDepth, offline::kUnknownSize);
    {
        offline::Writer writer(path, 2, 13);
        writer.Write(first);
        writer.Write(second);
    }

    {
        offline::Reader reader(path);
        EXPECT_EQ(reader.Entries(), 2);
        EXPECT_EQ(reader.Bitness(), 13);
        ExpectEntry(reader.Read(1), second);
        EXPECT_FALSE(reader.Read(1).TargetKnown());
        ExpectEntry(reader.Read(0), first);
        EXPECT_TRUE(reader.Read(0).TargetKnown());
    }

    std::filesystem::remove(path);
}

TEST(OfflineDataTest, WritesEmptyFile) {
    const std::string path = testing::TempDir() + "deepcircus_offline_data_empty.bin";
    { offline::Writer writer(path, 0, 19); }
    offline::Reader reader(path);
    EXPECT_EQ(reader.Entries(), 0);
    EXPECT_EQ(std::filesystem::file_size(path), 5u);
    std::filesystem::remove(path);
}
