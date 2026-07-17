#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "generator.h"

namespace {

std::filesystem::path CircuitsRoot() {
    return std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() / "data" / "circuits";
}

std::vector<std::string> ExpectedCircuitSets() {
    std::vector<std::string> sets;
    for (const auto& entry : std::filesystem::directory_iterator(CircuitsRoot())) {
        if (!entry.is_directory()) {
            continue;
        }
        for (const auto& file : std::filesystem::directory_iterator(entry.path())) {
            if (file.path().extension() == ".aig") {
                sets.push_back(entry.path().filename().string());
                break;
            }
        }
    }
    std::sort(sets.begin(), sets.end());
    return sets;
}

std::vector<std::string> ExpectedCircuitCases(const std::string& set_name) {
    std::vector<std::string> cases;
    for (const auto& file : std::filesystem::directory_iterator(CircuitsRoot() / set_name)) {
        if (file.path().extension() == ".aig") {
            cases.push_back(file.path().stem().string());
        }
    }
    std::sort(cases.begin(), cases.end());
    return cases;
}

struct AigMetadata {
    size_t inputs;
    size_t outputs;
};

// Reads the "aig M I L O A [B C J F]" header: inputs count primary inputs
// plus latches, outputs count primary outputs plus bad-state signals.
AigMetadata ReadAigMetadata(const std::string& set_name, const std::string& case_name) {
    std::ifstream file(CircuitsRoot() / set_name / (case_name + ".aig"), std::ios::binary);
    std::string header_line;
    EXPECT_TRUE(std::getline(file, header_line));

    std::istringstream header(header_line);
    std::string magic;
    size_t max_var = 0, inputs = 0, latches = 0, outputs = 0, ands = 0, bads = 0;
    header >> magic >> max_var >> inputs >> latches >> outputs >> ands;
    EXPECT_TRUE(header) << set_name << "/" << case_name;
    EXPECT_EQ(magic, "aig") << set_name << "/" << case_name;
    header >> bads;

    return {inputs + latches, outputs + bads};
}

}  // namespace

TEST(CircuitTest, Discovery) {
    const std::vector<std::string> sets = gen::CircuitSets();
    EXPECT_EQ(sets, ExpectedCircuitSets());

    for (const std::string& set_name : sets) {
        EXPECT_EQ(gen::CircuitCases(set_name), ExpectedCircuitCases(set_name)) << set_name;
    }
}

TEST(CircuitTest, Metadata) {
    for (const std::string& set_name : ExpectedCircuitSets()) {
        for (const std::string& case_name : ExpectedCircuitCases(set_name)) {
            const AigMetadata expected = ReadAigMetadata(set_name, case_name);
            EXPECT_EQ(gen::CircuitInputs(set_name, case_name), expected.inputs)
                << set_name << "/" << case_name;
            EXPECT_EQ(gen::CircuitOutputs(set_name, case_name), expected.outputs)
                << set_name << "/" << case_name;
        }
    }

    EXPECT_EQ(gen::CircuitInputs("iscas85", "c17"), 5u);
    EXPECT_EQ(gen::CircuitOutputs("iscas85", "c17"), 2u);

    EXPECT_EQ(gen::CircuitInputs("iscas87", "s27"), 7u);
    EXPECT_EQ(gen::CircuitOutputs("iscas87", "s27"), 1u);
}

TEST(CircuitTest, Value) {
    const std::string c17 = gen::CircuitValue("iscas85", "c17", "00000");
    EXPECT_EQ(c17.size(), 5u + 2u * 6u);
    EXPECT_EQ(c17, gen::CircuitValue("iscas85", "c17", "00000"));

    const std::string s27 = gen::CircuitValue("iscas87", "s27", "0000000");
    EXPECT_EQ(s27.size(), 7u + 1u * 8u);
    EXPECT_EQ(s27, gen::CircuitValue("iscas87", "s27", "0000000"));
}
