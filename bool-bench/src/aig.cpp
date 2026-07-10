#include "aig.h"

extern "C" {
#include <aiger.h>
}

#include <cassert>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

struct AigHeader {
    uint32_t max_var = 0;
    uint32_t inputs = 0;
    uint32_t latches = 0;
    uint32_t outputs = 0;
    uint32_t ands = 0;
    uint32_t bads = 0;
    uint32_t constraints = 0;
    uint32_t justice = 0;
    uint32_t fairness = 0;
};

struct CircuitMeta {
    std::filesystem::path path;
    uint32_t inputs = 0;
    uint32_t primary_inputs = 0;
    uint32_t latches = 0;
    uint32_t outputs = 0;
    uint32_t primary_outputs = 0;
    uint32_t bads = 0;
    uint32_t constraints = 0;
    uint32_t justice = 0;
    uint32_t fairness = 0;
    uint32_t ands = 0;
    uint32_t max_var = 0;
};

struct CircuitCatalog {
    std::map<std::string, std::map<std::string, CircuitMeta>> sets;
    std::string joined_sets;
    std::map<std::string, std::string> joined_cases;
};

struct AigCircuit {
    CircuitMeta meta;
    aiger* aig = nullptr;

    AigCircuit() = default;
    AigCircuit(const AigCircuit&) = delete;
    AigCircuit& operator=(const AigCircuit&) = delete;

    AigCircuit(AigCircuit&& other) noexcept
        : meta(std::move(other.meta))
        , aig(other.aig)
    {
        other.aig = nullptr;
    }

    AigCircuit& operator=(AigCircuit&& other) noexcept {
        if (this != &other) {
            if (aig != nullptr) {
                aiger_reset(aig);
            }
            meta = std::move(other.meta);
            aig = other.aig;
            other.aig = nullptr;
        }
        return *this;
    }

    ~AigCircuit() {
        if (aig != nullptr) {
            aiger_reset(aig);
        }
    }
};

std::string JoinNames(const std::vector<std::string>& names) {
    std::string joined;
    for (size_t i = 0; i < names.size(); ++i) {
        if (i != 0) {
            joined.push_back('\n');
        }
        joined += names[i];
    }
    return joined;
}

std::filesystem::path CircuitRoot() {
    const std::filesystem::path source_root = std::filesystem::path(__FILE__).parent_path().parent_path() / "circuits";
    if (std::filesystem::exists(source_root)) {
        return source_root;
    }

    for (std::filesystem::path dir = std::filesystem::current_path(); !dir.empty(); dir = dir.parent_path()) {
        const std::filesystem::path candidate = dir / "circuits";
        if (std::filesystem::exists(candidate)) {
            return candidate;
        }
        if (dir == dir.root_path()) {
            break;
        }
    }

    return "circuits";
}

AigHeader ReadAigHeader(std::istream& in) {
    std::string header_line;
    if (!std::getline(in, header_line)) {
        throw std::runtime_error("missing AIG header");
    }

    std::istringstream header(header_line);
    std::string magic;
    AigHeader parsed;
    header >> magic
           >> parsed.max_var
           >> parsed.inputs
           >> parsed.latches
           >> parsed.outputs
           >> parsed.ands;
    if (!header || magic != "aig") {
        throw std::runtime_error("unsupported AIG header");
    }
    header >> parsed.bads
           >> parsed.constraints
           >> parsed.justice
           >> parsed.fairness;

    return parsed;
}

CircuitMeta ReadCircuitMeta(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open AIG file");
    }

    const AigHeader header = ReadAigHeader(in);
    CircuitMeta meta;
    meta.path = path;
    meta.inputs = header.inputs + header.latches;
    meta.primary_inputs = header.inputs;
    meta.latches = header.latches;
    meta.outputs = header.outputs + header.bads;
    meta.primary_outputs = header.outputs;
    meta.bads = header.bads;
    meta.constraints = header.constraints;
    meta.justice = header.justice;
    meta.fairness = header.fairness;
    meta.ands = header.ands;
    meta.max_var = header.max_var;
    return meta;
}

CircuitCatalog BuildCircuitCatalog() {
    CircuitCatalog catalog;
    const std::filesystem::path root = CircuitRoot();
    if (!std::filesystem::exists(root)) {
        return catalog;
    }

    for (const auto& set_entry : std::filesystem::directory_iterator(root)) {
        if (!set_entry.is_directory()) {
            continue;
        }

        std::map<std::string, CircuitMeta> cases;
        for (const auto& case_entry : std::filesystem::directory_iterator(set_entry.path())) {
            if (!case_entry.is_regular_file() || case_entry.path().extension() != ".aig") {
                continue;
            }

            const std::string case_name = case_entry.path().stem().string();
            cases.emplace(case_name, ReadCircuitMeta(case_entry.path()));
        }

        if (!cases.empty()) {
            catalog.sets.emplace(set_entry.path().filename().string(), std::move(cases));
        }
    }

    std::vector<std::string> set_names;
    for (const auto& [set_name, cases] : catalog.sets) {
        set_names.push_back(set_name);

        std::vector<std::string> case_names;
        for (const auto& [case_name, meta] : cases) {
            case_names.push_back(case_name);
        }
        catalog.joined_cases.emplace(set_name, JoinNames(case_names));
    }
    catalog.joined_sets = JoinNames(set_names);

    return catalog;
}

const CircuitCatalog kCircuitCatalog = BuildCircuitCatalog();

AigCircuit LoadAigCircuit(const CircuitMeta& meta) {
    AigCircuit circuit;
    circuit.meta = meta;
    circuit.aig = aiger_init();
    const char* error = aiger_open_and_read_from_file(circuit.aig, meta.path.c_str());
    if (error != nullptr) {
        throw std::runtime_error(error);
    }

    assert(circuit.aig->num_inputs == meta.primary_inputs);
    assert(circuit.aig->num_latches == meta.latches);
    assert(circuit.aig->num_outputs == meta.primary_outputs);
    assert(circuit.aig->num_bad == meta.bads);
    assert(circuit.aig->num_ands == meta.ands);
    assert(circuit.aig->maxvar == meta.max_var);
    return circuit;
}

const CircuitCatalog& GetCircuitCatalog() {
    return kCircuitCatalog;
}

const CircuitMeta& GetCircuitMeta(const char* set_name, const char* case_name) {
    assert(set_name != nullptr);
    assert(case_name != nullptr);

    const CircuitCatalog& catalog = GetCircuitCatalog();
    const auto set_it = catalog.sets.find(set_name);
    assert(set_it != catalog.sets.end());
    const auto case_it = set_it->second.find(case_name);
    assert(case_it != set_it->second.end());
    return case_it->second;
}

const AigCircuit& GetAigCircuit(const char* set_name, const char* case_name) {
    using CircuitKey = std::pair<std::string, std::string>;

    static std::map<CircuitKey, AigCircuit> circuits;
    static std::mutex circuits_mutex;

    std::lock_guard<std::mutex> lock(circuits_mutex);

    const CircuitKey key{set_name, case_name};
    auto it = circuits.find(key);
    if (it == circuits.end()) {
        it = circuits.emplace(key, LoadAigCircuit(GetCircuitMeta(set_name, case_name))).first;
    }
    return it->second;
}

bool EvaluateAigLiteral(
    const AigCircuit& circuit,
    uint32_t literal,
    std::vector<int8_t>& cache)
{
    const unsigned stripped = aiger_strip(literal);
    const uint32_t var = aiger_lit2var(literal);
    const bool inverted = aiger_sign(literal) != 0;

    bool result = false;
    if (var == 0) {
        result = false;
    } else {
        assert(var < cache.size());
        int8_t& cached = cache[var];
        if (cached < 0) {
            const aiger_and* gate = aiger_is_and(circuit.aig, stripped);
            assert(gate != nullptr);
            cached = EvaluateAigLiteral(circuit, gate->rhs0, cache) &&
                     EvaluateAigLiteral(circuit, gate->rhs1, cache);
        }
        result = cached != 0;
    }

    return result != inverted;
}

void WriteAigOutputs(
    std::string& value,
    size_t offset,
    const AigCircuit& circuit,
    std::string_view input_state)
{
    std::vector<int8_t> cache(circuit.aig->maxvar + 1, -1);
    for (uint32_t input_id = 0; input_id < circuit.meta.primary_inputs; ++input_id) {
        const uint32_t var = aiger_lit2var(circuit.aig->inputs[input_id].lit);
        cache[var] = input_state[input_id] == '1';
    }
    for (uint32_t latch_id = 0; latch_id < circuit.meta.latches; ++latch_id) {
        const uint32_t var = aiger_lit2var(circuit.aig->latches[latch_id].lit);
        cache[var] = input_state[circuit.meta.primary_inputs + latch_id] == '1';
    }

    size_t output_id = 0;
    for (uint32_t id = 0; id < circuit.aig->num_outputs; ++id) {
        value[offset + output_id] = EvaluateAigLiteral(
            circuit,
            circuit.aig->outputs[id].lit,
            cache) ? '1' : '0';
        ++output_id;
    }
    for (uint32_t id = 0; id < circuit.aig->num_bad; ++id) {
        value[offset + output_id] = EvaluateAigLiteral(
            circuit,
            circuit.aig->bad[id].lit,
            cache) ? '1' : '0';
        ++output_id;
    }
}

}  // namespace

const std::string& CircuitSets() {
    return GetCircuitCatalog().joined_sets;
}

const std::string& CircuitCases(const char* set_name) {
    assert(set_name != nullptr);

    const CircuitCatalog& catalog = GetCircuitCatalog();
    const auto it = catalog.joined_cases.find(set_name);
    assert(it != catalog.joined_cases.end());
    return it->second;
}

size_t CircuitInputs(const char* set_name, const char* case_name) {
    return GetCircuitMeta(set_name, case_name).inputs;
}

size_t CircuitOutputs(const char* set_name, const char* case_name) {
    return GetCircuitMeta(set_name, case_name).outputs;
}

const char* CircuitValue(
    const char* set_name,
    const char* case_name,
    const char* input_state)
{
    assert(input_state != nullptr);

    const AigCircuit& circuit = GetAigCircuit(set_name, case_name);
    const size_t input_state_size = circuit.meta.inputs;
    const size_t outputs = circuit.meta.outputs;
    assert(std::strlen(input_state) == input_state_size);

    thread_local std::string value;
    thread_local std::string point_input;

    point_input.assign(input_state, input_state_size);
    value.assign(input_state_size + outputs * (input_state_size + 1), '0');
    value.replace(0, input_state_size, point_input);

    WriteAigOutputs(value, input_state_size, circuit, point_input);
    for (size_t bit_id = 0; bit_id < input_state_size; ++bit_id) {
        point_input[bit_id] = point_input[bit_id] == '1' ? '0' : '1';
        WriteAigOutputs(
            value,
            input_state_size + outputs * (bit_id + 1),
            circuit,
            point_input);
        point_input[bit_id] = point_input[bit_id] == '1' ? '0' : '1';
    }

    return value.c_str();
}
