#include "tile_ir.h"
#include <charconv>
#include <fstream>
#include <limits>
#include <sstream>
#include <unordered_map>

namespace GTSim {
namespace {
using Fields = std::unordered_map<std::string, std::string>;

int integer(const std::string& value) {
    int result = -1;
    auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
    if (error != std::errc{} || end != value.data() + value.size() || result < 0)
        throw std::invalid_argument("expected nonnegative int32, got '" + value + "'");
    return result;
}

std::vector<int> list(const std::string& value) {
    if (value.size() < 2 || value.front() != '[' || value.back() != ']')
        throw std::invalid_argument("expected bracketed integer list");
    std::vector<int> result;
    if (value == "[]") return result;
    std::size_t start = 1;
    while (start < value.size() - 1) {
        auto end = value.find(',', start);
        if (end == std::string::npos) end = value.size() - 1;
        result.push_back(integer(value.substr(start, end - start)));
        start = end + 1;
        if (end != value.size() - 1 && start == value.size() - 1)
            throw std::invalid_argument("trailing list comma");
    }
    return result;
}

Fields fields(std::istringstream& line, std::initializer_list<const char*> keys) {
    Fields result;
    std::string token;
    while (line >> token) {
        auto equals = token.find('=');
        if (equals == std::string::npos ||
            !result.emplace(token.substr(0, equals), token.substr(equals + 1)).second)
            throw std::invalid_argument("malformed or duplicate field '" + token + "'");
    }
    if (result.size() != keys.size()) throw std::invalid_argument("incorrect field count");
    for (auto key : keys)
        if (!result.contains(key)) throw std::invalid_argument(std::string("missing field '") + key + "'");
    return result;
}

DataType dtype(const std::string& name) {
    if (name == "bf16") return DataType::BF16;
    if (name == "fp16") return DataType::FP16;
    if (name == "fp32") return DataType::FP32;
    throw std::invalid_argument("unsupported dtype '" + name + "'");
}

std::vector<std::uint64_t> dimensions(const std::string& value) {
    auto values = list(value);
    if (values.size() != 2 || values[0] == 0 || values[1] == 0)
        throw std::invalid_argument("expected two positive dimensions/strides");
    return {static_cast<std::uint64_t>(values[0]), static_cast<std::uint64_t>(values[1])};
}
} // namespace

TileIR parse_tile_ir(std::istream& input) {
    TileIR ir;
    bool header = false;
    std::string line;
    int line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        line = line.substr(0, line.find('#'));
        std::istringstream stream(line);
        std::string kind;
        if (!(stream >> kind)) continue;
        try {
            if (!header) {
                std::string version, extra;
                if (kind != "gtsim.tile_ir" || !(stream >> version) || (version != "1" && version != "2") || stream >> extra)
                    throw std::invalid_argument("expected gtsim.tile_ir 1 or 2");
                ir.version = integer(version);
                header = true;
            } else if (kind == "tensor") {
                auto f = fields(stream, {"id", "name", "shape", "strides", "dtype", "tier"});
                if (f.at("tier") != "HBM" && f.at("tier") != "HBF")
                    throw std::invalid_argument("tensor device must be HBM or HBF");
                ir.tensors.push_back({integer(f.at("id")), f.at("name"), dimensions(f.at("shape")),
                                      dimensions(f.at("strides")), dtype(f.at("dtype")),
                                      f.at("tier") == "HBM" ? MemoryTier::HBM : MemoryTier::HBF});
            } else if (kind == "cta") {
                auto f = fields(stream, {"id", "sm", "warps", "shared", "registers"});
                ir.ctas.push_back({integer(f.at("id")), integer(f.at("sm")),
                    {integer(f.at("shared")), integer(f.at("registers")), integer(f.at("warps"))}});
            } else if (kind.starts_with('%')) {
                int id = integer(kind.substr(1));
                std::string equals, op;
                if (!(stream >> equals >> op) || equals != "=")
                    throw std::invalid_argument("expected %id = operation");
                auto f = ir.version == 1
                    ? fields(stream, {"name", "cta", "warp", "tensor", "dtype", "tile", "after_issue", "after_complete"})
                    : fields(stream, {"name", "cta", "warp", "tensor", "dtype", "tile", "after_issue", "after_complete", "granularity", "layout"});
                auto tile = list(f.at("tile"));
                if (tile.size() != 4 || tile[2] == 0 || tile[3] == 0 ||
                    tile[2] > std::numeric_limits<int>::max() / tile[3])
                    throw std::invalid_argument("tile must be [row,column,rows,columns] with a positive int32 area");
                ir.operations.push_back({id, f.at("name"), op, integer(f.at("cta")), integer(f.at("warp")),
                    integer(f.at("tensor")), dtype(f.at("dtype")), Tile(tile[0], tile[1], tile[2], tile[3]),
                    list(f.at("after_issue")), list(f.at("after_complete"))});
                if (ir.version == 2) {
                    auto& operation = ir.operations.back();
                    operation.granularity_bytes = integer(f.at("granularity"));
                    if (f.at("layout") != "row" && f.at("layout") != "swizzled")
                        throw std::invalid_argument("unknown memory layout");
                    operation.layout = f.at("layout") == "swizzled" ? Layout::Swizzled : Layout::RowMajor;
                }
            } else {
                throw std::invalid_argument("unsupported statement '" + kind + "'");
            }
        } catch (const std::exception& e) {
            throw std::invalid_argument("Tile IR line " + std::to_string(line_number) + ": " + e.what());
        }
    }
    if (input.bad()) throw std::runtime_error("error reading Tile IR");
    if (!header) throw std::invalid_argument("missing Tile IR header");
    if (ir.version == 2 && (ir.tensors.empty() || ir.ctas.empty() || ir.operations.empty()))
        throw std::invalid_argument("empty version 2 program");
    return ir;
}

TileIR read_tile_ir(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open Tile IR: " + path);
    return parse_tile_ir(input);
}
} // namespace GTSim
