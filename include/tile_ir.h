#ifndef GTSIM_TILE_IR_H
#define GTSIM_TILE_IR_H

#include "dag_node.h"
#include <istream>
#include <memory>
#include <string>
#include <vector>

namespace GTSim {
struct SimulatorConfig;

// Version 1 is an explicit, two-dimensional tile graph. No addresses or timing
// overrides cross this boundary. See docs/tile-ir.md for the wire contract.
struct TileIRTensor {
    int id;
    std::string name;
    std::vector<std::uint64_t> shape;
    std::vector<std::uint64_t> strides;
    DataType dtype;
    MemoryTier tier{MemoryTier::HBM};
};

struct TileIRCTA {
    int id;
    int sm;
    CTAResources resources;
};

struct TileIROperation {
    int id;
    std::string name;
    std::string op;
    int cta;
    int warp; // Local warp; materialization assigns cta * 12 + warp.
    int tensor;
    DataType dtype;
    Tile region;
    std::vector<int> after_issue;
    std::vector<int> after_complete;
    int granularity_bytes{4};
    Layout layout{Layout::RowMajor};
};

struct TileIR {
    std::vector<TileIRTensor> tensors;
    std::vector<TileIRCTA> ctas;
    std::vector<TileIROperation> operations;
    int version{1};
};

TileIR parse_tile_ir(std::istream& input);
TileIR read_tile_ir(const std::string& path);
std::unique_ptr<DAG> materialize_tile_ir(const TileIR& ir, const SimulatorConfig& config);
// Read-only hardware query used by TVM scheduling preflight, before simulation.
std::string describe_tile_ir_hardware(const SimulatorConfig& config);
} // namespace GTSim
#endif
