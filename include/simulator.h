#ifndef SIMULATOR_H
#define SIMULATOR_H

#include "gpu.h"
#include "dag_node.h"
#include "mem_conf.h"
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <cstdint>
#include <cmath>

namespace GTSim {

enum class WorkloadType {
    GEMM,
    FA3,
    FlashMLA,
    FlashDecoding,
    ClusterReduce
};

// Configuration for the simulator
struct SimulatorConfig {
    int warp_count{4};
    int shared_memory_per_sm_bytes{228 * 1024};
    int shared_memory_per_cta_bytes{227 * 1024};
    int register_file_per_sm_bytes{65536 * 4};
    int max_warps_per_sm{64};
    int tensor_core_throughput{256};  // Tensor ops per instruction
    int tensor_core_latency_cycles{48};
    int simd_throughput{32};
    int simd_latency_cycles{4};
    int sfu_throughput{32};
    int sfu_latency_cycles{30};
    int shfl_latency_cycles{4};
    int ls_throughput_bytes{128};
    int ls_latency_cycles{10};
    int sram_latency_cycles{33};  // H100 local DSMEM access (microbenchmark)
    int sram_bandwidth_bytes_per_cycle{128};
    int sram_queue_depth{64};
    int tensor_core_width{32};
    int simd_width{32};
    int sfu_width{4};
    int ls_width_bytes{128};
    int num_sms{1};
    int max_concurrent_blocks_per_sm{4};  // Maximum concurrent thread blocks per SM
    bool O2_dynamic_allocation{false};
    bool aggressive_store_coalesce{false};
    bool silence_mode{false};
    SchedulePolicy schedule_policy{SchedulePolicy::GTO};
    WorkloadType workload_type{WorkloadType::GEMM};

    // L2 cache configuration
    int l2_cache_size_bytes{40 * 1024 * 1024};
    int l2_line_size_bytes{128};
    int l2_hit_latency_cycles{300};
    int l2_bandwidth_bytes_per_cycle{1853};
    int l2_write_bandwidth_bytes_per_cycle{926};  // Half of the default L2 bandwidth
    int l2_queue_depth{65536};
    int l2_max_transactions_per_cycle_per_sp{1};
    bool l2_bypass_cache{false};

    double core_frequency_mhz{1410.0};
    MemorySystemConfig memory{};

    // Cluster NoC configuration (DSM)
    int cluster_size{0};
    int cluster_noc_bandwidth_bytes_per_cycle{0};
    int cluster_noc_latency_cycles{0};

    // TMA configuration
    int tma_setup_latency_cycles{200};
    int tma_issue_interval_cycles{60};
    int tma_issue_rate_bytes_per_cycle{100};

    // Tensor Memory (Blackwell/UMMA) configuration
    int tmem_cols_per_cta{256};
    int tmem_issue_interval_cycles{1};
    int tmem_read_latency_cycles{33};
    int tmem_read_bandwidth_bytes_per_cycle{128};
    int tmem_write_latency_cycles{33};
    int tmem_write_bandwidth_bytes_per_cycle{128};
    int tmem_queue_depth{64};

    // Startup delay (cycles) before any node can issue
    int startup_delay_cycles{0};
    // Delay (cycles) before scheduling a new TB after a TB completes (per SM)
    int block_schedule_latency_cycles{0};

    // ========================================================================
    // NoC configuration — H100 DSMEM reference parameters
    // ========================================================================
    //
    // Source: "Dissecting the NVIDIA Hopper Architecture through
    //          Microbenchmarking" (arXiv 2501.12084, 2025)
    //
    // H100 DSMEM measured data:
    //   Local SMEM access:             33 cycles
    //   SM-to-SM read (cluster_size=2): 181 cycles
    //   SM-to-SM read (cluster_size=16): 184-213 cycles
    //   Aggregate bandwidth:           3.28 TB/s (@cluster_size=2)
    //   Cluster size:                  16 SMs per GPC
    //   Near-flat latency across cluster sizes → suggests crossbar topology
    //
    // ---- Crossbar defaults ----
    //
    // IMPORTANT: 3.28 TB/s is AGGREGATE GPU bandwidth (all ~7 GPCs).
    // Per-GPC crossbar bandwidth = 3.28 / 7 ≈ 0.469 TB/s = 237 B/cyc.
    //
    // Latency decomposition (read path, cluster_size=2):
    //   total = sram_overhead + 2 × noc_one_way
    //   181 = 33 + 2 × 74  →  noc_latency = 74 cycles (one-way)
    //   sram_latency = 33 cycles (local DSMEM interface + SRAM access)
    //
    // Bandwidth (per-GPC):
    //   3.28 TB/s / 7 GPCs @ 1980MHz = 0.469e12 / 1.98e9 ≈ 237 bytes/cycle
    //
    // ---- Mesh equivalent (what-if analysis) ----
    //
    // Mesh per-hop latency: 3 cycles
    //   Based on 2-stage router pipeline (standard academic baseline).
    //   Ref: "Crossbar NoCs Are Scalable Beyond 100 Nodes"
    //        (Passas et al., IEEE TCAD 2012)
    //
    // Mesh per-link bandwidth: 30 bytes/cycle (for 4×4 mesh)
    //   Derived from bisection bandwidth equivalence:
    //     Crossbar bisection BW = 237 / 2 = 118.5 bytes/cycle
    //     Mesh bisection (4×4):  k = 4 links cross the cut
    //     4 × b_link = 118.5  →  b_link ≈ 30 bytes/cycle
    //
    // Mesh base latency (DSMEM protocol overhead): 60 cycles
    //   Dominates total latency — same as crossbar. Measured crossbar
    //   latency barely changes with cluster_size (181 vs 184-213),
    //   confirming protocol overhead >> network traversal.
    //
    // Mesh per-hop latency: 5 cycles
    //   Realistic 5-stage router pipeline (BW+RC+VA+SA+ST/LT).
    //
    // Mesh latency model: base(60) + hops × per_hop(5)
    //
    // Resulting end-to-end latencies (SM-to-SM read = sram + 2×noc_one_way):
    //                    Mesh one-way    Crossbar(74)  Round-trip comparison
    //   1 hop (neighbor):  60+1×5 = 65    74          Mesh:33+2×65=163  CB:181
    //   2 hops:            60+2×5 = 70    74          Mesh:33+2×70=173  CB:181
    //   3 hops:            60+3×5 = 75    74          Mesh:33+2×75=183  CB:181
    //   6 hops (diagonal): 60+6×5 = 90    74          Mesh:33+2×90=213  CB:181
    //   avg (3.33 hops):   60+17  = 77    74          Nearly identical
    //
    // Key insight: mesh vs crossbar difference is mainly in BANDWIDTH
    // contention, not latency. Near-neighbor traffic avoids far links.
    // ========================================================================
    bool noc_enabled{false};
    NoCTopology noc_topology{NoCTopology::CROSSBAR};
    int noc_latency_cycles{74};  // Crossbar one-way: (181 - 33) / 2
    int noc_bandwidth_bytes_per_cycle{237};  // 3.28 TB/s / 7 GPCs @ 1980 MHz
    int noc_queue_depth{128};
    int noc_mesh_rows{4};  // 4×4 mesh for 16-SM cluster
    int noc_mesh_cols{4};
    int noc_sms_per_cluster{16};  // H100: 16 SMs per GPC
    int noc_link_bandwidth_bytes_per_cycle{30};  // Bisection equiv: 237/2/4 ≈ 30
    int noc_mesh_base_latency_cycles{60};  // DSMEM protocol overhead (dominant term)
    std::vector<int> noc_mesh_sm_mapping{};  // Empty means identity mapping
};

// Shared H100 SXM kernel-simulation profile.  Workload builders must not
// override timing or resource fields: they describe tile-graph semantics only.
inline SimulatorConfig make_h100_sxm_config() {
    SimulatorConfig cfg;
    cfg.tensor_core_throughput = 128;
    cfg.tensor_core_latency_cycles = 16;
    cfg.tensor_core_width = 32;
    cfg.l2_hit_latency_cycles = 380;
    cfg.l2_bandwidth_bytes_per_cycle = 4300;
    cfg.l2_write_bandwidth_bytes_per_cycle = 2150;
    cfg.l2_cache_size_bytes = 50 * 1024 * 1024;
    cfg.l2_bypass_cache = false;
    cfg.core_frequency_mhz = 1980;
    cfg.num_sms = 132;
    cfg.max_concurrent_blocks_per_sm = 1;
    cfg.O2_dynamic_allocation = false;
    cfg.tma_setup_latency_cycles = 100;
    // The TMA engine itself serializes bulk transfers. One scheduler cycle is
    // the shared command-front-end baseline; no workload bypasses it.
    cfg.tma_issue_interval_cycles = 1;
    cfg.tma_issue_rate_bytes_per_cycle = 100;
    // Kernel simulation starts after launch. CTA reuse is represented by the
    // graph's residency and dependency structure rather than a fitted delay.
    cfg.startup_delay_cycles = 0;
    cfg.block_schedule_latency_cycles = 0;
    return cfg;
}

// Convert peak throughput in TB/s to simulator bytes/cycle at core clock.
inline int bytes_per_cycle_from_tbps(double tbps, double core_frequency_mhz) {
    if (tbps <= 0.0 || core_frequency_mhz <= 0.0) return 0;
    return static_cast<int>(std::lround((tbps * 1e12) / (core_frequency_mhz * 1e6)));
}

// Apply Blackwell B200 defaults from the paper evaluation configuration.
// - HBM bandwidth: up to 8 TB/s per GPU (NVIDIA HGX/DGX B200 docs).
// - TMEM peak: 12 TB/s and latency: 28 cycles.
inline void apply_blackwell_b200_defaults(SimulatorConfig& cfg,
                                          double tmem_peak_tbps = 12.0) {
    cfg.core_frequency_mhz = 1965.0;
    cfg.memory.hbm.read_bw_GBps = 8000.0;
    cfg.memory.hbm.write_max_bw_GBps = 4000.0;
    cfg.memory.gpu_to_hbm.bandwidth_GBps = 8000.0;
    cfg.num_sms = 148;
    cfg.l2_cache_size_bytes = 50 * 1024 * 1024; // 50MB
    cfg.l2_hit_latency_cycles = 380;
    cfg.l2_bandwidth_bytes_per_cycle = 6000;
    cfg.l2_write_bandwidth_bytes_per_cycle = 3000;
    cfg.tma_issue_rate_bytes_per_cycle = 100;
    cfg.tma_setup_latency_cycles = 100;
    cfg.tma_issue_interval_cycles = 200;

    int tmem_bw_bpc = bytes_per_cycle_from_tbps(tmem_peak_tbps, cfg.core_frequency_mhz);
    if (tmem_bw_bpc > 0) {
        cfg.tmem_read_bandwidth_bytes_per_cycle = tmem_bw_bpc;
        cfg.tmem_write_bandwidth_bytes_per_cycle = tmem_bw_bpc;
    }
}

// Main simulator class
class Simulator {
public:
    SimulatorConfig config;
    GPU* simulated_gpu;
    DAG* dag;
    int cycle;
    std::ofstream* log_file;
    std::ostream* output_stream;
    std::vector<std::uint8_t> global_scoreboard;
    std::vector<Subpartition*> node_id_to_sp;
    std::vector<int> node_id_to_sm;
    std::size_t completed_node_count;

    Simulator(DAG* input_dag, const SimulatorConfig& cfg = SimulatorConfig())
        : config(cfg), dag(input_dag), cycle(0), log_file(nullptr), output_stream(&std::cout),
          completed_node_count(0) {

        int max_warp_id = -1;
        for (auto* node : dag->nodes) {
            max_warp_id = std::max(max_warp_id, node->warp_id);
        }
        if (max_warp_id >= 0) {
            config.warp_count = max_warp_id + 1;
        }

        if (config.cluster_size > 0) {
            int active_sms = (config.num_sms / config.cluster_size) * config.cluster_size;
            if (active_sms <= 0) {
                active_sms = 0;
            }
            config.num_sms = active_sms;
        }

        simulated_gpu = new GPU(config.num_sms, config.warp_count,
                               config.tensor_core_throughput,
                               config.simd_throughput,
                               config.sfu_throughput,
                               config.ls_throughput_bytes,
                               config.tensor_core_latency_cycles,
                               config.simd_latency_cycles,
                               config.sfu_latency_cycles,
                               config.shfl_latency_cycles,
                               config.ls_latency_cycles,
                               config.tensor_core_width,
                               config.simd_width,
                               config.sfu_width,
                               config.ls_width_bytes,
                               config.sram_latency_cycles,
                               config.sram_bandwidth_bytes_per_cycle,
                               config.sram_queue_depth,
                               config.O2_dynamic_allocation,
                               config.schedule_policy,
                               config.max_concurrent_blocks_per_sm,
                               config.l2_cache_size_bytes,
                               config.l2_line_size_bytes,
                               config.l2_hit_latency_cycles,
                               config.l2_bandwidth_bytes_per_cycle,
                               config.l2_write_bandwidth_bytes_per_cycle,
                               config.l2_queue_depth,
                               config.l2_max_transactions_per_cycle_per_sp,
                               config.l2_bypass_cache,
                               config.memory,
                               config.core_frequency_mhz,
                               config.tma_setup_latency_cycles,
                               config.tma_issue_interval_cycles,
                               config.tma_issue_rate_bytes_per_cycle,
                               config.tmem_cols_per_cta,
                               config.tmem_issue_interval_cycles,
                               config.tmem_read_latency_cycles,
                               config.tmem_read_bandwidth_bytes_per_cycle,
                               config.tmem_write_latency_cycles,
                               config.tmem_write_bandwidth_bytes_per_cycle,
                               config.tmem_queue_depth,
                               config.startup_delay_cycles,
                               config.block_schedule_latency_cycles,
                               dag->tensor_registry);

        // Initialize schedulers with nodes
        int total_nodes = static_cast<int>(dag->nodes.size());
        dag->build_dependency_graph();

        global_scoreboard.assign(total_nodes, 0);

        node_id_to_sp.assign(total_nodes, nullptr);
        node_id_to_sm.assign(total_nodes, -1);

        for (auto* sm_inst : simulated_gpu->sms) {
            for (auto* sp : sm_inst->sps) {
                // Assign warps to subpartitions: warp_id % 4 == sp.subpartition_id
                std::vector<DAGNode*> sp_nodes;
                for (auto* node : dag->nodes) {
                    if (node->warp_id % 4 == sp->subpartition_id) {
                        // Assign node to this SM based on sm_id or thread_block_id
                        if (node->sm_id == -1 || node->sm_id == sm_inst->sm_id) {
                            sp_nodes.push_back(node);
                            if (node->id >= 0 && node->id < total_nodes) {
                                node_id_to_sp[node->id] = sp;
                                node_id_to_sm[node->id] = sm_inst->sm_id;
                            }
                        }
                    }
                }
                sp->sp_scheduler->init_with_nodes(sp_nodes, total_nodes, sp->subpartition_id,
                                                  &global_scoreboard);
                sp->set_node_to_sp_map(&node_id_to_sp);
                sp->set_completed_node_counter(&completed_node_count);
            }
        }
        simulated_gpu->set_node_to_sp_map(&node_id_to_sp);
        simulated_gpu->set_node_to_sm_map(&node_id_to_sm);

        // Initialize NoC if enabled
        if (config.noc_enabled) {
            int cluster_size = config.noc_sms_per_cluster;  // 0 = no restriction
            if (config.noc_topology == NoCTopology::MESH) {
                int rows = config.noc_mesh_rows;
                int cols = config.noc_mesh_cols;
                // Auto-compute mesh dimensions if not specified
                // When clustering is active, mesh covers one cluster; otherwise all SMs
                if (rows <= 0 || cols <= 0) {
                    int mesh_sms = (cluster_size > 0) ? cluster_size : config.num_sms;
                    cols = static_cast<int>(std::ceil(std::sqrt(mesh_sms)));
                    rows = (mesh_sms + cols - 1) / cols;
                }

                // Mesh per-hop latency: noc_latency_cycles is used directly as
                // per_hop when topology is MESH (typically 2-3 cycles for a
                // simple router pipeline, vs 88 cycles for crossbar one-way).
                // User should set noc_latency_cycles appropriately per topology.

                // Derive per-link bandwidth from aggregate bandwidth using
                // bisection equivalence: b_link = (N/2)/k × (BW_total/N)
                // For simpler default: aggregate / num_links
                int num_links = rows * (cols - 1) + (rows - 1) * cols;
                int link_bw = config.noc_link_bandwidth_bytes_per_cycle;
                if (link_bw <= 0 && num_links > 0) {
                    link_bw = config.noc_bandwidth_bytes_per_cycle / num_links;
                }

                simulated_gpu->noc = new MeshNoC(
                    config.noc_mesh_base_latency_cycles,
                    config.noc_latency_cycles,
                    config.noc_bandwidth_bytes_per_cycle,
                    config.noc_queue_depth,
                    rows, cols, cluster_size,
                    link_bw,
                    config.noc_mesh_sm_mapping);
            } else {
                simulated_gpu->noc = new CrossbarNoC(
                    config.noc_latency_cycles,
                    config.noc_bandwidth_bytes_per_cycle,
                    config.noc_queue_depth,
                    cluster_size);
            }
            for (auto* sm : simulated_gpu->sms) {
                sm->set_noc(simulated_gpu->noc);
            }
        }

        // Initialize thread block residency scheduler for each SM
        // Constructor errors do not invoke ~Simulator(). Guard GPU ownership.
        std::unique_ptr<GPU> admission_guard(simulated_gpu);
        for (auto* sm_inst : simulated_gpu->sms) {
            std::vector<DAGNode*> sm_nodes;
            for (auto* node : dag->nodes) {
                if (node->sm_id == -1 || node->sm_id == sm_inst->sm_id) {
                    sm_nodes.push_back(node);
                }
            }
            sm_inst->cta_resources = dag->cta_resources;
            sm_inst->shared_capacity = config.shared_memory_per_sm_bytes;
            sm_inst->register_capacity = config.register_file_per_sm_bytes;
            sm_inst->warp_capacity = config.max_warps_per_sm;
            sm_inst->initialize_tb_scheduler(sm_nodes);
        }
        admission_guard.release();

    }

    ~Simulator() {
        delete simulated_gpu;
        if (log_file) {
            log_file->close();
            delete log_file;
        }
    }

    // Set log file for output
    bool set_log_file(const std::string& filename) {
        log_file = new std::ofstream(filename);
        if (!log_file->is_open()) {
            delete log_file;
            log_file = nullptr;
            return false;
        }
        output_stream = log_file;
        return true;
    }

    // Step simulation by one cycle
    void step() {
        cycle++;
        simulated_gpu->step(cycle);
    }

    // Run simulation until completion or timeout
    bool run(int max_cycles = 500000, bool verbose = true);

    // Print execution results
    void print_results();

private:
    void print_configuration(bool verbose);
    void print_statistics(bool verbose);
};

} // namespace GTSim

#endif // SIMULATOR_H
