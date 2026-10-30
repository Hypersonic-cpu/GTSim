#include "simulator.h"
#include "hbf_address.h"
#include <algorithm>
#include <cctype>
#include <iomanip>
#include <limits>

namespace GTSim {

void Simulator::print_configuration(bool verbose) {
    if (!verbose) return;

    *output_stream << "Starting simulation with " << dag->nodes.size() << " nodes...\n";

    // Show warp distribution
    std::unordered_map<int, int> warp_distribution;
    for (auto* node : dag->nodes) {
        warp_distribution[node->warp_id]++;
    }

    *output_stream << "Warp distribution: ";
    for (const auto& [warp_id, count] : warp_distribution) {
        *output_stream << warp_id << ":" << count << " ";
    }
    *output_stream << "\n";
    *output_stream << "Total warps in use: " << warp_distribution.size() << "\n";
    *output_stream << "Warps per subpartition: " << config.warp_count / 4
              << " (SP0: warps 0,4,8,12... | SP1: 1,5,9,13... | SP2: 2,6,10,14... | SP3: 3,7,11,15...)\n";
    *output_stream << "\n";

    *output_stream << "Dynamic Allocation (O2): " << (config.O2_dynamic_allocation ? "ENABLED" : "DISABLED") << "\n";
    if (config.O2_dynamic_allocation) {
        *output_stream << "  O2 Behavior: Scan entire warp for ready nodes before switching warps\n";
        *output_stream << "  Pipeline fairness: Prefer different pipeline than last issued\n";
    }

    *output_stream << "Scheduling Policy: Per-Subpartition GTO (Greedy-Then-Oldest)\n";
    *output_stream << "  - Each subpartition schedules independently among its 4 warps\n";
    *output_stream << "  - Greedy: Continue from last issued warp if possible\n";
    *output_stream << "  - Oldest: Pick warp with longest stall time (oldest last_issue_cycle)\n";
    *output_stream << "Memory model: T_sum = T_frontend + T_queuing + T_data + T_bank_conflict\n";
    *output_stream << "Tensor throughput: " << config.tensor_core_throughput << " ops/inst\n";
    *output_stream << "SIMD throughput: " << config.simd_throughput << " ops/inst\n";
    *output_stream << "SFU throughput: " << config.sfu_throughput << " ops/inst\n";
    *output_stream << "LS throughput: " << config.ls_throughput_bytes << " bytes/inst\n";
    *output_stream << "Tensor width: " << config.tensor_core_width << " ops/cycle\n";
    *output_stream << "SIMD width: " << config.simd_width << " ops/cycle\n";
    *output_stream << "SFU width: " << config.sfu_width << " ops/cycle\n";
    *output_stream << "LS width: " << config.ls_width_bytes << " bytes/cycle\n";
    *output_stream << "Number of SMs: " << config.num_sms << "\n";
    if (config.cluster_size > 0) {
        *output_stream << "Cluster size: " << config.cluster_size << " SMs\n";
        *output_stream << "Cluster NoC bandwidth: "
                       << config.cluster_noc_bandwidth_bytes_per_cycle
                       << " bytes/cycle\n";
        *output_stream << "Cluster NoC latency: "
                       << config.cluster_noc_latency_cycles << " cycles\n";
    } else {
        *output_stream << "Cluster size: DISABLED\n";
    }
    *output_stream << "Max concurrent blocks per SM: " << config.max_concurrent_blocks_per_sm << "\n";
    const char* workload_name = "Unknown";
    switch (config.workload_type) {
        case WorkloadType::GEMM: workload_name = "GEMM"; break;
        case WorkloadType::FA3: workload_name = "FA3"; break;
        case WorkloadType::FlashMLA: workload_name = "FlashMLA"; break;
        case WorkloadType::FlashDecoding: workload_name = "FlashDecoding"; break;
        case WorkloadType::ClusterReduce: workload_name = "ClusterReduce"; break;
    }
    *output_stream << "Workload type: " << workload_name << "\n";
    if (!simulated_gpu->sms.empty()) {
        *output_stream << "SRAM read queue depth: " << simulated_gpu->sms[0]->sram->memory_queue_depth << "\n";
        *output_stream << "SRAM write queue depth: " << simulated_gpu->sms[0]->sram->memory_write_queue_depth << "\n";
        *output_stream << "TMEM read queue depth: " << simulated_gpu->sms[0]->tmem_mem->memory_queue_depth << "\n";
        *output_stream << "TMEM write queue depth: " << simulated_gpu->sms[0]->tmem_mem->memory_write_queue_depth << "\n";
        *output_stream << "TMEM read latency: " << config.tmem_read_latency_cycles << " cycles\n";
        *output_stream << "TMEM write latency: " << config.tmem_write_latency_cycles << " cycles\n";
        *output_stream << "TMEM read bandwidth: " << config.tmem_read_bandwidth_bytes_per_cycle << " bytes/cycle\n";
        *output_stream << "TMEM write bandwidth: " << config.tmem_write_bandwidth_bytes_per_cycle << " bytes/cycle\n";
    }
    *output_stream << "L2 cache size: " << config.l2_cache_size_bytes << " bytes\n";
    *output_stream << "L2 line size: " << config.l2_line_size_bytes << " bytes\n";
    *output_stream << "L2 hit latency: " << config.l2_hit_latency_cycles << " cycles\n";
    *output_stream << "L2 bandwidth: " << config.l2_bandwidth_bytes_per_cycle << " bytes/cycle\n";
    *output_stream << "L2 write bandwidth: " << config.l2_write_bandwidth_bytes_per_cycle
                   << " bytes/cycle\n";
    *output_stream << "L2 queue depth: " << config.l2_queue_depth << "\n";
    *output_stream << "L2 max transactions per cycle per subpartition: "
                   << config.l2_max_transactions_per_cycle_per_sp << "\n";
    *output_stream << "Core frequency: " << config.core_frequency_mhz << " MHz\n";
    *output_stream << "Memory topology: "
                   << memory_topology_name(config.memory.topology) << "\n";
    if (config.memory.hbm.backend == HBMBackend::Naive) {
        *output_stream << "HBM backend: naive; read bandwidth: " << config.memory.hbm.read_bw_GBps
                       << " GB/s, write max: " << config.memory.hbm.write_max_bw_GBps << " GB/s\n";
        *output_stream << "HBM read/write latency: " << config.memory.hbm.read_latency_cycles
                       << "/" << config.memory.hbm.write_latency_cycles << " cycles\n";
    } else {
        *output_stream << "HBM backend: ramulator2; timing profile: "
                       << config.memory.hbm.ramulator.config_file << "\n";
        const int stacks = config.memory.hbm.ramulator.stacks;
        *output_stream << "HBM topology: " << stacks << " stacks, " << stacks * 2
                       << " logical MC groups, " << stacks * 16 << " physical channels, "
                       << stacks * 32 << " pseudochannels\n";
    }
    *output_stream << "HBF read/write bandwidth: "
                   << config.memory.hbf.read_bw_GBps << "/"
                   << config.memory.hbf.write_bw_GBps << " GB/s\n";
    *output_stream << "HBF read/write latency: "
                   << config.memory.hbf.read_latency_cycles << "/"
                   << config.memory.hbf.write_latency_cycles << " cycles\n";
    if (config.memory.hbf.backend == HBFBackend::BankPlane) {
        const auto& c = config.memory.hbf.controller;
        *output_stream << "HBF backend: bank_plane; profile "
                       << (c.geometry == HBFGeometryProfile::OCPReference ? "ocp_reference" : "hbflex_research")
                       << "; stacks " << c.stacks << "; Host Channels " << c.stacks * 16
                       << "; Banks " << c.stacks * 16 * c.banks_per_channel
                       << "; Planes " << c.stacks * c.dies_per_stack * c.planes_per_die << "; buffers/Bank 2; sense engines/Bank 1\n";
        *output_stream << "HBF per-stack target read/write GB/s: " << c.read_target_GBps_per_stack
                       << "/" << c.write_target_GBps_per_stack << "; array read/program ceilings: "
                       << hbf_array_read_ceiling_GBps(c) << "/" << hbf_array_write_ceiling_GBps(c)
                       << "; UCIe ceiling: " << c.ucie_ceiling_GBps_per_stack
                       << "; SRAM latency hiding: disabled; page caches: initially empty\n";
    }
    const auto print_link_config = [&](const char* name,
                                       const D2DLinkConfig& link) {
        *output_stream << name << " link: " << link.bandwidth_GBps
                       << " GB/s, " << link.latency_cycles << " cycles\n";
    };
    switch (config.memory.topology) {
        case MemoryTopology::FullHBM:
            print_link_config("gpu_to_hbm", config.memory.gpu_to_hbm);
            break;
        case MemoryTopology::FullHBF:
            print_link_config("gpu_to_hbf", config.memory.gpu_to_hbf);
            break;
        case MemoryTopology::CoLocated:
            print_link_config("gpu_to_hbm", config.memory.gpu_to_hbm);
            print_link_config("gpu_to_hbf", config.memory.gpu_to_hbf);
            break;
        case MemoryTopology::Cascaded:
            print_link_config("gpu_to_hbm", config.memory.gpu_to_hbm);
            print_link_config("hbm_to_hbf", config.memory.hbm_to_hbf);
            break;
    }
    *output_stream << "\n";
}

void Simulator::print_statistics(bool verbose) {
    if (!verbose) return;

    *output_stream << "\n✅ Simulation completed at cycle " << cycle << "\n\n";

    // Bank conflict statistics
    int total_conflicts = 0;
    int total_conflict_cycles = 0;
    for (auto* sm_inst : simulated_gpu->sms) {
        total_conflicts += sm_inst->sram->num_loads_with_conflicts;
        total_conflict_cycles += sm_inst->sram->total_bank_conflict_cycles;
    }

    *output_stream << "Shared Memory Bank Conflict Stats:\n";
    *output_stream << "  Total ld.sram2reg with conflicts: " << total_conflicts << "\n";
    *output_stream << "  Total cycles added by conflicts: " << total_conflict_cycles << "\n";
    if (total_conflicts > 0) {
        double avg_conflict = static_cast<double>(total_conflict_cycles) / total_conflicts;
        *output_stream << "  Average conflict cycles per conflicted load: "
                  << std::fixed << std::setprecision(2) << avg_conflict << "\n";
    }
    const auto memory_stats = simulated_gpu->memory_system->statistics(cycle);
    const auto print_tier_stats = [&](const char* name,
                                      const TierMemoryStatistics& stats) {
        *output_stream << name << " requests: " << stats.request_count
                       << " (read " << stats.read_requests << ", write "
                       << stats.write_requests << "), bytes: read "
                       << stats.read_bytes << ", write " << stats.write_bytes
                       << ", average latency: " << std::fixed
                       << std::setprecision(2)
                       << stats.average_latency_cycles() << " cycles\n";
    };
    *output_stream << "Off-chip memory statistics:\n";
    print_tier_stats("  HBM", memory_stats.hbm);
    print_tier_stats("  HBF", memory_stats.hbf);
    const auto backend = simulated_gpu->memory_system->hbm_backend_statistics();
    if (config.memory.hbm.backend == HBMBackend::Ramulator2) {
        const auto channels_used = std::count_if(backend.channel_transactions.begin(),
                                                backend.channel_transactions.end(),
                                                [](auto count) { return count > 0; });
        *output_stream << "Ramulator HBM3 parents: " << backend.accepted_requests
                       << "/" << backend.completed_requests << " accepted/completed; bursts: "
                       << backend.submitted_transactions << "/" << backend.completed_transactions
                       << "; bytes: " << backend.transaction_bytes << "; admission retries: "
                       << backend.admission_retries << "; DRAM ticks: " << backend.dram_ticks
                       << "; channels used: " << channels_used << "/"
                       << backend.channel_transactions.size() << "\n";
        const auto [min_channel, max_channel] = std::minmax_element(
            backend.channel_transactions.begin(), backend.channel_transactions.end());
        *output_stream << "Ramulator controllers: read/write accepted " << backend.controller_reads
                       << "/" << backend.controller_writes << ", served "
                       << backend.controller_served_reads << "/" << backend.controller_served_writes
                       << "; row hit/miss/conflict " << backend.row_hits << "/"
                       << backend.row_misses << "/" << backend.row_conflicts
                       << "; channel burst min/max " << *min_channel << "/" << *max_channel << "\n";
    }
    if (config.memory.hbf.backend == HBFBackend::BankPlane) {
        const auto hbf = simulated_gpu->memory_system->hbf_backend_statistics();
        *output_stream << "HBF detailed JSON: {";
        bool first = true;
        const auto field = [&](const char* name) {
            if (!first) *output_stream << ",";
            first = false;
            *output_stream << "\"" << name << "\":";
        };
        const auto scalar = [&](const char* name, auto value) { field(name); *output_stream << value; };
        const auto array = [&](const char* name, const auto& values) {
            field(name); *output_stream << "[";
            for (std::size_t i = 0; i < values.size(); ++i) {
                if (i) *output_stream << ",";
                *output_stream << values[i];
            }
            *output_stream << "]";
        };
        scalar("accepted_requests", hbf.accepted_requests);
        scalar("completed_requests", hbf.completed_requests);
        scalar("failed_requests", hbf.failed_requests);
        scalar("host_read_bytes", hbf.host_read_bytes);
        scalar("host_write_bytes", hbf.host_write_bytes);
        scalar("sensed_bytes", hbf.sensed_bytes);
        scalar("senses", hbf.senses);
        scalar("programs", hbf.programs);
        scalar("erases", hbf.erases);
        scalar("cache_hits", hbf.cache_hits);
        scalar("cache_misses", hbf.cache_misses);
        scalar("cache_evictions", hbf.cache_evictions);
        scalar("queue_stalls", hbf.queue_stalls);
        scalar("bank_wait_ps", hbf.bank_wait_ps);
        scalar("transfer_wait_ps", hbf.transfer_wait_ps);
        scalar("events_processed", hbf.events_processed);
        scalar("elapsed_ps", hbf.elapsed_ps);
        array("stack_read_bytes", hbf.stack_read_bytes);
        array("channel_read_bytes", hbf.channel_read_bytes);
        array("bank_reads", hbf.bank_reads);
        array("plane_reads", hbf.plane_reads);
        array("stack_busy_ps", hbf.stack_busy_ps);
        array("channel_read_busy_ps", hbf.channel_read_busy_ps);
        array("channel_write_busy_ps", hbf.channel_write_busy_ps);
        array("bank_busy_ps", hbf.bank_busy_ps);
        array("plane_sense_busy_ps", hbf.plane_sense_busy_ps);
        array("plane_program_busy_ps", hbf.plane_program_busy_ps);
        array("plane_erase_busy_ps", hbf.plane_erase_busy_ps);
        array("die_transfer_busy_ps", hbf.die_transfer_busy_ps);
        array("ecc_busy_ps", hbf.ecc_busy_ps);
        *output_stream << "}\n";
    }
    const auto print_link_stats = [&](const char* name,
                                      const D2DLinkStatistics& stats) {
        *output_stream << "  " << name << " transferred bytes: GPU->memory "
                       << stats.forward_bytes << ", memory->GPU "
                       << stats.reverse_bytes << "; utilization: "
                       << std::fixed << std::setprecision(1)
                       << (stats.forward_utilization * 100.0) << "%/"
                       << (stats.reverse_utilization * 100.0) << "%\n";
    };
    if (memory_stats.has_gpu_to_hbm) {
        print_link_stats("gpu_to_hbm", memory_stats.gpu_to_hbm);
    }
    if (memory_stats.has_gpu_to_hbf) {
        print_link_stats("gpu_to_hbf", memory_stats.gpu_to_hbf);
    }
    if (memory_stats.has_hbm_to_hbf) {
        print_link_stats("hbm_to_hbf", memory_stats.hbm_to_hbf);
    }
    *output_stream << "\n";
}

bool Simulator::run(int max_cycles, bool verbose) {
    print_configuration(verbose);

    if (cycle == 0 && config.startup_delay_cycles > 0) {
        cycle = std::max(0, config.startup_delay_cycles - 1);
    }

    while (cycle < max_cycles) {
        step();
        if (verbose && cycle % 10000 == 0) {
            *output_stream << "Progress: cycle " << cycle << "\n";
        }

        if (completed_node_count == dag->nodes.size()) {
            // Kernel timing remains the last DAG completion cycle. Drain issued
            // traffic separately, including dirty eviction writebacks, without
            // adding an implicit L2 flush or changing reference kernel cycles.
            int drain_cycle = cycle;
            while (!simulated_gpu->memory_system->idle() && drain_cycle < max_cycles) {
                simulated_gpu->memory_system->step(++drain_cycle);
            }
            if (!simulated_gpu->memory_system->idle()) {
                *output_stream << "Memory drain timed out at cycle " << drain_cycle << "\n";
                return false;
            }
            *output_stream << "Memory drained at cycle: " << drain_cycle << "\n";
            print_statistics(verbose);
            print_results();
            return true;
        }
    }

    if (verbose) {
        *output_stream << "⚠️ Timeout at cycle " << cycle << "\n\n";
        print_results();
    }
    return false;
}

void Simulator::print_results() {
    if (!config.silence_mode) {
        const int kLineWidth = 160;
        *output_stream << std::string(kLineWidth, '=') << "\n";
        *output_stream << std::left << std::setw(6) << "ID"
                  << std::setw(40) << "Name"
                  << std::setw(8) << "Type"
                  << std::setw(24) << "Op"
                  << std::setw(6) << "Warp"
                  << std::setw(6) << "SM"
                  << std::setw(6) << "TB"
                  << std::setw(8) << "Start"
                  << std::setw(8) << "End"
                  << std::setw(6) << "Dur" << "\n";
        *output_stream << std::string(kLineWidth, '=') << "\n";

        // Find max warp_id
        int max_warp_id = 0;
        for (auto* node : dag->nodes) {
            max_warp_id = std::max(max_warp_id, node->warp_id);
        }

        for (int warp_id = 0; warp_id <= max_warp_id; ++warp_id) {
            std::vector<DAGNode*> warp_nodes;
            for (auto* node : dag->nodes) {
                if (node->warp_id == warp_id) {
                    warp_nodes.push_back(node);
                }
            }

            if (warp_nodes.empty()) continue;

            std::sort(warp_nodes.begin(), warp_nodes.end(),
                      [](const DAGNode* a, const DAGNode* b) {
                          int a_issue = (a->issue_cycle >= 0) ? a->issue_cycle
                                                              : std::numeric_limits<int>::max();
                          int b_issue = (b->issue_cycle >= 0) ? b->issue_cycle
                                                              : std::numeric_limits<int>::max();
                          if (a_issue != b_issue) return a_issue < b_issue;
                          if (a->start != b->start) return a->start < b->start;
                          return a->id < b->id;
                      });

            *output_stream << "\n--- Warp " << warp_id << " (" << warp_nodes.size() << " nodes) ---\n";

            for (auto* node : warp_nodes) {
                int duration = (node->end != -1 && node->start != -1) ? (node->end - node->start) : -1;
                std::string status = node->finished ? "✓" : "✗";

                *output_stream << std::left << std::setw(6) << node->id
                          << std::setw(40) << node->name
                          << std::setw(8) << node->pipeline_type
                          << std::setw(24) << node->op
                          << std::setw(6) << node->warp_id
                          << std::setw(6) << node->sm_id
                          << std::setw(6) << node->thread_block_id
                          << std::setw(8) << node->start
                          << std::setw(8) << node->end
                          << std::setw(6) << duration
                          << status << "\n";
            }
        }

        *output_stream << "\n" << std::string(kLineWidth, '=') << "\n";
    }
    int finished = 0;
    for (auto* node : dag->nodes) {
        if (node->finished) finished++;
    }

    int kernel_start = std::numeric_limits<int>::max();
    int last_tma_issue = std::numeric_limits<int>::min();
    int last_output_store_issue = std::numeric_limits<int>::min();
    int last_dram_store_issue = std::numeric_limits<int>::min();
    for (auto* node : dag->nodes) {
        if (node->start >= 0) {
            kernel_start = std::min(kernel_start, node->start);
        }
        if (node->op_type == OpType::CP_SRAM2DRAM_TMA &&
            node->tma_issue_complete_cycle >= 0) {
            last_tma_issue = std::max(last_tma_issue, node->tma_issue_complete_cycle);
        }
        if ((node->op_type == OpType::CP_SRAM2DRAM_TMA ||
             node->op_type == OpType::CP_SRAM2DRAM ||
             node->op_type == OpType::ST_REG2DRAM) &&
            node->issue_cycle >= 0) {
            last_output_store_issue = std::max(last_output_store_issue, node->issue_cycle);
        }
        if (node->op_type == OpType::ST_REG2DRAM &&
            node->issue_cycle >= 0) {
            last_dram_store_issue = std::max(last_dram_store_issue, node->issue_cycle);
        }
    }
    int total_cycles = cycle;
    std::string kernel_time_line;
    if (last_tma_issue != std::numeric_limits<int>::min()) {
        kernel_time_line = "Kernel time (last TMA issue cycle): " + std::to_string(last_tma_issue);
    }

    struct TBStats {
        int min_start;
        int max_end;
        int node_count;
        int finished_count;
        int first_sm_id;
        int first_wgmma_issue;
        int first_st_start;
        int last_st_end;
        int first_div_start;
        int last_div_end;
        int first_tma_store_start;
        int last_tma_store_end;
        TBStats()
            : min_start(std::numeric_limits<int>::max()),
              max_end(std::numeric_limits<int>::min()),
              node_count(0),
              finished_count(0),
              first_sm_id(-1),
              first_wgmma_issue(std::numeric_limits<int>::max()),
              first_st_start(std::numeric_limits<int>::max()),
              last_st_end(std::numeric_limits<int>::min()),
              first_div_start(std::numeric_limits<int>::max()),
              last_div_end(std::numeric_limits<int>::min()),
              first_tma_store_start(std::numeric_limits<int>::max()),
              last_tma_store_end(std::numeric_limits<int>::min()) {}
    };
    struct TileKey {
        int tb_id;
        int tile_id;
        bool operator==(const TileKey& other) const {
            return tb_id == other.tb_id && tile_id == other.tile_id;
        }
    };
    struct TileKeyHash {
        std::size_t operator()(const TileKey& key) const {
            std::size_t h1 = std::hash<int>{}(key.tb_id);
            std::size_t h2 = std::hash<int>{}(key.tile_id);
            return h1 ^ (h2 + 0x9e3779b9U + (h1 << 6) + (h1 >> 2));
        }
    };

    auto parse_tile_id = [](const std::string& name) -> int {
        std::size_t pos = name.find("_Tile");
        if (pos == std::string::npos) return -1;
        pos += 5;
        int tile_id = 0;
        bool has_digits = false;
        while (pos < name.size() &&
               std::isdigit(static_cast<unsigned char>(name[pos]))) {
            has_digits = true;
            tile_id = tile_id * 10 + (name[pos] - '0');
            ++pos;
        }
        return has_digits ? tile_id : -1;
    };

    std::unordered_map<TileKey, TBStats, TileKeyHash> tile_stats;
    struct RoundCpStats {
        int min_start;
        int max_end;
        RoundCpStats()
            : min_start(std::numeric_limits<int>::max()),
              max_end(std::numeric_limits<int>::min()) {}
    };
    struct RoundComputeStats {
        int min_load_start;
        int max_mma_end;
        RoundComputeStats()
            : min_load_start(std::numeric_limits<int>::max()),
              max_mma_end(std::numeric_limits<int>::min()) {}
    };
    std::unordered_map<TileKey, std::unordered_map<int, RoundCpStats>, TileKeyHash> tile_cp_rounds;
    std::unordered_map<TileKey, std::unordered_map<int, RoundComputeStats>, TileKeyHash> tile_compute_rounds;
    auto parse_outer_id = [](const std::string& name) -> int {
        std::size_t pos = name.find("_outer");
        if (pos == std::string::npos) {
            pos = name.find("_k");
            if (pos == std::string::npos) return -1;
            pos += 2;
        } else {
            pos += 6;
        }
        int outer_id = 0;
        bool has_digits = false;
        while (pos < name.size() &&
               std::isdigit(static_cast<unsigned char>(name[pos]))) {
            has_digits = true;
            outer_id = outer_id * 10 + (name[pos] - '0');
            ++pos;
        }
        return has_digits ? outer_id : -1;
    };
    if (config.workload_type == WorkloadType::GEMM) {
        for (auto* node : dag->nodes) {
            if (node->thread_block_id < 0) continue;
            int tile_id = parse_tile_id(node->name);
            if (tile_id < 0) {
                continue;
            }
            TileKey key{node->thread_block_id, tile_id};
            auto& stats = tile_stats[key];
            stats.node_count++;
            if (node->finished) stats.finished_count++;
            if (stats.first_sm_id < 0 && node->sm_id >= 0) {
                stats.first_sm_id = node->sm_id;
            }
            if (node->start >= 0) stats.min_start = std::min(stats.min_start, node->start);
            if (node->end >= 0) stats.max_end = std::max(stats.max_end, node->end);
            if (node->start >= 0) {
                if (node->op_type == OpType::WGMMA && node->issue_cycle >= 0) {
                    stats.first_wgmma_issue =
                        std::min(stats.first_wgmma_issue, node->issue_cycle);
                }
                if (node->op_type == OpType::ST_REG2SRAM) {
                    stats.first_st_start = std::min(stats.first_st_start, node->start);
                }
                if (node->op_type == OpType::CP_SRAM2DRAM_TMA) {
                    stats.first_tma_store_start =
                        std::min(stats.first_tma_store_start, node->start);
                }
            }
            if (node->end >= 0) {
                if (node->op_type == OpType::ST_REG2SRAM) {
                    stats.last_st_end = std::max(stats.last_st_end, node->end);
                }
                if (node->op_type == OpType::CP_SRAM2DRAM_TMA) {
                    stats.last_tma_store_end =
                        std::max(stats.last_tma_store_end, node->end);
                }
            }
            int outer_id = parse_outer_id(node->name);
            if (outer_id >= 0) {
                if (node->op_type == OpType::CP_DRAM2SRAM_TMA &&
                    node->start >= 0 && node->end >= 0) {
                    auto& round = tile_cp_rounds[key][outer_id];
                    round.min_start = std::min(round.min_start, node->start);
                    round.max_end = std::max(round.max_end, node->end);
                }
                if (node->op_type == OpType::WGMMA &&
                    node->end >= 0) {
                    auto& round = tile_compute_rounds[key][outer_id];
                    round.min_load_start = std::min(round.min_load_start, node->start);
                    round.max_mma_end = std::max(round.max_mma_end, node->end);
                }
            }
        }
        if (!tile_stats.empty()) {
            *output_stream << "\nTile Stats:\n";
            *output_stream << std::left << std::setw(8) << "TB"
                      << std::setw(6) << "SM"
                      << std::setw(8) << "Tile"
                      << std::setw(10) << "Start"
                      << std::setw(10) << "End"
                      << std::setw(10) << "Latency"
                      << std::setw(12) << "PreSt"
                      << std::setw(10) << "St"
                      << std::setw(10) << "Store"
                      << std::setw(12) << "CpAvg"
                      << std::setw(12) << "Mma"
                      << std::setw(12) << "MmaStart"
                      << std::setw(12) << "Finished"
                      << std::setw(8) << "Nodes" << "\n";
            *output_stream << std::string(140, '-') << "\n";
            std::vector<TileKey> keys;
            keys.reserve(tile_stats.size());
            for (const auto& [key, _] : tile_stats) keys.push_back(key);
            std::sort(keys.begin(), keys.end(),
                      [](const TileKey& a, const TileKey& b) {
                          if (a.tb_id != b.tb_id) return a.tb_id < b.tb_id;
                          return a.tile_id < b.tile_id;
                      });
            for (const auto& key : keys) {
                const auto& stats = tile_stats[key];
                int latency = (stats.min_start != std::numeric_limits<int>::max() &&
                               stats.max_end != std::numeric_limits<int>::min())
                                  ? (stats.max_end - stats.min_start)
                                  : -1;
                int pre_st = (stats.min_start != std::numeric_limits<int>::max() &&
                              stats.first_st_start != std::numeric_limits<int>::max())
                                 ? (stats.first_st_start - stats.min_start)
                                 : -1;
                int st_span = (stats.first_st_start != std::numeric_limits<int>::max() &&
                               stats.last_st_end != std::numeric_limits<int>::min())
                                  ? (stats.last_st_end - stats.first_st_start)
                                  : -1;
                int store_span = (stats.first_tma_store_start != std::numeric_limits<int>::max() &&
                                  stats.last_tma_store_end != std::numeric_limits<int>::min())
                                     ? (stats.last_tma_store_end - stats.first_tma_store_start)
                                     : -1;
                double cp_avg = 0.0;
                int cp_count = 0;
                auto cp_it = tile_cp_rounds.find(key);
                if (cp_it != tile_cp_rounds.end()) {
                    long long sum = 0;
                    for (const auto& [_, round] : cp_it->second) {
                        if (round.min_start != std::numeric_limits<int>::max() &&
                            round.max_end != std::numeric_limits<int>::min()) {
                            sum += (round.max_end - round.min_start);
                            cp_count++;
                        }
                    }
                    if (cp_count > 0) {
                        cp_avg = static_cast<double>(sum) / cp_count;
                    }
                }

                double load_mma_avg = 0.0;
                int load_mma_count = 0;
                auto comp_it = tile_compute_rounds.find(key);
                if (comp_it != tile_compute_rounds.end()) {
                    long long sum = 0;
                    for (const auto& [_, round] : comp_it->second) {
                        if (round.min_load_start != std::numeric_limits<int>::max() &&
                            round.max_mma_end != std::numeric_limits<int>::min()) {
                            sum += (round.max_mma_end - round.min_load_start);
                            load_mma_count++;
                        }
                    }
                    if (load_mma_count > 0) {
                        load_mma_avg = static_cast<double>(sum) / load_mma_count;
                    }
                }
            *output_stream << std::left << std::setw(8) << key.tb_id
                      << std::setw(6) << stats.first_sm_id
                      << std::setw(8) << key.tile_id
                      << std::setw(10) << (stats.min_start == std::numeric_limits<int>::max() ? -1 : stats.min_start)
                      << std::setw(10) << (stats.max_end == std::numeric_limits<int>::min() ? -1 : stats.max_end)
                      << std::setw(10) << latency
                          << std::setw(12) << pre_st
                          << std::setw(10) << st_span
                          << std::setw(10) << store_span
                          << std::setw(12) << std::fixed << std::setprecision(2) << cp_avg
                          << std::setw(12) << std::fixed << std::setprecision(2) << load_mma_avg
                          << std::setw(12) << (stats.first_wgmma_issue == std::numeric_limits<int>::max()
                                                  ? -1
                                                  : stats.first_wgmma_issue)
                          << std::setw(12) << (std::to_string(stats.finished_count) + "/" +
                                               std::to_string(stats.node_count))
                          << std::setw(8) << stats.node_count << "\n";
            }
        }
    } else if (config.workload_type == WorkloadType::FA3 ||
               config.workload_type == WorkloadType::FlashDecoding) {
        struct TBKey {
            int tb_id;
            bool operator==(const TBKey& other) const {
                return tb_id == other.tb_id;
            }
        };
        struct TBKeyHash {
            std::size_t operator()(const TBKey& key) const {
                return std::hash<int>{}(key.tb_id);
            }
        };
        struct RoundSpanStats {
            int min_start;
            int max_end;
            RoundSpanStats()
                : min_start(std::numeric_limits<int>::max()),
                  max_end(std::numeric_limits<int>::min()) {}
        };
        std::unordered_map<TBKey, TBStats, TBKeyHash> tb_stats;
        std::unordered_map<int, std::unordered_map<int, RoundSpanStats>> tb_wgmma1_rounds;
        std::unordered_map<int, std::unordered_map<int, RoundSpanStats>> tb_mid_rounds;
        std::unordered_map<int, std::unordered_map<int, RoundSpanStats>> tb_wgmma2_rounds;
        std::unordered_map<int, std::unordered_map<int, RoundSpanStats>> tb_kload_rounds;
        std::unordered_map<int, std::unordered_map<int, RoundSpanStats>> tb_vload_rounds;
        std::unordered_map<int, std::unordered_map<int, RoundSpanStats>> tb_cons_loop_start_rounds;
        std::unordered_map<int, std::unordered_map<int, RoundSpanStats>> tb_cons_loop_end_rounds;
        const bool is_flashdecoding =
            (config.workload_type == WorkloadType::FlashDecoding);
        for (auto* node : dag->nodes) {
            if (node->thread_block_id < 0) continue;
            TBKey key{node->thread_block_id};
            auto& stats = tb_stats[key];
            stats.node_count++;
            if (node->finished) stats.finished_count++;
            if (stats.first_sm_id < 0 && node->sm_id >= 0) {
                stats.first_sm_id = node->sm_id;
            }
            if (node->start >= 0) stats.min_start = std::min(stats.min_start, node->start);
            if (node->end >= 0) stats.max_end = std::max(stats.max_end, node->end);
            if (node->start >= 0) {
                if (node->op_type == OpType::ST_REG2SRAM) {
                    stats.first_st_start = std::min(stats.first_st_start, node->start);
                }
                if (node->op == "acc_o_div") {
                    stats.first_div_start = std::min(stats.first_div_start, node->start);
                }
                if ((!is_flashdecoding && node->op_type == OpType::CP_SRAM2DRAM_TMA) ||
                    (is_flashdecoding &&
                     node->op_type == OpType::ST_REG2DRAM &&
                     node->name.find("_OutPartial_Store_") != std::string::npos)) {
                    stats.first_tma_store_start =
                        std::min(stats.first_tma_store_start, node->start);
                }
            }
            if (node->end >= 0) {
                if (node->op_type == OpType::ST_REG2SRAM) {
                    stats.last_st_end = std::max(stats.last_st_end, node->end);
                }
                if (node->op == "acc_o_div") {
                    stats.last_div_end = std::max(stats.last_div_end, node->end);
                }
                if ((!is_flashdecoding && node->op_type == OpType::CP_SRAM2DRAM_TMA) ||
                    (is_flashdecoding &&
                     node->op_type == OpType::ST_REG2DRAM &&
                     node->name.find("_OutPartial_Store_") != std::string::npos)) {
                    stats.last_tma_store_end =
                        std::max(stats.last_tma_store_end, node->end);
                }
            }

            int outer_id = parse_outer_id(node->name);
            if (outer_id >= 0 && node->start >= 0 && node->end >= 0) {
                if (node->op_type == OpType::WGMMA &&
                    node->name.find("_WGMMA_SS_") != std::string::npos) {
                    auto& round = tb_wgmma1_rounds[node->thread_block_id][outer_id];
                    round.min_start = std::min(round.min_start, node->start);
                    round.max_end = std::max(round.max_end, node->end);
                } else if (node->op_type == OpType::WGMMA &&
                           node->name.find("_WGMMA_RS_") != std::string::npos) {
                    auto& round = tb_wgmma2_rounds[node->thread_block_id][outer_id];
                    round.min_start = std::min(round.min_start, node->start);
                    round.max_end = std::max(round.max_end, node->end);
                } else if (node->name.find("_Softmax_") != std::string::npos) {
                    auto& round = tb_mid_rounds[node->thread_block_id][outer_id];
                    round.min_start = std::min(round.min_start, node->start);
                    round.max_end = std::max(round.max_end, node->end);
                } else if (node->name.find("_K_Load") != std::string::npos) {
                    auto& round = tb_kload_rounds[node->thread_block_id][outer_id];
                    round.min_start = std::min(round.min_start, node->start);
                    round.max_end = std::max(round.max_end, node->end);
                } else if (node->name.find("_V_Load") != std::string::npos) {
                    auto& round = tb_vload_rounds[node->thread_block_id][outer_id];
                    round.min_start = std::min(round.min_start, node->start);
                    round.max_end = std::max(round.max_end, node->end);
                } else if (node->name.find("_ConsumerLoopStart_") != std::string::npos) {
                    auto& round = tb_cons_loop_start_rounds[node->thread_block_id][outer_id];
                    round.min_start = std::min(round.min_start, node->start);
                    round.max_end = std::max(round.max_end, node->end);
                } else if (node->name.find("_ConsumerLoopEnd_") != std::string::npos) {
                    auto& round = tb_cons_loop_end_rounds[node->thread_block_id][outer_id];
                    round.min_start = std::min(round.min_start, node->start);
                    round.max_end = std::max(round.max_end, node->end);
                }
            }
        }
        if (!tb_stats.empty()) {
            *output_stream << "\nBlock Stats ("
                           << (is_flashdecoding ? "FlashDecoding" : "FA3")
                           << "):\n";
            *output_stream << std::left << std::setw(8) << "TB"
                      << std::setw(6) << "SM"
                      << std::setw(10) << "Start"
                      << std::setw(10) << "End"
                      << std::setw(10) << "Latency"
                      << std::setw(12) << "LoopStart"
                      << std::setw(12) << "LoopEnd"
                      << std::setw(12) << "KLoad"
                      << std::setw(12) << "VLoad"
                      << std::setw(12) << "Wgmma1"
                      << std::setw(12) << "Mid"
                      << std::setw(12) << "Wgmma2"
                      << std::setw(12) << "PreDiv"
                      << std::setw(10) << "Div"
                      << std::setw(10) << "St"
                      << std::setw(10) << "Store"
                      << std::setw(12) << "Finished"
                      << std::setw(8) << "Nodes" << "\n";
            *output_stream << std::string(168, '-') << "\n";
            std::vector<TBKey> keys;
            keys.reserve(tb_stats.size());
            for (const auto& [key, _] : tb_stats) keys.push_back(key);
            std::sort(keys.begin(), keys.end(),
                      [](const TBKey& a, const TBKey& b) {
                          return a.tb_id < b.tb_id;
                      });
            for (const auto& key : keys) {
                const auto& stats = tb_stats[key];
                int latency = (stats.min_start != std::numeric_limits<int>::max() &&
                               stats.max_end != std::numeric_limits<int>::min())
                                  ? (stats.max_end - stats.min_start)
                                  : -1;
                double kload_avg = 0.0;
                int kload_count = 0;
                auto k_it = tb_kload_rounds.find(key.tb_id);
                if (k_it != tb_kload_rounds.end()) {
                    long long sum = 0;
                    for (const auto& [_, round] : k_it->second) {
                        if (round.min_start != std::numeric_limits<int>::max() &&
                            round.max_end != std::numeric_limits<int>::min()) {
                            sum += (round.max_end - round.min_start);
                            kload_count++;
                        }
                    }
                    if (kload_count > 0) {
                        kload_avg = static_cast<double>(sum) / kload_count;
                    }
                }

                double vload_avg = 0.0;
                int vload_count = 0;
                auto v_it = tb_vload_rounds.find(key.tb_id);
                if (v_it != tb_vload_rounds.end()) {
                    long long sum = 0;
                    for (const auto& [_, round] : v_it->second) {
                        if (round.min_start != std::numeric_limits<int>::max() &&
                            round.max_end != std::numeric_limits<int>::min()) {
                            sum += (round.max_end - round.min_start);
                            vload_count++;
                        }
                    }
                    if (vload_count > 0) {
                        vload_avg = static_cast<double>(sum) / vload_count;
                    }
                }
                double wgmma1_avg = 0.0;
                int wgmma1_count = 0;
                auto w1_it = tb_wgmma1_rounds.find(key.tb_id);
                if (w1_it != tb_wgmma1_rounds.end()) {
                    long long sum = 0;
                    for (const auto& [_, round] : w1_it->second) {
                        if (round.min_start != std::numeric_limits<int>::max() &&
                            round.max_end != std::numeric_limits<int>::min()) {
                            sum += (round.max_end - round.min_start);
                            wgmma1_count++;
                        }
                    }
                    if (wgmma1_count > 0) {
                        wgmma1_avg = static_cast<double>(sum) / wgmma1_count;
                    }
                }

                double mid_avg = 0.0;
                int mid_count = 0;
                auto mid_it = tb_mid_rounds.find(key.tb_id);
                if (mid_it != tb_mid_rounds.end()) {
                    long long sum = 0;
                    for (const auto& [_, round] : mid_it->second) {
                        if (round.min_start != std::numeric_limits<int>::max() &&
                            round.max_end != std::numeric_limits<int>::min()) {
                            sum += (round.max_end - round.min_start);
                            mid_count++;
                        }
                    }
                    if (mid_count > 0) {
                        mid_avg = static_cast<double>(sum) / mid_count;
                    }
                }

                double wgmma2_avg = 0.0;
                int wgmma2_count = 0;
                auto w2_it = tb_wgmma2_rounds.find(key.tb_id);
                if (w2_it != tb_wgmma2_rounds.end()) {
                    long long sum = 0;
                    for (const auto& [_, round] : w2_it->second) {
                        if (round.min_start != std::numeric_limits<int>::max() &&
                            round.max_end != std::numeric_limits<int>::min()) {
                            sum += (round.max_end - round.min_start);
                            wgmma2_count++;
                        }
                    }
                    if (wgmma2_count > 0) {
                        wgmma2_avg = static_cast<double>(sum) / wgmma2_count;
                    }
                }
                int loop_start = std::numeric_limits<int>::max();
                auto ls_it = tb_cons_loop_start_rounds.find(key.tb_id);
                if (ls_it != tb_cons_loop_start_rounds.end()) {
                    for (const auto& [_, round] : ls_it->second) {
                        if (round.min_start != std::numeric_limits<int>::max()) {
                            loop_start = std::min(loop_start, round.min_start);
                        }
                    }
                }
                if (loop_start == std::numeric_limits<int>::max()) {
                    loop_start = -1;
                }
                int loop_end = std::numeric_limits<int>::min();
                auto le_it = tb_cons_loop_end_rounds.find(key.tb_id);
                if (le_it != tb_cons_loop_end_rounds.end()) {
                    for (const auto& [_, round] : le_it->second) {
                        if (round.max_end != std::numeric_limits<int>::min()) {
                            loop_end = std::max(loop_end, round.max_end);
                        }
                    }
                }
                if (loop_end == std::numeric_limits<int>::min()) {
                    loop_end = -1;
                }
                int pre_div = (stats.min_start != std::numeric_limits<int>::max() &&
                               stats.first_div_start != std::numeric_limits<int>::max())
                                  ? (stats.first_div_start - stats.min_start)
                                  : -1;
                int div_span = (stats.first_div_start != std::numeric_limits<int>::max() &&
                                stats.last_div_end != std::numeric_limits<int>::min())
                                   ? (stats.last_div_end - stats.first_div_start)
                                   : -1;
                int st_span = (stats.first_st_start != std::numeric_limits<int>::max() &&
                               stats.last_st_end != std::numeric_limits<int>::min())
                                  ? (stats.last_st_end - stats.first_st_start)
                                  : -1;
                int store_span = (stats.first_tma_store_start != std::numeric_limits<int>::max() &&
                                  stats.last_tma_store_end != std::numeric_limits<int>::min())
                                     ? (stats.last_tma_store_end - stats.first_tma_store_start)
                                     : -1;
                *output_stream << std::left << std::setw(8) << key.tb_id
                          << std::setw(6) << stats.first_sm_id
                          << std::setw(10) << (stats.min_start == std::numeric_limits<int>::max() ? -1 : stats.min_start)
                          << std::setw(10) << (stats.max_end == std::numeric_limits<int>::min() ? -1 : stats.max_end)
                          << std::setw(10) << latency
                          << std::setw(12) << loop_start
                          << std::setw(12) << loop_end
                          << std::setw(12) << std::fixed << std::setprecision(2) << kload_avg
                          << std::setw(12) << std::fixed << std::setprecision(2) << vload_avg
                          << std::setw(12) << std::fixed << std::setprecision(2) << wgmma1_avg
                          << std::setw(12) << std::fixed << std::setprecision(2) << mid_avg
                          << std::setw(12) << std::fixed << std::setprecision(2) << wgmma2_avg
                          << std::setw(12) << pre_div
                          << std::setw(10) << div_span
                          << std::setw(10) << st_span
                          << std::setw(10) << store_span
                          << std::setw(12) << (std::to_string(stats.finished_count) + "/" +
                                               std::to_string(stats.node_count))
                          << std::setw(8) << stats.node_count << "\n";
            }
        }
    } else if (config.workload_type == WorkloadType::ClusterReduce) {
        struct ReduceSpanStats {
            int min_start;
            int max_end;
            int node_count;
            int finished_count;
            ReduceSpanStats()
                : min_start(std::numeric_limits<int>::max()),
                  max_end(std::numeric_limits<int>::min()),
                  node_count(0),
                  finished_count(0) {}
        };
        auto classify_reduce = [](const std::string& name) -> const char* {
            // Ring-reduce naming in clusterfusion_crossbar.cpp
            if (name.find("RMS_Ring_") != std::string::npos ||
                name.find("RMSReduce") != std::string::npos) {
                return "RMS";
            }
            if (name.find("QKV_Ring_") != std::string::npos ||
                name.find("QKVReduce") != std::string::npos) {
                return "QKV";
            }
            if (name.find("SoftmaxStats_Ring_") != std::string::npos ||
                name.find("SoftmaxStatsReduce") != std::string::npos) {
                return "SoftmaxStats";
            }
            if (name.find("AttnOut_Ring_") != std::string::npos ||
                name.find("AttnOutReduce") != std::string::npos) {
                return "AttnOut";
            }
            return nullptr;
        };

        std::unordered_map<std::string, ReduceSpanStats> reduce_stats;
        for (auto* node : dag->nodes) {
            const char* key = classify_reduce(node->name);
            if (!key) continue;
            auto& stats = reduce_stats[key];
            stats.node_count++;
            if (node->finished) stats.finished_count++;
            if (node->start >= 0) stats.min_start = std::min(stats.min_start, node->start);
            if (node->end >= 0) stats.max_end = std::max(stats.max_end, node->end);
        }

        const std::vector<std::string> reduce_order = {
            "RMS", "QKV", "SoftmaxStats", "AttnOut"
        };
        long long total_reduce_cycles = 0;
        for (const auto& key : reduce_order) {
            auto it = reduce_stats.find(key);
            int start = -1;
            int end = -1;
            int cycles = 0;
            int nodes = 0;
            int finished_nodes = 0;
            if (it != reduce_stats.end()) {
                const auto& stats = it->second;
                nodes = stats.node_count;
                finished_nodes = stats.finished_count;
                if (stats.min_start != std::numeric_limits<int>::max() &&
                    stats.max_end != std::numeric_limits<int>::min()) {
                    start = stats.min_start;
                    end = stats.max_end;
                    cycles = end - start + 1;
                }
            }
            total_reduce_cycles += cycles;
            *output_stream << "[ClusterReduceStats] " << key
                           << " start=" << start
                           << " end=" << end
                           << " cycles=" << cycles
                           << " nodes=" << nodes
                           << " finished=" << finished_nodes << "\n";
        }
        *output_stream << "[ClusterReduceStats] total_cycles="
                       << total_reduce_cycles << "\n";
    } else if (config.workload_type == WorkloadType::FlashMLA) {
        struct TBKey {
            int tb_id;
            bool operator==(const TBKey& other) const {
                return tb_id == other.tb_id;
            }
        };
        struct TBKeyHash {
            std::size_t operator()(const TBKey& key) const {
                return std::hash<int>{}(key.tb_id);
            }
        };
        struct RoundSpanStats {
            int min_start;
            int max_end;
            RoundSpanStats()
                : min_start(std::numeric_limits<int>::max()),
                  max_end(std::numeric_limits<int>::min()) {}
        };
        std::unordered_map<TBKey, TBStats, TBKeyHash> tb_stats;
        std::unordered_map<int, std::unordered_map<int, RoundSpanStats>> tb_wgmma1_rounds;
        std::unordered_map<int, std::unordered_map<int, RoundSpanStats>> tb_mid_rounds;
        std::unordered_map<int, std::unordered_map<int, RoundSpanStats>> tb_wgmma2_rounds;

        for (auto* node : dag->nodes) {
            if (node->thread_block_id < 0) continue;
            TBKey key{node->thread_block_id};
            auto& stats = tb_stats[key];
            stats.node_count++;
            if (node->finished) stats.finished_count++;
            if (stats.first_sm_id < 0 && node->sm_id >= 0) {
                stats.first_sm_id = node->sm_id;
            }
            if (node->start >= 0) stats.min_start = std::min(stats.min_start, node->start);
            if (node->end >= 0) stats.max_end = std::max(stats.max_end, node->end);
            if (node->start >= 0) {
                if (node->op_type == OpType::ST_REG2SRAM &&
                    node->name.find("_ST_acco_") != std::string::npos) {
                    stats.first_st_start = std::min(stats.first_st_start, node->start);
                }
                if (node->op == "acc_o_norm") {
                    stats.first_div_start = std::min(stats.first_div_start, node->start);
                }
                if (node->op_type == OpType::CP_SRAM2DRAM_TMA) {
                    stats.first_tma_store_start =
                        std::min(stats.first_tma_store_start, node->start);
                }
            }
            if (node->end >= 0) {
                if (node->op_type == OpType::ST_REG2SRAM &&
                    node->name.find("_ST_acco_") != std::string::npos) {
                    stats.last_st_end = std::max(stats.last_st_end, node->end);
                }
                if (node->op == "acc_o_norm") {
                    stats.last_div_end = std::max(stats.last_div_end, node->end);
                }
                if (node->op_type == OpType::CP_SRAM2DRAM_TMA) {
                    stats.last_tma_store_end =
                        std::max(stats.last_tma_store_end, node->end);
                }
            }

            int outer_id = parse_outer_id(node->name);
            if (outer_id >= 0 && node->start >= 0 && node->end >= 0) {
                if (node->op_type == OpType::WGMMA &&
                    (node->name.find("_WGMMA_QK_") != std::string::npos ||
                     node->name.find("_WGMMA_QpeKpe_") != std::string::npos)) {
                    auto& round = tb_wgmma1_rounds[node->thread_block_id][outer_id];
                    round.min_start = std::min(round.min_start, node->start);
                    round.max_end = std::max(round.max_end, node->end);
                } else if (node->op_type == OpType::WGMMA &&
                           node->name.find("_WGMMA_PV_") != std::string::npos) {
                    auto& round = tb_wgmma2_rounds[node->thread_block_id][outer_id];
                    round.min_start = std::min(round.min_start, node->start);
                    round.max_end = std::max(round.max_end, node->end);
                } else if (node->name.find("_Softmax_") != std::string::npos) {
                    auto& round = tb_mid_rounds[node->thread_block_id][outer_id];
                    round.min_start = std::min(round.min_start, node->start);
                    round.max_end = std::max(round.max_end, node->end);
                }
            }
        }
        if (!tb_stats.empty()) {
            *output_stream << "\nBlock Stats (FlashMLA):\n";
            *output_stream << std::left << std::setw(8) << "TB"
                      << std::setw(6) << "SM"
                      << std::setw(10) << "Start"
                      << std::setw(10) << "End"
                      << std::setw(10) << "Latency"
                      << std::setw(14) << "MainLoopStart"
                      << std::setw(12) << "Wgmma1"
                      << std::setw(12) << "Mid"
                      << std::setw(12) << "Wgmma2"
                      << std::setw(12) << "PreNorm"
                      << std::setw(10) << "Norm"
                      << std::setw(10) << "St"
                      << std::setw(10) << "Store"
                      << std::setw(12) << "Finished"
                      << std::setw(8) << "Nodes" << "\n";
            *output_stream << std::string(168, '-') << "\n";
            std::vector<TBKey> keys;
            keys.reserve(tb_stats.size());
            for (const auto& [key, _] : tb_stats) keys.push_back(key);
            std::sort(keys.begin(), keys.end(),
                      [](const TBKey& a, const TBKey& b) {
                          return a.tb_id < b.tb_id;
                      });
            for (const auto& key : keys) {
                const auto& stats = tb_stats[key];
                int latency = (stats.min_start != std::numeric_limits<int>::max() &&
                               stats.max_end != std::numeric_limits<int>::min())
                                  ? (stats.max_end - stats.min_start)
                                  : -1;

                auto avg_span = [&](const std::unordered_map<int, std::unordered_map<int, RoundSpanStats>>& rounds,
                                    int tb_id) -> double {
                    auto it = rounds.find(tb_id);
                    if (it == rounds.end()) return 0.0;
                    long long sum = 0;
                    int count = 0;
                    for (const auto& [_, round] : it->second) {
                        if (round.min_start != std::numeric_limits<int>::max() &&
                            round.max_end != std::numeric_limits<int>::min()) {
                            sum += (round.max_end - round.min_start);
                            count++;
                        }
                    }
                    return count > 0 ? static_cast<double>(sum) / count : 0.0;
                };

                double wgmma1_avg = avg_span(tb_wgmma1_rounds, key.tb_id);
                double mid_avg = avg_span(tb_mid_rounds, key.tb_id);
                double wgmma2_avg = avg_span(tb_wgmma2_rounds, key.tb_id);
                int main_loop_start = std::numeric_limits<int>::max();
                auto w1_it = tb_wgmma1_rounds.find(key.tb_id);
                if (w1_it != tb_wgmma1_rounds.end()) {
                    for (const auto& [_, round] : w1_it->second) {
                        if (round.min_start != std::numeric_limits<int>::max()) {
                            main_loop_start = std::min(main_loop_start, round.min_start);
                        }
                    }
                }
                if (main_loop_start == std::numeric_limits<int>::max()) {
                    main_loop_start = -1;
                }

                int prenorm = (stats.min_start != std::numeric_limits<int>::max() &&
                               stats.first_div_start != std::numeric_limits<int>::max())
                                  ? (stats.first_div_start - stats.min_start)
                                  : -1;
                int norm_span = (stats.first_div_start != std::numeric_limits<int>::max() &&
                                 stats.last_div_end != std::numeric_limits<int>::min())
                                    ? (stats.last_div_end - stats.first_div_start)
                                    : -1;
                int st_span = (stats.first_st_start != std::numeric_limits<int>::max() &&
                               stats.last_st_end != std::numeric_limits<int>::min())
                                  ? (stats.last_st_end - stats.first_st_start)
                                  : -1;
                int store_span = (stats.first_tma_store_start != std::numeric_limits<int>::max() &&
                                  stats.last_tma_store_end != std::numeric_limits<int>::min())
                                     ? (stats.last_tma_store_end - stats.first_tma_store_start)
                                     : -1;

                *output_stream << std::left << std::setw(8) << key.tb_id
                          << std::setw(6) << stats.first_sm_id
                          << std::setw(10) << (stats.min_start == std::numeric_limits<int>::max() ? -1 : stats.min_start)
                          << std::setw(10) << (stats.max_end == std::numeric_limits<int>::min() ? -1 : stats.max_end)
                          << std::setw(10) << latency
                          << std::setw(14) << main_loop_start
                          << std::setw(12) << std::fixed << std::setprecision(2) << wgmma1_avg
                          << std::setw(12) << std::fixed << std::setprecision(2) << mid_avg
                          << std::setw(12) << std::fixed << std::setprecision(2) << wgmma2_avg
                          << std::setw(12) << prenorm
                          << std::setw(10) << norm_span
                          << std::setw(10) << st_span
                          << std::setw(10) << store_span
                          << std::setw(12) << (std::to_string(stats.finished_count) + "/" +
                                               std::to_string(stats.node_count))
                          << std::setw(8) << stats.node_count << "\n";
            }
        }
    }

    *output_stream << "Total cycles: " << total_cycles << "\n";
    *output_stream << "Finished: " << finished << "/" << dag->nodes.size() << "\n";
    if (last_dram_store_issue != std::numeric_limits<int>::min()) {
        *output_stream << "Kernel time (last DRAM store issue cycle): " << last_dram_store_issue << "\n";
    }
    if (last_output_store_issue != std::numeric_limits<int>::min()) {
        *output_stream << "Kernel time (last output store issue cycle): " << last_output_store_issue << "\n";
    }
    if (!kernel_time_line.empty()) {
        *output_stream << kernel_time_line << "\n";
    }



    if (!config.silence_mode) {
        *output_stream << "\nRegister Usage:\n";
        for (auto* sm_inst : simulated_gpu->sms) {
            for (auto* sp : sm_inst->sps) {
                // Check active warps for this subpartition
                std::vector<int> active_warps;
                for (int w = 0; w < sp->sp_scheduler->scheduler_warp_num; ++w) {
                    if (w % 4 == sp->subpartition_id && sp->sp_scheduler->warp_head_index[w] > 0) {
                        active_warps.push_back(w);
                    }
                }
                if (!active_warps.empty()) {
                    *output_stream << "  SM" << sm_inst->sm_id << " SP" << sp->subpartition_id
                              << " (Warps ";
                    for (size_t i = 0; i < active_warps.size(); ++i) {
                        *output_stream << active_warps[i];
                        if (i < active_warps.size() - 1) *output_stream << ",";
                    }
                    *output_stream << "): " << sp->sp_register->register_totally_used
                              << " / " << sp->sp_register->register_size << " bytes\n";
                }
            }
        }
        *output_stream << std::string(120, '=') << "\n";
    }
}

} // namespace GTSim
