#include "memory.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

using namespace GTSim;

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

MemorySystemConfig fast_config(MemoryTopology topology) {
    MemorySystemConfig config;
    config.topology = topology;
    config.hbm = {10000.0, 5000.0, 0, 0, 64};
    config.hbf = {10000.0, 10000.0, 0, 0, 64};
    config.gpu_to_hbm = {10000.0, 0, 64};
    config.gpu_to_hbf = {10000.0, 0, 64};
    config.hbm_to_hbf = {10000.0, 0, 64};
    return config;
}

MemorySystem make_memory(const TensorRegistry& tensors,
                         const MemorySystemConfig& config,
                         int cache_size = 0, bool bypass_cache = true) {
    return MemorySystem(cache_size, 128, 0, 4096, 128, 64, bypass_cache,
                        config, 1000.0, tensors);
}

CacheLineKey line_key(const TensorMetadata& tensor) {
    return {tensor.id, 0, tensor.global_base_address};
}

int run_one(MemorySystem& memory, int node_id, const CacheLineKey& key,
            bool is_write, int& cycle) {
    require(memory.l2()->enqueue_transaction_key(node_id, key, is_write),
            "L2 request was rejected");
    for (int limit = 0; limit < 100000; ++limit, ++cycle) {
        const auto completed = memory.step(cycle);
        if (std::find(completed.begin(), completed.end(), node_id) !=
            completed.end()) {
            const int completed_cycle = cycle;
            ++cycle;
            return completed_cycle;
        }
    }
    throw std::runtime_error("memory request did not complete");
}

std::unordered_map<int, int> run_pair(MemorySystem& memory, int first_id,
                                      const CacheLineKey& first_key,
                                      int second_id,
                                      const CacheLineKey& second_key) {
    require(memory.l2()->enqueue_transaction_key(first_id, first_key, false),
            "first L2 request was rejected");
    require(memory.l2()->enqueue_transaction_key(second_id, second_key, false),
            "second L2 request was rejected");
    std::unordered_map<int, int> completed_at;
    for (int cycle = 0; cycle < 1000 && completed_at.size() < 2; ++cycle) {
        for (const int node_id : memory.step(cycle)) {
            completed_at[node_id] = cycle;
        }
    }
    require(completed_at.size() == 2, "both tier requests must complete");
    return completed_at;
}

void test_single_tier_overrides() {
    TensorRegistry hbf_tensors;
    const auto& hbf_tensor = hbf_tensors.allocate_tensor(
        0, "hbf_input", {64}, {1}, DataType::FP16, MemoryTier::HBF);
    auto hbm_memory = make_memory(hbf_tensors,
                                  fast_config(MemoryTopology::FullHBM));
    int cycle = 0;
    run_one(hbm_memory, 1, line_key(hbf_tensor), false, cycle);
    auto stats = hbm_memory.statistics(cycle);
    require(stats.hbm.read_requests == 1 && stats.hbf.request_count == 0,
            "FullHBM must route HBF-tagged tensors to its only controller");
    require(stats.has_gpu_to_hbm && !stats.has_gpu_to_hbf,
            "FullHBM must instantiate only its HBM link");

    TensorRegistry hbm_tensors;
    const auto& hbm_tensor = hbm_tensors.allocate_tensor(
        0, "hbm_input", {64}, {1}, DataType::FP16, MemoryTier::HBM);
    auto hbf_memory = make_memory(hbm_tensors,
                                  fast_config(MemoryTopology::FullHBF));
    cycle = 0;
    run_one(hbf_memory, 2, line_key(hbm_tensor), false, cycle);
    stats = hbf_memory.statistics(cycle);
    require(stats.hbm.request_count == 0 && stats.hbf.read_requests == 1,
            "FullHBF must route HBM-tagged tensors to its only controller");
    require(!stats.has_gpu_to_hbm && stats.has_gpu_to_hbf,
            "FullHBF must instantiate only its HBF link");
}

void test_co_located_independent_routes() {
    TensorRegistry tensors;
    const auto& hbm_tensor = tensors.allocate_tensor(
        0, "hbm_data", {64}, {1}, DataType::FP16, MemoryTier::HBM);
    const auto& hbf_tensor = tensors.allocate_tensor(
        1, "hbf_data", {64}, {1}, DataType::FP16, MemoryTier::HBF);
    auto config = fast_config(MemoryTopology::CoLocated);
    config.gpu_to_hbm.bandwidth_GBps = 100.0;
    auto memory = make_memory(tensors, config);
    const auto completed = run_pair(memory, 10, line_key(hbm_tensor),
                                    11, line_key(hbf_tensor));
    const auto stats = memory.statistics(100);
    require(stats.hbm.read_requests == 1 && stats.hbf.read_requests == 1,
            "CoLocated must route each tagged address to its own controller");
    require(stats.has_gpu_to_hbm && stats.has_gpu_to_hbf &&
                !stats.has_hbm_to_hbf,
            "CoLocated must instantiate two independent GPU links");
    require(completed.at(11) < completed.at(10),
            "the fast HBF link must not wait behind the slower HBM link");
}

void test_cascaded_shared_link_and_bypass() {
    TensorRegistry tensors;
    const auto& hbm_tensor = tensors.allocate_tensor(
        0, "hbm_data", {64}, {1}, DataType::FP16, MemoryTier::HBM);
    const auto& hbf_tensor = tensors.allocate_tensor(
        1, "hbf_data", {64}, {1}, DataType::FP16, MemoryTier::HBF);
    auto config = fast_config(MemoryTopology::Cascaded);
    config.gpu_to_hbm.bandwidth_GBps = 100.0;
    auto memory = make_memory(tensors, config);
    run_pair(memory, 20, line_key(hbm_tensor), 21, line_key(hbf_tensor));
    const auto stats = memory.statistics(100);
    require(stats.hbm.read_bytes == 128 && stats.hbf.read_bytes == 128,
            "cascaded HBF pass-through must not consume HBM controller bytes");
    require(stats.gpu_to_hbm.forward_bytes == 2 * kMemoryControlBytes,
            "both tiers must share the first link for read requests");
    require(stats.hbm_to_hbf.forward_bytes == kMemoryControlBytes &&
                stats.hbm_to_hbf.reverse_bytes == 128,
            "HBF requests and responses must traverse the second link");
}

void test_controller_bandwidths() {
    HBFConfig hbf_config{1.0, 0.5, 0, 0, 8};
    HBFModel hbf(hbf_config, 1000.0);
    hbf.submit({1, 0x1000, 4, MemoryOperation::Read, MemoryTier::HBF}, 0);
    hbf.submit({2, 0x2000, 4, MemoryOperation::Write, MemoryTier::HBF}, 0);
    int hbf_read_cycle = -1;
    int hbf_write_cycle = -1;
    for (int cycle = 0; cycle < 12; ++cycle) {
        for (const auto& response : hbf.step(cycle)) {
            if (response.request_id == 1) hbf_read_cycle = cycle;
            if (response.request_id == 2) hbf_write_cycle = cycle;
        }
    }
    require(hbf_read_cycle >= 0 && hbf_write_cycle >= 0 &&
                hbf_read_cycle < hbf_write_cycle,
            "HBF read and write queues must use their independent bandwidths");

    HBMConfig hbm_config{2.0, 1.0, 0, 0, 8};
    HBMModel hbm(hbm_config, 1000.0);
    hbm.submit({3, 0x3000, 4, MemoryOperation::Read, MemoryTier::HBM}, 0);
    hbm.submit({4, 0x4000, 4, MemoryOperation::Write, MemoryTier::HBM}, 0);
    int hbm_read_cycle = -1;
    int hbm_write_cycle = -1;
    for (int cycle = 0; cycle < 12; ++cycle) {
        for (const auto& response : hbm.step(cycle)) {
            if (response.request_id == 3) hbm_read_cycle = cycle;
            if (response.request_id == 4) hbm_write_cycle = cycle;
        }
    }
    require(hbm_read_cycle >= 0 && hbm_write_cycle >= 4 &&
                hbm_read_cycle < hbm_write_cycle,
            "HBM writes must obey write_max_bw while sharing total bandwidth");
}

void test_d2d_link_payload_and_full_duplex() {
    D2DLink link({1.0, 0, 4}, 1000.0);
    const MemRequest read{40, 0x5000, 128, MemoryOperation::Read,
                          MemoryTier::HBM};
    const MemRequest write{41, 0x6000, 128, MemoryOperation::Write,
                           MemoryTier::HBM};
    require(link.enqueue_request(read, 0) && link.enqueue_request(write, 0),
            "D2D link should accept requests within its queue capacity");
    require(link.enqueue_response(
                {40, 0x5000, MemoryTier::HBM, true, 128,
                 MemoryOperation::Read},
                0),
            "D2D reverse direction should accept the read response");
    require(link.enqueue_response(
                {41, 0x6000, MemoryTier::HBM, true, kMemoryControlBytes,
                 MemoryOperation::Write},
                0),
            "D2D reverse direction should accept the write acknowledgment");
    require(link.step_requests(15).empty(),
            "a 16-byte read command should consume forward-link bandwidth");
    require(link.step_requests(16).size() == 1,
            "read command should finish after 16 byte-cycles");
    require(link.step_responses(128).size() == 1,
            "reverse read data should progress independently of forward writes");
    require(link.step_responses(144).size() == 1,
            "write acknowledgment should use only control bytes");
    const auto stats = link.statistics(200);
    require(stats.forward_bytes == kMemoryControlBytes + 128 &&
                stats.reverse_bytes == 128 + kMemoryControlBytes,
            "D2D traffic sizes must distinguish requests, payloads, and acks");
}

void test_l2_hits_and_dirty_writeback_tier() {
    TensorRegistry tensors;
    const auto& hbf_tensor = tensors.allocate_tensor(
        0, "hbf_data", {64}, {1}, DataType::FP16, MemoryTier::HBF);
    const auto& hbm_tensor = tensors.allocate_tensor(
        1, "hbm_data", {64}, {1}, DataType::FP16, MemoryTier::HBM);
    auto memory = make_memory(tensors, fast_config(MemoryTopology::CoLocated),
                              128, false);
    int cycle = 0;
    run_one(memory, 30, line_key(hbf_tensor), false, cycle);
    run_one(memory, 31, line_key(hbf_tensor), false, cycle);
    auto stats = memory.statistics(cycle);
    require(stats.hbf.read_requests == 1,
            "an L2 hit must not issue a second HBF request");

    run_one(memory, 32, line_key(hbf_tensor), true, cycle);
    run_one(memory, 33, line_key(hbm_tensor), false, cycle);
    stats = memory.statistics(cycle);
    require(stats.hbf.write_requests == 1 && stats.hbm.write_requests == 0,
            "a dirty HBF cache-line eviction must write back to HBF");
}

} // namespace

int main() {
    try {
        test_single_tier_overrides();
        test_co_located_independent_routes();
        test_cascaded_shared_link_and_bypass();
        test_controller_bandwidths();
        test_d2d_link_payload_and_full_duplex();
        test_l2_hits_and_dirty_writeback_tier();
        std::cout << "Memory topology tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Memory topology test failure: " << error.what() << '\n';
        return 1;
    }
}
