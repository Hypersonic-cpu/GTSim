#include "arch_conf.h"
#include "memory_backend.h"
#include <iostream>
#include <unordered_map>

using namespace GTSim;
namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void test_bursts(HBMConfig config) {
    config.queue_capacity = 1;
    auto backend = make_hbm_backend(config, 1980);
    int cycle = 0;
    std::uint64_t id = 1;
    std::uint64_t expected_bursts = 0;
    for (const auto operation : {MemoryOperation::Read, MemoryOperation::Write, MemoryOperation::Writeback}) {
        for (const int size : {32, 64, 128}) {
            const MemRequest request{id++, 0x100000000, size, operation, MemoryTier::HBM};
            require(backend->try_submit(request, cycle), "empty adapter rejected request");
            require(!backend->try_submit({id + 100, 0x100010000, 32, operation, MemoryTier::HBM}, cycle),
                    "bounded adapter failed to backpressure");
            bool completed = false;
            for (int limit = 0; limit < 100000 && !completed; ++limit) {
                for (const auto& response : backend->step(++cycle)) {
                    require(!completed, "request completed twice");
                    require(response.request_id == request.request_id && response.completed &&
                            response.global_addr == request.global_addr && response.operation == operation &&
                            response.tier == request.tier, "callback lost request identity");
                    require(response.size_bytes == (operation == MemoryOperation::Read ? size : 16),
                            "response payload size mismatch");
                    completed = true;
                }
            }
            require(completed && backend->pending_requests() == 0, "burst request failed to drain");
            expected_bursts += size / 32;
        }
    }
    for (int extra = 0; extra < 100; ++extra) {
        require(backend->step(++cycle).empty(), "late duplicate callback");
    }
    const auto stats = backend->backend_statistics();
    require(stats.accepted_requests == 9 && stats.completed_requests == 9,
            "parent conservation failure");
    require(stats.submitted_transactions == expected_bursts &&
            stats.completed_transactions == expected_bursts &&
            stats.transaction_bytes == expected_bursts * 32, "burst/byte conservation failure");
    require(stats.controller_reads + stats.controller_writes == expected_bursts &&
            stats.controller_served_reads + stats.controller_served_writes == expected_bursts,
            "Ramulator controller and adapter counters disagree");
    bool rejected = false;
    try { backend->try_submit({9999, 1, 32, MemoryOperation::Read, MemoryTier::HBM}, cycle); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "unaligned burst was accepted");
}

void test_controller_backpressure(HBMConfig config) {
    config.queue_capacity = 512;
    auto backend = make_hbm_backend(config, 1375.5);
    constexpr int count = 256;
    for (int i = 0; i < count; ++i) {
        // The same physical channel and bank, different rows. Exceeds the
        // profile's 32-entry read/write queues and forces actual send() retries.
        const std::uint64_t address = 128ULL * 80 * 512 * (i / 2);
        require(backend->try_submit({static_cast<std::uint64_t>(i), address, 128,
                    i % 2 ? MemoryOperation::Write : MemoryOperation::Read, MemoryTier::HBM}, 0),
                "adapter capacity unexpectedly exhausted");
    }
    std::unordered_map<std::uint64_t, int> completions;
    for (int cycle = 1; cycle <= 100000 && completions.size() < count; ++cycle) {
        for (const auto& response : backend->step(cycle)) {
            require(++completions[response.request_id] == 1, "duplicate completion after retry");
        }
    }
    const auto stats = backend->backend_statistics();
    require(completions.size() == count && backend->pending_requests() == 0,
            "controller backpressure lost a request");
    require(stats.submitted_transactions == count * 4 && stats.completed_transactions == count * 4,
            "retry duplicated or lost a burst");
    require(stats.admission_retries > 0, "controller queue backpressure was not exercised");
}

void test_arrival_and_ordering(const HBMConfig& config) {
    auto backend = make_hbm_backend(config, 1980);
    require(backend->try_submit({1, 0, 32, MemoryOperation::Write, MemoryTier::HBM}, 1000),
            "future write rejected");
    require(backend->try_submit({2, 0, 32, MemoryOperation::Read, MemoryTier::HBM}, 1000),
            "dependent read rejected");
    for (int cycle = 1; cycle < 1000; ++cycle) {
        require(backend->step(cycle).empty(), "request completed before arrival");
    }
    std::vector<std::uint64_t> order;
    for (int cycle = 1000; cycle < 2000 && order.size() < 2; ++cycle) {
        for (const auto& response : backend->step(cycle)) order.push_back(response.request_id);
    }
    require(order == std::vector<std::uint64_t>{1, 2}, "read retired before same-address write data");
}

void test_routing_and_writeback(const MemorySystemConfig& config) {
    TensorRegistry tensors;
    const auto& tensor = tensors.allocate_tensor(0, "override_input", {1024}, {1},
                                                 DataType::FP16, MemoryTier::HBF);
    const auto base = tensor.global_base_address;
    MemorySystem memory(128, 128, 0, 4096, 4096, 64, false, config, 1980, tensors);
    int cycle = 0;
    const auto access = [&](int id, std::uint64_t offset, bool write) {
        require(memory.l2()->enqueue_transaction_key(id, {0, offset, base + offset}, write),
                "L2 rejected test access");
        bool done = false;
        for (int limit = 0; limit < 100000 && !done; ++limit) {
            for (int completed : memory.step(++cycle)) {
                require(completed == id, "unexpected L2 completion");
                done = true;
            }
        }
        require(done, "L2 routed access did not complete");
    };
    access(1, 0, false);
    access(2, 0, true);  // Dirty the cached line.
    access(3, 128, false);  // Evicts it; the writeback has no DAG waiter.
    for (int limit = 0; limit < 100000 && !memory.idle(); ++limit) memory.step(++cycle);
    require(memory.idle(), "dirty writeback did not drain");
    const auto stats = memory.statistics(cycle);
    require(stats.hbm.read_requests == 2 && stats.hbm.write_requests == 1 &&
            stats.hbm.request_count == stats.hbm.completed_requests && stats.hbf.request_count == 0,
            "FullHBM override or writeback conservation failed");
    const auto backend = memory.hbm_backend_statistics();
    require(backend.accepted_requests == 3 && backend.completed_requests == 3 &&
            backend.submitted_transactions == 12 && backend.completed_transactions == 12,
            "L2-to-controller conservation failed");
}
}

int main(int argc, char** argv) {
    try {
        require(argc == 2, "expected Ramulator memory profile");
        SimulatorConfig config;
        ArchConfYamlReader(argv[1]).parse(config);
        test_bursts(config.memory.hbm);
        test_controller_backpressure(config.memory.hbm);
        test_arrival_and_ordering(config.memory.hbm);
        test_routing_and_writeback(config.memory);
        std::cout << "Ramulator burst, routing, retry and drain checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
