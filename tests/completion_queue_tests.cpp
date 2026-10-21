#include "memory.h"
#include "simulator.h"
#include <iostream>
#include <random>
#include <stdexcept>

using namespace GTSim;

static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

// Preserve the original scan-and-copy behavior as an independent oracle for
// queue compaction, including public queues with non-monotonic deadlines.
static std::vector<int> reference_step(Memory& memory, int cycle) {
    std::vector<int> completed;
    for (auto* queue : {&memory.memory_queue, &memory.memory_write_queue}) {
        decltype(memory.memory_queue) remaining;
        for (const auto& entry : *queue) {
            if (std::get<2>(entry) <= cycle) {
                const auto& ids = std::get<0>(entry);
                completed.insert(completed.end(), ids.begin(), ids.end());
            } else {
                remaining.push_back(entry);
            }
        }
        *queue = std::move(remaining);
    }
    return completed;
}

static void check_memory_compaction() {
    Memory actual(7, 128, 8), reference(7, 128, 8);
    actual.memory_queue = reference.memory_queue = {{{90, 91}, 128, 10}, {{92}, 128, 2}};
    actual.memory_write_queue = reference.memory_write_queue = {{{93}, 128, 2}};
    std::mt19937 random(42);
    for (int cycle = 0; cycle < 600; ++cycle) {
        if (cycle < 300) {
            for (int attempt = 0; attempt < 3; ++attempt) {
                const bool write = random() % 2;
                const int bytes = 32 * (1 + random() % 16);
                const int conflict = random() % 8;
                const int id = cycle * 3 + attempt + 100;
                require(actual.enqueue(id, bytes, cycle, conflict, write) ==
                        reference.enqueue(id, bytes, cycle, conflict, write), "queue admission changed");
            }
        }
        require(actual.step(cycle) == reference_step(reference, cycle), "completion order/timing changed");
        require(actual.memory_queue == reference.memory_queue &&
                actual.memory_write_queue == reference.memory_write_queue, "pending requests changed");
    }
    require(actual.memory_queue.empty() && actual.memory_write_queue.empty(), "queues did not drain");
}

static void check_l2_completions(int latency) {
    TensorRegistry tensors;
    const auto& tensor = tensors.allocate_tensor(0, "data", {2, 64}, {64, 1}, DataType::FP16);
    L2Cache l2(256, 128, latency, 512, 256, 8, false, tensors);
    const CacheLineKey a{0, 0, tensor.global_base_address};
    const CacheLineKey b{0, 1, tensor.global_base_address + 128};
    require(l2.enqueue_transaction_key(10, a, false) && l2.enqueue_transaction_key(11, a, false), "miss enqueue");
    require(l2.step(0).empty(), "miss completed early");
    auto requests = l2.take_backend_requests();
    require(requests.size() == 1, "same-line misses did not merge");
    const auto respond = [&](const MemRequest& request, int cycle) {
        l2.handle_response({request.request_id, request.global_addr, request.tier,
                            true, 128, MemoryOperation::Read}, cycle);
    };
    respond(requests.front(), 0);
    for (int cycle = 1; cycle <= std::max(1, latency); ++cycle) {
        const auto completed = l2.step(cycle);
        require(completed == (cycle == std::max(1, latency) ? std::vector<int>{10, 11} : std::vector<int>{}),
                "merged response completion phase/order changed");
    }
    const int now = std::max(1, latency) + 1;
    require(l2.enqueue_transaction_key(12, a, false) && l2.enqueue_transaction_key(13, b, false), "hit/miss enqueue");
    require(l2.step(now).empty(), "new hit bypassed completion phase");
    requests = l2.take_backend_requests();
    require(requests.size() == 1, "hit unexpectedly reached backend");
    respond(requests.front(), now);
    for (int cycle = now + 1; cycle <= now + std::max(1, latency); ++cycle) {
        const auto completed = l2.step(cycle);
        require(completed == (cycle == now + std::max(1, latency) ? std::vector<int>{12, 13} : std::vector<int>{}),
                "equal-cycle hit/response order changed");
    }
}

static void check_node_partitioning() {
    DAG dag;
    for (int i = 0; i < 8; ++i) {
        auto* node = new DAGNode(i, "node" + std::to_string(i), "SIMD", "add", i % 4, {}, 0, Tile());
        node->sm_id = i % 3 - 1; // Include the legacy unassigned/broadcast case.
        node->thread_block_id = i / 4;
        dag.add_node(node);
    }
    auto config = make_h100_sxm_config();
    config.num_sms = 2;
    Simulator simulator(&dag, config);
    for (auto* sm : simulator.simulated_gpu->sms) {
        for (auto* sp : sm->sps) {
            for (int warp = 0; warp < simulator.config.warp_count; ++warp) {
                std::vector<DAGNode*> expected;
                for (auto* node : dag.nodes) {
                    if ((node->sm_id == -1 || node->sm_id == sm->sm_id) &&
                        node->warp_id == warp && warp % 4 == sp->subpartition_id)
                        expected.push_back(node);
                }
                require(sp->sp_scheduler->scheduler_warp[warp] == expected, "SM/SP partition order changed");
            }
        }
    }
    for (auto* node : dag.nodes)
        require(simulator.node_id_to_sm[node->id] == (node->sm_id == -1 ? 1 : node->sm_id),
                "broadcast ownership changed");
}

int main() {
    try {
        check_memory_compaction();
        check_l2_completions(0);
        check_l2_completions(7);
        check_node_partitioning();
        std::cout << "Completion queues and DAG partitioning preserve timing and order\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
