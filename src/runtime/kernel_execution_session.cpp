#include "runtime/kernel_execution_session.h"
#include "simulator.h"

namespace GTSim {
void KernelExecutionSession::retire(const Simulator& simulator) {
    if (simulator.completed_node_count != simulator.dag->nodes.size() ||
        !simulator.simulated_gpu->memory_system->idle())
        throw std::logic_error(
            "kernel retirement requires output and memory completion");
    const auto* gpu = simulator.simulated_gpu;
    if (!gpu->pending_remote_reads.empty() ||
        !gpu->pending_remote_writes.empty() ||
        !gpu->forward_to_sram_retry_queue.empty() ||
        !gpu->return_retry_queue.empty())
        throw std::logic_error("kernel retirement has pending remote work");
    for (const auto* sm : gpu->sms) {
        if (!sm->tma->idle() || !sm->sram->memory_queue.empty() ||
            !sm->sram->memory_write_queue.empty() ||
            !sm->tmem_mem->memory_queue.empty() ||
            !sm->tmem_mem->memory_write_queue.empty())
            throw std::logic_error(
                "kernel retirement has pending device operations");
        for (const auto* sp : sm->sps) {
            if (!sp->pending_memory_ops.empty() || !sp->barrier_queue.empty() ||
                !sp->retry_queue_sram.empty() || !sp->retry_queue_l2.empty() ||
                !sp->deferred_tmem_pair_completions.empty())
                throw std::logic_error(
                    "kernel retirement has pending launch-local operations");
            for (const auto* pipeline : sp->pipelines)
                if (!pipeline->pipeline_executing_nodes.empty())
                    throw std::logic_error("pipeline not retired");
        }
    }
}

void KernelExecutionSession::initialize(Simulator& simulator, DAG& dag,
                                        bool replace_schedulers) {
    auto* gpu = simulator.simulated_gpu;
    const auto& config = simulator.config;
    simulator.dag = &dag;
    simulator.completed_node_count = 0;
    dag.build_dependency_graph();
    const int count = static_cast<int>(dag.nodes.size());
    simulator.global_scoreboard.assign(count, 0);
    simulator.node_id_to_sp.assign(count, nullptr);
    simulator.node_id_to_sm.assign(count, -1);
    int warps = replace_schedulers ? 1 : config.warp_count;
    for (const auto* node : dag.nodes)
        warps = std::max(warps, node->warp_id + 1);
    simulator.config.warp_count = warps;
    std::vector<std::vector<DAGNode*>> nodes_by_sm(gpu->sms.size());
    for (auto* node : dag.nodes) {
        if (node->sm_id == -1 && !replace_schedulers)
            for (auto& nodes : nodes_by_sm) nodes.push_back(node);
        else if (node->sm_id >= 0 &&
                 node->sm_id < static_cast<int>(nodes_by_sm.size()))
            nodes_by_sm[node->sm_id].push_back(node);
        else
            throw std::invalid_argument("invalid launch SM binding");
    }
    for (auto* sm : gpu->sms) {
        if (replace_schedulers) {
            sm->all_tb_ids.clear();
            sm->resident_tb_ids.clear();
            sm->tb_to_nodes.clear();
            sm->tb_remaining_count.clear();
            sm->node_id_to_tb_id.clear();
            sm->tb_dispatch_ready.clear();
            sm->waiting_tb_queue = {};
            sm->shared_used = sm->registers_used = sm->warps_used = 0;
            sm->tmem->init_warp_state(warps);
        }
        for (auto* sp : sm->sps) {
            if (replace_schedulers) {
                delete sp->sp_scheduler;
                sp->sp_scheduler =
                    new Scheduler(warps, config.O2_dynamic_allocation,
                                  config.schedule_policy);
                sm->sp_schedulers[sp->subpartition_id] = sp->sp_scheduler;
                sp->sp_scheduler->set_pipelines(&sp->pipelines);
                sp->sp_scheduler->set_tma_unit(sm->tma);
                sp->sp_scheduler->set_tmem_unit(sm->tmem);
                sp->sp_scheduler->set_peer_schedulers(&sm->sp_schedulers);
                sp->sp_register->register_totally_used = 0;
            }
            std::vector<DAGNode*> nodes;
            for (auto* node : nodes_by_sm[sm->sm_id])
                if (node->warp_id % 4 == sp->subpartition_id) {
                    nodes.push_back(node);
                    simulator.node_id_to_sp[node->id] = sp;
                    simulator.node_id_to_sm[node->id] = sm->sm_id;
                }
            sp->sp_scheduler->init_with_nodes(nodes, count, sp->subpartition_id,
                                              &simulator.global_scoreboard);
            sp->set_node_to_sp_map(&simulator.node_id_to_sp);
            sp->set_completed_node_counter(&simulator.completed_node_count);
        }
        sm->cta_resources = dag.cta_resources;
        sm->shared_capacity = config.shared_memory_per_sm_bytes;
        sm->register_capacity = config.register_file_per_sm_bytes;
        sm->warp_capacity = config.max_warps_per_sm;
        sm->initialize_tb_scheduler(nodes_by_sm[sm->sm_id]);
    }
    gpu->set_node_to_sp_map(&simulator.node_id_to_sp);
    gpu->set_node_to_sm_map(&simulator.node_id_to_sm);
}

void KernelExecutionSession::rebind(Simulator& simulator, DAG& dag) {
    retire(simulator);
    simulator.cycle = std::max(simulator.cycle, simulator.memory_drain_cycle);
    initialize(simulator, dag, true);
}
} // namespace GTSim
