#include "arch_conf.h"
#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace GTSim;

// FA2 inference/decode specialization: split-Q warps, Ampere cp.async,
// m16n8k16 warp MMA, register online softmax, and a separate split-KV merge.
// query length is one; the GQA query heads become rows of a packed Q tile.
DAG* build_fa2_a100(int batch, int hq, int hkv, int seq_kv, int dim, int splits,
                    const SimulatorConfig& hardware) {
    constexpr int bm = 64, bn = 64, warps = 4;
    if (batch <= 0 || hq <= 0 || hkv <= 0 || hq % hkv || hq / hkv > bm ||
        seq_kv <= 0 || (dim != 64 && dim != 128 && dim != 256) ||
        splits <= 0 || splits > 128 || splits > (seq_kv + bn - 1) / bn)
        throw std::invalid_argument("invalid FA2 decode dimensions/splits");
    const int group = hq / hkv, jobs = batch * hkv;
    const int blocks = (seq_kv + bn - 1) / bn;
    const int blocks_per_split = (blocks + splits - 1) / splits;
    const CTAResources resources{2 * dim * (bm + 2 * bn),
                                4 * bm * (dim + bn) + warps * 32 * 16 * 4, warps};
    if (resources.shared_bytes > hardware.shared_memory_per_cta_bytes ||
        resources.shared_bytes > hardware.shared_memory_per_sm_bytes ||
        resources.register_bytes > hardware.register_file_per_sm_bytes ||
        resources.register_bytes > warps * 32 * 255 * 4)
        throw std::invalid_argument("FA2 CTA exceeds hardware resources");
    auto owner = std::make_unique<DAG>();
    DAG* dag = owner.get();
    auto alloc = [&](int id, std::string name, int r, int c, DataType dtype) {
        dag->tensor_registry.allocate_tensor(id, name,
            {static_cast<std::uint64_t>(r), static_cast<std::uint64_t>(c)},
            {static_cast<std::uint64_t>(c), 1}, dtype, MemoryTier::HBM);
    };
    auto add = [&](int tb, int lane, std::string name, std::string pipe,
                   std::string op, Tile tile, const std::vector<int>& deps,
                   int tensor = -1, DataType dtype = DataType::FP32) {
        int id = static_cast<int>(dag->nodes.size());
        auto* n = new DAGNode(id, "FA2_" + std::to_string(tb) + "_" + name + "_" + std::to_string(id),
            pipe, op, tb * warps + lane, deps, 0, tile, dtype);
        n->thread_block_id = tb;
        n->sm_id = tb % hardware.num_sms;
        n->matrix_id = tensor;
        // CUTLASS SM80 uses 128-bit vector copies and swizzled shared tiles.
        if (n->op_type == OpType::CP_DRAM2SRAM || n->op_type == OpType::LD_SRAM2REG)
            n->memory_access_granularity_bytes = 16;
        if (n->op_type == OpType::LD_SRAM2REG) n->layout = Layout::Swizzled;
        if (tensor >= 0) n->matrix_leading_dim = static_cast<int>(dag->tensor_registry.at(tensor).shape[1]);
        dag->add_node(n);
        return id;
    };
    // ldmatrix.x4 transfers four 8x8 BF16 matrices (512 bytes/warp).
    // Keeping these as instruction-sized fragments also avoids repeatedly
    // expanding one huge SRAM tile for every memory sub-operation.
    auto shared_load = [&](int tb, int w, std::string name, int rows, int cols,
                           const std::vector<int>& deps) {
        std::vector<int> reads;
        for (int r=0;r<rows;r+=16) for(int c=0;c<cols;c+=16)
            reads.push_back(add(tb,w,name,"LD","ld.sram2reg",Tile(r,c,16,16),deps,-1,DataType::BF16));
        // A zero-latency scoreboard join, not a CTA/group synchronization.
        return add(tb,w,"ldmatrix_ready","SIMD","barrier.registers",Tile(0,0,1,1),reads);
    };
    // Tensor IDs per KV head: packed Q, K, V, O, then FP32 partial O/LSE pairs.
    const int stride = 5 + 2 * splits;
    std::vector<int> all_main_stores;
    for (int job = 0; job < jobs; ++job) {
        int base = job * stride;
        alloc(base, "PackedQ", group, dim, DataType::BF16);
        alloc(base+1, "K", seq_kv, dim, DataType::BF16);
        alloc(base+2, "V", seq_kv, dim, DataType::BF16);
        alloc(base+3, "O", group, dim, DataType::BF16);
        alloc(base+4, "FinalLSE", group, 1, DataType::FP32);
        if (splits > 1) for (int split = 0; split < splits; ++split) {
            alloc(base+5+2*split, "PartialO", group, dim, DataType::FP32);
            alloc(base+6+2*split, "PartialLSE", group, 1, DataType::FP32);
        }
        for (int split = 0; split < splits; ++split) {
            int tb = job * splits + split;
            dag->cta_resources[tb] = resources;
            std::vector<int> q_copies;
            for (int w = 0; w < warps; ++w) {
                int live = std::max(0, std::min(16, group-w*16));
                if (live) q_copies.push_back(add(tb,w,"Q_cp_async","LD","cp.dram2sram",
                    Tile(w*16,0,live,dim),{},base,DataType::BF16));
            }
            std::array<int,4> qreg{}, previous_o{}, previous_l{}, previous_m{};
            std::vector<int> old_k_read, old_v_read;
            for (int w = 0; w < warps; ++w) {
                qreg[w] = shared_load(tb,w,"Q_ldmatrix",16,dim,q_copies);
                previous_o[w] = add(tb,w,"O_zero","SIMD","fill",Tile(0,0,16,dim),{});
                previous_l[w] = add(tb,w,"L_zero","SIMD","fill",Tile(0,0,16,1),{});
                previous_m[w] = add(tb,w,"M_neg_inf","SIMD","fill",Tile(0,0,16,1),{});
            }
            const int lo = split * blocks_per_split;
            const int hi = std::min(blocks, (split+1)*blocks_per_split);
            // FA2 visits KV tiles backwards. Reuse completion edges implement
            // cp.async wait / CTA synchronization without Hopper TMA.
            for (int block = hi-1; block >= lo; --block) {
                const int offset = block * bn, live_kv = std::min(bn, seq_kv-offset);
                std::vector<int> copies_k, copies_v;
                for (int w = 0; w < warps; ++w) {
                    int count = std::max(0,std::min(16,live_kv-w*16));
                    if (!count) continue;
                    copies_k.push_back(add(tb,w,"K_cp_async","LD","cp.dram2sram",
                        Tile(offset+w*16,0,count,dim),old_k_read,base+1,DataType::BF16));
                    copies_v.push_back(add(tb,w,"V_cp_async","LD","cp.dram2sram",
                        Tile(offset+w*16,0,count,dim),old_v_read,base+2,DataType::BF16));
                }
                std::vector<int> next_k_read, next_v_read;
                for (int w = 0; w < warps; ++w) {
                    int kr = shared_load(tb,w,"K_ldmatrix",bn,dim,[&]{auto d=copies_k;d.push_back(previous_o[w]);return d;}());
                    int vr = shared_load(tb,w,"V_ldmatrix_transpose",bn,dim,[&]{auto d=copies_v;d.push_back(previous_o[w]);return d;}());
                    next_k_read.push_back(kr); next_v_read.push_back(vr);
                    std::array<int,bn/8> qk{}; qk.fill(-1);
                    // Independent N fragments carry independent accumulators.
                    for (int k = 0; k < dim; k += 16) for (int n = 0; n < bn/8; ++n) {
                        std::vector<int> deps{qreg[w],kr};
                        if (qk[n]>=0) deps.push_back(qk[n]);
                        qk[n] = add(tb,w,"QK_mma","Tensor","mma.sync.m16n8k16",
                            Tile(w*16,n*8,16,8),deps);
                    }
                    std::vector<int> scores(qk.begin(),qk.end());
                    int mask = add(tb,w,"Scale_mask_scores","SIMD","scale_mask",Tile(0,0,16,bn),scores);
                    int mx = add(tb,w,"Row_max","SIMD","row_max",Tile(0,0,16,bn),{mask});
                    int shmax = add(tb,w,"Shuffle_max","SHFL","shfl_max",Tile(0,0,16,4),{mx});
                    int new_m = add(tb,w,"Online_max","SIMD","max",Tile(0,0,16,1),{shmax,previous_m[w]});
                    int alpha = add(tb,w,"Rescale_alpha","SFU","exp2",Tile(0,0,16,1),{new_m,previous_m[w]});
                    int shift = add(tb,w,"Subtract_max","SIMD","subtract",Tile(0,0,16,bn),{mask,new_m});
                    int prob = add(tb,w,"Exp_scores","SFU","exp2",Tile(0,0,16,bn),{shift});
                    int sum = add(tb,w,"Row_sum","SIMD","row_sum",Tile(0,0,16,bn),{prob});
                    int shsum = add(tb,w,"Shuffle_sum","SHFL","shfl_sum",Tile(0,0,16,4),{sum});
                    int new_l = add(tb,w,"Online_sum","SIMD","fma",Tile(0,0,16,1),{shsum,previous_l[w],alpha});
                    int scale_o = add(tb,w,"Rescale_O","SIMD","mul",Tile(0,0,16,dim),{previous_o[w],alpha});
                    int cast_p = add(tb,w,"P_to_bf16","SIMD","cast",Tile(0,0,16,bn),{prob});
                    std::vector<int> pv(dim/8,scale_o);
                    for (int k = 0; k < bn; k += 16) for (int n = 0; n < dim/8; ++n)
                        pv[n] = add(tb,w,"PV_mma","Tensor","mma.sync.m16n8k16",Tile(w*16,n*8,16,8),{cast_p,vr,pv[n]});
                    previous_o[w] = add(tb,w,"PV_complete","SIMD","identity",Tile(0,0,1,1),pv);
                    previous_m[w] = new_m; previous_l[w] = new_l;
                }
                old_k_read = next_k_read; old_v_read = next_v_read;
            }
            std::vector<int> cta_done(previous_o.begin(),previous_o.end());
            cta_done.insert(cta_done.end(),previous_l.begin(),previous_l.end());
            for (int w = 0; w < warps; ++w) {
                int live = std::max(0,std::min(16,group-w*16));
                if (!live) continue;
                int norm = add(tb,w,lo>=hi?"Empty_split_zero":"Final_normalize",lo>=hi?"SIMD":"SFU",lo>=hi?"fill":"divide",Tile(0,0,16,dim),cta_done);
                int out = splits>1 ? base+5+2*split : base+3;
                std::vector<int> deps{norm};
                if (splits==1) deps={add(tb,w,"O_to_bf16","SIMD","cast",Tile(0,0,live,dim),deps)};
                all_main_stores.push_back(add(tb,w,"Store_partial_O","ST","st.reg2dram",Tile(w*16,0,live,dim),deps,out,splits>1?DataType::FP32:DataType::BF16));
                if (splits>1) {
                    int log_l = add(tb,w,lo>=hi?"Empty_LSE_neg_inf":"Log_L",lo>=hi?"SIMD":"SFU",lo>=hi?"fill":"log",Tile(0,0,live,1),{previous_l[w]});
                    int lse = add(tb,w,"LSE","SIMD","add",Tile(0,0,live,1),{log_l,previous_m[w]});
                    all_main_stores.push_back(add(tb,w,"Store_LSE","ST","st.reg2dram",Tile(w*16,0,live,1),{lse},out+1));
                }
            }
        }
    }
    // Separate kernel: every combine CTA waits for every main CTA, matching
    // a CUDA stream kernel boundary; each warp handles one live query head.
    int padded_splits=1; while(padded_splits<splits)padded_splits*=2;
    if (splits>1) for (int job=0; job<jobs; ++job) {
        int base=job*stride, tb=jobs*splits+job;
        dag->cta_resources[tb]={padded_splits * 5 * 4,32*1024,warps};
        for (int row=0; row<group; ++row) {
            int w=row%warps;
            std::vector<int> lses;
            for (int s=0;s<splits;++s) {
                lses.push_back(add(tb,w,"Load_LSE","LD","ld.dram2reg",Tile(row,0,1,1),all_main_stores,base+6+2*s));
            }
            int mx=add(tb,w,"Combine_max","SIMD","max",Tile(0,0,1,padded_splits),lses);
            int shmax=mx;
            for(int i=0;i<5;++i) shmax=add(tb,w,"Combine_max_shuffle","SHFL","shfl",Tile(0,0,1,32),{shmax});
            int sub=add(tb,w,"Combine_subtract","SIMD","sub",Tile(0,0,1,padded_splits),{shmax});
            int exp=add(tb,w,"Combine_exp","SFU","exp2",Tile(0,0,1,padded_splits),{sub});
            int sum=add(tb,w,"Combine_sum","SIMD","sum",Tile(0,0,1,padded_splits),{exp});
            int shsum=sum;
            for(int i=0;i<5;++i) shsum=add(tb,w,"Combine_sum_shuffle","SHFL","shfl",Tile(0,0,1,32),{shsum});
            int logsum=add(tb,w,"Combine_logsum","SFU","log",Tile(0,0,1,1),{shsum});
            int final_lse=add(tb,w,"Combine_final_LSE","SIMD","add",Tile(0,0,1,1),{logsum,shmax});
            add(tb,w,"Combine_store_LSE","ST","st.reg2dram",Tile(row,0,1,1),{final_lse},base+4);
            int reciprocal=add(tb,w,"Combine_inverse_sum","SFU","rcp",Tile(0,0,1,1),{shsum});
            int weights=add(tb,w,"Combine_weights","SIMD","mul",Tile(0,0,1,padded_splits),{exp,reciprocal});
            int stored_weights=add(tb,w,"Combine_weights_to_shared","ST","st.reg2sram",Tile(0,0,1,padded_splits),{weights});
            int acc=-1;
            for (int s=0;s<splits;++s) {
                // FA2 uses one reusable Oaccum register fragment in its split
                // loop, not one simultaneously live fragment per partition.
                std::vector<int> load_deps{stored_weights}; if(acc>=0)load_deps.push_back(acc);
                int partial=add(tb,w,"Load_partial_O","LD","ld.dram2reg",Tile(row,0,1,dim),load_deps,base+5+2*s);
                int weight=add(tb,w,"Load_shared_weight","LD","ld.sram2reg",Tile(0,0,1,1),load_deps);
                std::vector<int> deps{partial,weight}; if(acc>=0)deps.push_back(acc);
                acc=add(tb,w,"Combine_weighted_sum","SIMD","fma",Tile(0,0,1,dim),deps);
            }
            int cast=add(tb,w,"Combine_cast","SIMD","cast",Tile(0,0,1,dim),{acc});
            add(tb,w,"Combine_store","ST","st.reg2dram",Tile(row,0,1,dim),{cast},base+3,DataType::BF16);
        }
    }
    return owner.release();
}

#ifndef GTSIM_FA2_NO_MAIN
int main(int argc, char** argv) {
    if (argc!=9) {
        std::cerr << "Usage: gtsim_fa2 architecture.yaml batch heads_q heads_kv seq_q seq_kv head_dim splits\n";
        return 1;
    }
    try {
        SimulatorConfig config;
        ArchConfYamlReader(argv[1]).parse(config);
        int b=std::stoi(argv[2]),hq=std::stoi(argv[3]),hkv=std::stoi(argv[4]);
        if (std::stoi(argv[5])!=1) throw std::invalid_argument("FA2 decode specialization requires seq_q=1");
        int kv=std::stoi(argv[6]),dim=std::stoi(argv[7]),splits=std::stoi(argv[8]);
        std::unique_ptr<DAG> dag(build_fa2_a100(b,hq,hkv,kv,dim,splits,config));
        std::cout << "=== Ampere FA2 decode (warp MMA, cp.async, packed GQA) ===\n"
                  << "Generated DAG with " << dag->nodes.size() << " nodes\n"
                  << "Main grid: 1 x " << splits << " x " << b*hkv << "; 128 threads/CTA\n";
        config.silence_mode=true;
        config.workload_type=WorkloadType::FlashDecoding;
        Simulator sim(dag.get(),config);
        bool complete=sim.run(100000000,true);
        int main_end=0,combine_start=-1;
        for(auto* n:dag->nodes) {
            if(n->thread_block_id<b*hkv*splits)main_end=std::max(main_end,n->end);
            else if(n->issue_cycle>=0 && (combine_start<0 || n->issue_cycle<combine_start))combine_start=n->issue_cycle;
        }
        std::cout << "FA2 main complete cycle: " << main_end << '\n'
                  << "FA2 combine first issue cycle: " << combine_start << '\n';
        return complete?0:3;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 2;}
}
#endif
