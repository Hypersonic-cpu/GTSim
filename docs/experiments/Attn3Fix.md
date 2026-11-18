# GTSim — 通用 GQA Decode 的 PackGQA + Split-KV + Tensor Core 实施 Plan

> 目标代码：`Hypersonic-cpu/GTSim` 的 `sim-validation` 分支。这是交给 Coding Agent 的**具体实施规范**，不是允许 Agent 自行设计 Attention 数学算法的开放任务。**新 Operator 必须面向任意合法的 GQA 模型维度；Qwen3-235B-A22B 仅作为实例化参数和回归 workload。** 不得直接替换/删除现有 legacy operator；新增路径必须可切换、可回归。所有性能数字必须重新运行得出，禁止声称已经验证本草案的 FlowIR 可被当前 parser 直接编译。

## 0. 标准参考（按优先级）

1. **官方 CUDA Hopper FA3（最终性能/执行路径标准）**
   - `https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_api.cpp`：`get_pack_gqa`、`get_num_splits`；Split 时启用 PackGQA。
   - `https://github.com/Dao-AILab/flash-attention/blob/main/hopper/pack_gqa.h`：把多个同一 KV head 的 query heads 映射到 GEMM 的 M 维。
   - `https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_fwd_launch_template.h`：以 KV heads、packed query M blocks、split 数建立 scheduler。
   - `https://github.com/Dao-AILab/flash-attention/blob/main/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp`：SM90 TMA、GMMA、异步 pipeline。
   - `https://github.com/Dao-AILab/flash-attention/blob/main/hopper/benchmark_split_kv.py`：low-batch/seq_q=1 benchmark，并分别扫描 PackGQA 和 Split 数。
2. **官方 TileLang 可读算法参考**
   - `https://github.com/tile-ai/tilelang/blob/main/examples/flash_decoding/example_gqa_decode.py`：其 `flashattn_gqa_decode_split` 中的 grouped Q、每 split 共享 K/V、两次 GEMM、online row softmax、partial O/LSE 和 combine。
   - 说明：该范例默认 FP16，并不等于 Hopper FA3 的逐指令 CUDA 实现；在本仓库中采用 BF16 Q/K/V、FP32 m/l/Oacc、FP32 partial O/LSE。不要直接照抄其数值精度和静态 `seqlen_kv // num_split` 分割方式。
3. **GTSim 已有组件**
   - `frontend/lib/fa3_reference.{algoir,flowir}`：现存 `flow.matmul @qk` / `flow.matmul @pv` + Hopper native MATRIX 的**语法和 lowering 模板**。
   - `frontend/lib/fa3_decode_split.{algoir,flowir}`：旧的 per-query SIMT decode 保留作 legacy。
   - `frontend/lib/fa3_combine.{algoir,flowir}`：旧 split-major 布局不可无条件复用，须增加新 combine 或做 layout dispatch。
   - `frontend/asm/qwen3/flash-accel/one_block.progir`：本实验真实启动路径。

## 1. 通用 GQA 语义与算法（数学 contract 不变）

**参数绑定边界：** `.algoir/.flowir` 的主体仅表达与模型无关的计算和符号维度。`B`（batch）、`HQ`（query heads）、`HKV`（KV heads）、`D`（head dim）、`CONTEXTS`（每个请求的已有 cache 长度）、`QKV_DTYPE`/`OUTPUT_DTYPE`、`R`（每 CTA 有效 packed Q heads）、`S`（split 数）、`TILE`/`BN`（KV tile）由调用 Program、CLI 或 workload 配置传入；它们**不能以特定模型常数写死在库文件内**。`G=HQ/HKV`、`CTAS`、tile 尾长、各 tensor 形状和地址偏移均为 `static.let` 推导。GPU 架构/后端配置提供 `TARGET`、`NUM_SMS`、物理 MMA fragment 形状、warp/warpgroup、TMA 能力、`STAGES`、SMEM/寄存器上限；最终在绑定/降低阶段实例化。`0/1`、`sqrt(D)` 等真正的算法常量不属于 magic number。

**GTSim 的语义边界：** 当前模拟器主要生成/计时数据流和依赖图，**并不数值执行 Attention，也未证明浮点 Cast 的数值转换正确**。保留以下 Online Softmax 公式作为独立数学 oracle 和数据依赖 contract；FlowIR 的正确性目标是逻辑 Q/K/V/输出访存覆盖、Tensor Core QK/PV 矩阵节点、必要前后继、stage slot 生命周期，而不是在 GTSim 中计算出 FP32/BF16 数值。

Qwen3-235B-A22B 的 `B=1,HQ=64,HKV=4,D=128,QKV_DTYPE=bf16,CONTEXTS=[10000|100000|200000]` 只放进**模型 JSON/回归配置**，不进入通用 `.flowir`。现有 Program 先 append 当前 token，因此通用约定 `L_b=CONTEXTS[b]+1`（若其他模型使用不同 cache ABI，应在 Program 的 adapter 参数中明确，而非改写 Operator 数学）。继续兼容 GTSim 现有 KV packed-flat 布局，不在本次修复中修改其地址 ABI：

- `Q[(b*HQ+h), d]`（`QROWS=B*HQ`）。
- `CACHE_ROWS=(sum(CONTEXTS)+B)*HKV`。
- `CACHE_BASE(b,kvh)=(prefix_sum(CONTEXTS,b)+b)*HKV+kvh*L_b`。
- `K[CACHE_BASE+j,d]`、`V[CACHE_BASE+j,d]`，`0<=j<L_b`。
- Q head `h` 对应 `kvh=floor(h/G)`；**同一个 KV head 的每个 packed Q-block 共享一份 K/V tile**，不应对 block 内各 Q head 重复发起 K/V load。`G` 和 `R` 均取自参数，绝不假定 GQA group 固定为 16。
- 对 decode 的单个新 query token，因果可见范围是 `0..L_b-1`；不得把历史的所有 query token 当作计算目标，也不得读取 `CONTEXTS[b]+1` 以外的元素。

### 1.1 需要实现的完整 Online Softmax（我已确定公式；不要改写成不等价的逐 tile softmax）

对任意 `b, kvh, grouped query row r, split s`：

```
L = CONTEXTS[b] + 1
begin_s = floor(L*s/S)
end_s   = floor(L*(s+1)/S)
Q_g[r,d] = Q[b*HQ + kvh*G + group_tile*R + r, d], 0<=r<R_valid
m[r] = -inf ; ell[r] = 0.f ; acc[r,d] = 0.f
for j0 in range(begin_s, end_s, BN):
    J = [j0, min(end_s, j0+BN))
    scores[r,j] = sum_d(fp32(Q_g[r,d]) * fp32(K[b,kvh,j,d])) / sqrt(D)
    scores[r,j>=end_s] = -inf  # 对最后一块的越界列掩码
    tile_m[r] = max_j(scores[r,j])
    m_next[r] = max(m[r], tile_m[r])
    alpha[r] = 0 if m[r] == -inf else exp(m[r] - m_next[r])
    p[r,j] = exp(scores[r,j] - m_next[r])
    # 注意：p 是未归一化的局部权重；不能在每个 tile 内独立归一化
    ell_next[r] = alpha[r]*ell[r] + sum_j(fp32(p[r,j]))
    acc_next[r,d] = alpha[r]*acc[r,d] + sum_j(fp32(cast_to_QKV_DTYPE(p[r,j]))*fp32(V[b,kvh,j,d]))
    m,ell,acc = m_next,ell_next,acc_next
partial[r,d] = acc[r,d] / ell[r]                   # FP32
lse[r] = m[r] + ln(ell[r])                         # FP32 natural log
```

边界：本阶段要求所有 split 非空 `1 <= S <= min_b(L_b)`。不得使用 `-inf - (-inf)`；第一轮 `alpha=0`。逻辑上不存在的 padded Q rows 不得写 HBM，也不得计入有效归约。真正空 split 的支持只能在后续增加 `LSE=-inf, Partial=0` 语义及其 combine 掩码之后放开。

`sum(p)` 在数学 oracle 中以 FP32 p 计算；`P@V` 的输入 dtype 应与选定 Tensor Core 路径（由 `QKV_DTYPE` 决定）一致。对原 BF16 配置可保留 `prob_cast`，但它在 GTSim 中仅是**带有依赖/成本的转换节点**，不能当作已经测试过的数值转换。若编译/后端不支持当前转换，可降低成普通 elementwise/identity-cost node，或在显式证明不会丢失任何数据依赖和 dtype 元数据后消去；无论如何必须保留 `probability → PV` 的依赖与合法的矩阵输入类型。建立 **FP32 理想 oracle** 与 **按 QKV_DTYPE 转换概率的 oracle**，数值正确性在独立参考程序中验证，不能用 GTSim timing 结果代替。若使用 `exp2`，令 `scaled_log2 = scores * log2(e)`，整个递推都在对应 base-2 max 域内，LSE/Combine 也必须保持一致；本计划统一采用 `exp` + 自然对数，更容易与当前库对齐。**不得混用 log2 LSE 与自然指数 Combine**。

### 1.2 Split 合并公式（全局唯一归一化）

对每个真实 `(b,h)`：`M=max_s(LSE_s)`，`w_s=exp(LSE_s-M)`，`denom=sum_s w_s`，`O[b,h,d]=cast_to_OUTPUT_DTYPE(sum_s(w_s*Partial_s[b,h,d])/denom)`。保留 FP32 LSE 和 Partial 直到最终输出。禁止：先简单平均 Partial；将各 split 内已经归一化的 O 相加而不加权；将 `exp` 与 `log2` 混用。

## 2. 新 Packed CTA 映射与 HBM layout（符号化实现）

引入 `R`=每 CTA 打包的 **Q head 数上限**，由调度参数传入，不能在 Operator 内限制为 `4/8/16`。通用约束 `HQ>0, HKV>0, HQ%HKV==0, 1<=R<=G`；`QB=ceildiv(G,R)`，末尾 block 以 `R_VALID=min(R,G-qblock*R)` 处理，因此即使 `G%R!=0`，也不得越过当前 KV head。对 Qwen G=16 可选择 R=4/8/16 作为**实验配置**，不是 FlowIR 的类型/形状限制。

```
CTAS = B * HKV * QB * S
split   = cta % S
qblock  = (cta // S) % QB
R_valid = min(R, G - qblock*R)
kvhead  = (cta // (S*QB)) % HKV
request = cta // (S*QB*HKV)
head0   = kvhead*G + qblock*R
L       = CONTEXTS[request]+1
start   = floor(L*split/S)
end     = floor(L*(split+1)/S)
```

**输出布局：通用 split-major + head-contiguous**，保持 `PART_ROWS=B*HQ*S` 和 `Partial: PART_ROWS x D, LSE: PART_ROWS x 1` 的 tensor ABI，但改变其行地址含义：

```
partial_row(b,s,h) = (b*S + s)*HQ + h
partial_base(cta)  = (request*S + split)*HQ + head0
# 一个 CTA 一次仅存 R_valid 个连续有效 head rows；每个 split 的同一 b/h 唯一反查
```

这个设计避免旧布局 `((b*HQ+h)*S+s)` 导致每个 CTA 的 R 个头间隔 `S` 行、必须做 R 次散列 store。**相应的 combine load 必须改**：旧 `@Partial[%cta*S+%split,0]` 必须变成 `@Partial[(%REQUEST*%S+%split)*%HQ+%HEAD,0]`；LSE 同理。绝不允许只改 writer 不改 reader！

**实例而非通用参数：**Qwen 的 `B=1,HQ=64,HKV=4,G=16,R=16,S=4` 只有 16 个 CTA（旧方案为 256），可能闲置 H100 的 SM；因此 `R/S/BN` 必须在 Program/benchmark 层联合调参，而不是写在通用库里。

## 3. 新算子与需要更改的文件（优先追加，保留旧实现）

1. 新增 `frontend/lib/fa3_gqa_decode_split.algoir`：数学 contract，复用 `fa3_reference.algoir` 的 `@qk` 与 `@pv` **matrix** op 定义，而不是复制 legacy `qk_product + qk_sum` / `pv_product + pv_sum`；增加上述 `m,ell,acc` 递推与 `@lse`，`@contract_Partial/@contract_LSE` 使用新的 split-major 行地址。参数 `%B,%HQ,%HKV,%D,%S,%TILE,%R,%NUM_SMS,%CONTEXTS` 以及由调用者/目标类型系统绑定的 dtype；派生维度由 `static.let` 计算。
2. 新增 `frontend/lib/fa3_gqa_decode_split.flowir`：实现下节 IR 设计。必须有 `flow.matmul @qk`、`flow.matmul @pv`；QK 与 PV 的 work/fragment 必须采用 SM90 matrix 路径，不能被 `nvidia/vector.py` silently lowered 成 SIMD Multiply+Reduce。
3. 新增 `frontend/lib/fa3_gqa_combine.algoir` 和 `.flowir`：复制现有 combine 的 FP32 max/LSE/weights/numerator/denominator 算法，**仅改输入布局**，避免破坏 legacy combine。
4. `frontend/asm/qwen3/flash-accel/one_block.progir` 和 `frontend/asm/qwen3/one_block.progir`：仅作为**Qwen 调用示例**增加 `PACK_GQA`、`R` 等实参和静态条件：`PACK_GQA && S>1` 用新的 packed split + packed combine；否则调用现有的 `fa3_decode{,_split}` 与原始 combine；旧配置仍复现原曲线。新增 `S=1` 的 packed 直存 kernel 可作为第二阶段，不要用 `S>1` kernel 无代价冒充 `S=1`。
5. `frontend/tools/run_qwen_simulations.py`：作为一个可用的测试入口，增加 `--pack-gqa 0|1`、`--gqa-block-heads INT`、`--attention-splits N`、`--attention-kv-tile N`；**不允许借 `--gemm-bm/bn/bk` 修改 Attention**。保留所有参数到 provenance，并按精确哈希区分实验。
6. 原先 `frontend/src/compile/nvidia/hopper.py`, `nvidia/skinny.py`, `expand_native.py`, `parser/`：只在**符号参数绑定、矩阵 lowering、masked 真实行加载、stage-slot 映射**确有必要时修改；每项补测试，不随意改核心模拟器的吞吐/延迟参数。如果 parser 当前只接受 literal `schedule_config` / buffer size / dtype / target，优先在已有 AST/静态 Resolver 中支持符号表达式，或者在调用时结构化实例化并生成目标专用 ArchIR；**不得为了让 parser 通过而将 Qwen/H100 常量回填成新库的固定 literal**。
7. 新增 `frontend/tests/test_fa3_gqa_decode.py`、`frontend/tests/test_fa3_gqa_layout.py`、`frontend/tools/qwen_fa3_oracle.py`（文件名可合理调整）及独立 per-kernel benchmark 驱动。

### 3.1 Cast 现状与处理规则（已查源码）

当前 GTSim **存在 Cast 的时序指令路径**，无需删除：`frontend/src/compile/nvidia/vector.py` 中 `OperationClass.CONVERT → NativeOpcode.CAST`，`src/execution_ir/kernel_graph_builder.cpp` 注册 `"simd.cast"` 为 `"SIMD"`，且 `frontend/lib/fa3_reference.{algoir,flowir}` 已使用 `@prob_cast`/`@cast`。因此，优先**保留**现有 Cast 节点以维持依赖和合理的通用 elementwise 开销；不要假设其数值行为已被验证，也不要求补真正的数值仿真。

若新 dtype/后端的 CAST 路径确实不可用，按 `convert → generic elementwise/identity-cost →（有严格依赖保证时）省略转换节点` 的顺序降级。**两次 GEMM 必须仍产生真正 MATRIX/WGMMA 原生指令，不能把 QK/PV 替换成 SIMD multiply/reduce**。静态类型与输出 store 的 element-size/地址计算仍要正确；不能通过删 Cast 省略必须的 `probability → PV` 边或改变 HBM 字节计数。


## 4. FlowIR 的具体翻译骨架（按此写完整 .flowir）

**注意：下面包含完整的映射和主要算子/依赖算法，但不是已经经当前 parser 测试的可直接粘贴成品。** Agent 必须以 `fa3_reference.flowir` 已有 `flow.matmul`、`iterate<%executor>`、`flow.dependency_rule` 为语法模板，在参数绑定完成后进行 warp/fragment 物理展开。下方属于**符号化 FlowIR 设计草案**：其中 `%TARGET`、dtype、`schedule_config` 和 buffer size 等表达式可能需要最小 parser/AST/Resolver 支持；不得为了绕过解析而写死测试模型的参数。不可把矩阵路径降级为旧 vector 计算。

```text
flow.kernel @fa3_gqa_decode_split_flow schedule<explicit> target<%TARGET> {
  static.param %B : index
  static.param %HQ : index
  static.param %HKV : index
  static.param %D : index
  static.param %S : index
  static.param %R : index
  static.param %TILE : index
  static.param %NUM_SMS : index
  static.param %CONTEXTS : index_list
  # 以下架构/调度参数在调用方或 target/backend 上绑定（如当前语法不支持，由 Resolver/AST 最小扩展）
  static.param %TARGET : target
  static.param %QKV_DTYPE : dtype
  static.param %STAGES : index
  static.param %WARPS : index
  static.param %CONSUMER_WARPS : index
  static.param %PRODUCER_WARP : index
  static.param %WARPGROUP : index
  static.param %SMEM_SLOTS : index
  static.param %SMEM_BYTES_PER_SLOT : index
  static.let %G = floordiv(%HQ,%HKV)
  static.let %QB = ceildiv(%G,%R)
  static.let %CTAS = %B*%HKV*%QB*%S
  static.let %QROWS = %B*%HQ
  static.let %CACHE_ROWS = (sum(%CONTEXTS)+%B)*%HKV
  static.let %PART_ROWS = %B*%HQ*%S
  static.let %SPLIT = %cta mod %S
  static.let %QBLOCK = floordiv(%cta,%S) mod %QB
  static.let %R_VALID = min(%R,%G-%QBLOCK*%R)
  static.let %KVHEAD = floordiv(%cta,%S*%QB) mod %HKV
  static.let %REQUEST = floordiv(%cta,%S*%QB*%HKV)
  static.let %HEAD0 = %KVHEAD*%G+%QBLOCK*%R
  static.let %QBASE = %REQUEST*%HQ+%HEAD0
  static.let %LENGTH = %CONTEXTS[%REQUEST]+1
  static.let %START = floordiv(%LENGTH*%SPLIT,%S)
  static.let %END = floordiv(%LENGTH*(%SPLIT+1),%S)
  static.let %VALID = min(%TILE,%END-%START-%k*%TILE)
  static.let %CACHE_BASE = (prefix_sum(%CONTEXTS,%REQUEST)+%REQUEST)*%HKV + %KVHEAD*%LENGTH
  static.let %PART_BASE = (%REQUEST*%S+%SPLIT)*%HQ+%HEAD0
  flow.tensor @Q : tensor<QROWSxDxQKV_DTYPE, row_major>
  flow.tensor @K : tensor<CACHE_ROWSxDxQKV_DTYPE, row_major>
  flow.tensor @V : tensor<CACHE_ROWSxDxQKV_DTYPE, row_major>
  flow.tensor @Partial : tensor<PART_ROWSxDxfp32, row_major>
  flow.tensor @LSE : tensor<PART_ROWSx1xfp32, row_major>
  flow.schedule_config tile(RxTILExD) stages(STAGES) warps(WARPS)
      consumer_warps(CONSUMER_WARPS) producer_warp(PRODUCER_WARP) warpgroup(WARPGROUP)
  # WGMMA physical (m,n,k) geometry is selected in SM90 mapping/lowering, NOT fixed here.
  flow.loop @cta %cta = 0 to %CTAS step 1
  flow.loop @wave %wave = 0 to 1 step 1
  flow.executor %producer role<producer> organization<independent>
  flow.executor %matrix role<matrix> organization<cooperative>
  # SMEM_BYTES_PER_SLOT derives from symbolic Q/K/V layouts, stage count, target alignment/barriers.
  flow.buffer %scratch slots(SMEM_SLOTS) access<read_write> : bytes_per_slot<SMEM_BYTES_PER_SLOT>
  # 以下为有完整数学运算与依赖的实现草案；物理 WGMMA padded rows
  # 的 fragment 映射仍需按 fa3_reference 的合法 iterate / native mask 展开。
  # 真实 Q 仅有 R_VALID 行；按 target 的 physical MMA M-tile 做寄存器/SMEM padding，禁止额外 HBM load。
  static.if <%k>=%STAGES> {
    flow.dependency_rule @reuse_k from[@qk.complete]
       at[%wave=%wave,%k=%k-%STAGES] select<all>
    flow.dependency_rule @reuse_v from[@pv.complete]
       at[%wave=%wave,%k=%k-%STAGES] select<all>
  }
  static.if <%k==0> {
    flow.dependency_rule @state_m from[@max_init.complete] at[%wave=%wave] select<all>
    flow.dependency_rule @state_l from[@sum_init.complete] at[%wave=%wave] select<all>
    flow.dependency_rule @state_o from[@out_init.complete] at[%wave=%wave] select<all>
  }
  static.if <%k>0> {
    flow.dependency_rule @state_m from[@max_new.complete]
        at[%wave=%wave,%k=%k-1] select<all>
    flow.dependency_rule @state_l from[@sum_new.complete]
        at[%wave=%wave,%k=%k-1] select<all>
    flow.dependency_rule @state_o from[@pv.complete]
        at[%wave=%wave,%k=%k-1] select<all>
  }
  flow.dependency_rule @last from[@pv.complete,@sum_new.complete,@max_new.complete]
      at[%wave=%wave,%k=ceildiv(%END-%START,%TILE)-1] select<all>
  %query, !q_ready = flow.load_async @query @Q[%QBASE,0]<R_VALIDxD>
       by %producer requires<bulk_copy_exact> : tensor<R_VALIDxDxQKV_DTYPE> scope<prologue>
  %max_init = flow.arithmetic @max_init implements<@max_init> reads[]
       : tensor<R_VALIDx1xfp32> scope<prologue> static.args<value=-inf>
  %sum_init = flow.arithmetic @sum_init implements<@sum_init> reads[]
       : tensor<R_VALIDx1xfp32> scope<prologue> static.args<value=0>
  %out_init = flow.arithmetic @out_init implements<@out_init> reads[]
       : tensor<R_VALIDxDxfp32> scope<prologue> static.args<value=0>
  flow.pipeline @loop %k = 0 to ceildiv(%END-%START,%TILE) step 1 depth(%STAGES)
       generation_counts_per_wave[1] {
    # K(k) 与 V(k) 允许先行 issue；stage slot k%STAGES 只有在 k-STAGES 消费完成时复用。
    %key,!key_ready = flow.load_async @key @K[%CACHE_BASE+%START+%k*%TILE,0]<VALIDxD>
       by %producer after[@qk.complete:reuse_k] : tensor<VALIDxDxQKV_DTYPE> scope<pipeline>
    %value,!value_ready = flow.load_async @value @V[%CACHE_BASE+%START+%k*%TILE,0]<VALIDxD>
       by %producer after[@pv.complete:reuse_v] : tensor<VALIDxDxQKV_DTYPE> scope<pipeline>
    # 不得再有 K(k) 必须等 score_scaled(k-1) 才能 issue 的直接序列依赖！
    %qk = flow.matmul @qk implements<@qk> reads[%query,%key]
       after[@query.complete:data,@key.complete:data]
       : tensor<R_VALIDxVALIDxfp32> scope<pipeline>  # 目标后端根据 fragment geometry padding M/N
    %score = flow.arithmetic @score implements<@score> reads[%qk]
       after[@qk.complete:data] : tensor<R_VALIDxVALIDxfp32> scope<pipeline>
    %tile_max = flow.reduction @tile_max implements<@tile_max> reads[%score]
       after[@score.complete:data] : tensor<R_VALIDx1xfp32> scope<pipeline> work<R_VALIDxVALID>
    %max_new = flow.arithmetic @max_new implements<@max_new> reads[%tile_max]
       after[@tile_max.complete:data,@max_init.complete:state_m,@max_new.complete:state_m]
       : tensor<R_VALIDx1xfp32> scope<pipeline>
    %alpha_delta = flow.arithmetic @alpha_delta implements<@alpha_delta> reads[%max_new]
       after[@max_new.complete:data,@max_init.complete:state_m,@max_new.complete:state_m]
       : tensor<R_VALIDx1xfp32> scope<pipeline>
    %alpha = flow.special_function @alpha implements<@alpha> reads[%alpha_delta]
       after[@alpha_delta.complete:data] : tensor<R_VALIDx1xfp32> scope<pipeline>
    %delta = flow.arithmetic @delta implements<@delta> reads[%score,%max_new]
       after[@score.complete:data,@max_new.complete:data]
       : tensor<R_VALIDxVALIDxfp32> scope<pipeline>
    %probability = flow.special_function @probability implements<@probability> reads[%delta]
       after[@delta.complete:data] : tensor<R_VALIDxVALIDxfp32> scope<pipeline>
    %prob_cast = flow.arithmetic @prob_cast implements<@prob_cast> reads[%probability]
       after[@probability.complete:data] : tensor<R_VALIDxVALIDxQKV_DTYPE> scope<pipeline>
    %prob_sum = flow.reduction @prob_sum implements<@prob_sum> reads[%probability]
       after[@probability.complete:data] : tensor<R_VALIDx1xfp32> scope<pipeline> work<R_VALIDxVALID>
    %sum_scaled = flow.arithmetic @sum_scaled implements<@sum_scaled> reads[%alpha]
       after[@alpha.complete:data,@sum_init.complete:state_l,@sum_new.complete:state_l]
       : tensor<R_VALIDx1xfp32> scope<pipeline>
    %sum_new = flow.arithmetic @sum_new implements<@sum_new> reads[%sum_scaled,%prob_sum]
       after[@sum_scaled.complete:data,@prob_sum.complete:data]
       : tensor<R_VALIDx1xfp32> scope<pipeline>
    %out_scaled = flow.arithmetic @out_scaled implements<@out_scaled> reads[%alpha]
       after[@alpha.complete:data,@out_init.complete:state_o,@pv.complete:state_o]
       : tensor<R_VALIDxDxfp32> scope<pipeline>
    %pv = flow.matmul @pv implements<@pv> reads[%prob_cast,%value,%out_scaled]
       after[@prob_cast.complete:data,@value.complete:data,@out_scaled.complete:data]
       : tensor<R_VALIDxDxfp32> scope<pipeline>  # 保持 acc_next = alpha*acc_prev + P@V
  }
  %normalize = flow.arithmetic @normalize implements<@normalize> reads[%pv,%sum_new]
       after[@pv.complete:last,@sum_new.complete:last]
       : tensor<R_VALIDxDxfp32> scope<epilogue>
  %log_sum = flow.special_function @log_sum implements<@log_sum> reads[%sum_new]
       after[@sum_new.complete:last] : tensor<R_VALIDx1xfp32> scope<epilogue>
  %lse = flow.arithmetic @lse implements<@lse> reads[%max_new,%log_sum]
       after[@max_new.complete:last,@log_sum.complete:data]
       : tensor<R_VALIDx1xfp32> scope<epilogue>
  !partial_done = flow.store_async @partial_out reads[%normalize] cast<fp32>
       to @Partial[%PART_BASE,0]<R_VALIDxD> by %producer scope<epilogue>
  !lse_done = flow.store_async @lse_out reads[%lse] cast<fp32>
       to @LSE[%PART_BASE,0]<R_VALIDx1> by %producer scope<epilogue>
}
mapping.program @fa3_gqa_decode_split_mapping flow<@fa3_gqa_decode_split_flow>
    target<%TARGET> placement<round_robin> {
  map.execute @producer capability<bulk_copy> -> @TMA
  map.execute @matrix capability<matrix.mma> -> @WGMMA  # 该 target-specific mapping 仅在 SM90 实例化后生成
  map.store @scratch -> @SMEM
}
```

上方计算节点已经逐个展开，不留数学公式给 Agent 决定；**但仍是结构草案而非经过 parser 校验的成品**。具体待实现的是：基于 `fa3_reference.flowir` 使用实际支持的 `iterate<%executor ...>` / `iterate<%fragment ...>`，在 **SM90 后端**选择合法的 `work<MMA_KxMMA_N>` 物理 fragment（例如 H100 WGMMA 的硬件规格由后端定义），而不是在通用 FlowIR 中固定其数值；把 `%R_VALID` 个真实 query 行映射到后端支持的物理 GMMA M fragment，额外行在寄存器/SMEM padding 且不产生 HBM 访问；仅写回有效 R 行；尾部 `VALID` 对应的 score mask 必须在归约之前生效；`max_init` 的第一次迭代 `alpha` 应为 0（不能 `exp(-inf - -inf)`）。这里的 `%R_VALID`、`%VALID` 张量形状是数学有效区域，不构成自动存在的原生 masked-WGMMA 语法。主流程先 compile + Native opcode audit，之后才能跑性能。

### 4.1 Pipeline 必需的精确依赖 DAG

对由后端/配置绑定的 `STAGES`，在 `k>=STAGES` 时定义：

```
reuse_k(k) <- QK(k-STAGES).complete
reuse_v(k) <- PV(k-STAGES).complete
QK(k)     <- Q.ready, K(k).complete
softmax(k)<- QK(k).complete, m(k-1), ell(k-1)
PV(k)     <- p_as_QKV_DTYPE(k).ready, V(k).complete, acc_scaled(k).ready
m(k),ell(k),acc(k) <- previous state and this tile results
```

同一 SM 的 TMA 硬件仍遵循现有资源竞争，不准伪造多 TMA engine。**不要把所有这些独立任务按 k 串成一个线性完整链**。必要时建立 `flow.dependency_rule @reuse_k from[@qk.complete] at[%k=%k-%STAGES]` 和 `@reuse_v from[@pv.complete] at[%k=%k-%STAGES]`，语法参照 full-tile reference 的 `@reuse`。对 `k<STAGES` 不能附加不存在的历史 buffer-consumer 依赖。与上一迭代 `max/sum/acc` 的依赖是数学必需的，不能为了提速删除。

### 4.2 Shared memory 物理预算（后端根据参数计算，不能硬编码）

`E=bytes_per_element(QKV_DTYPE)`，`M_phys=ceildiv(R_VALID, MMA_M)*MMA_M`（`MMA_M` 从目标后端取得），粗略的逻辑下界为：

```
Q padded      = M_phys * D * E
K stage slots = STAGES * TILE * D * E
V stage slots = STAGES * TILE * D * E
SMEM_min      = Q padded + K stage slots + V stage slots
SMEM_actual   = target_layout_and_alignment(SMEM_min, aux_buffers, barriers)
```

这只是**资源检查公式，不是当前原生 SMEM slot 分配结构的声称**；`flow.buffer` 的 `slots` 和 `bytes_per_slot` 必须结合实际 Q 常驻、K/V 多 stage 的 buffer 生命周期分配，不能将一个总数重复乘以 `STAGES`。在 `ArchIR/NativeIR` lowering 时检查 shape、TMA 对齐、register、barrier、SMEM 上限，失败则明确拒绝该 `{R,TILE,STAGES,DTYPE,TARGET}` 组合。针对 H100 的 `D=128,R=16,BN=128,STAGES=2,BF16` 可以在测试中验证历史的 147,456 B 理论下界，但**不能出现在通用 FlowIR literal 中**。

### 4.3 Combine 新布局（需要明确具体修改）

新建 `fa3_gqa_combine.flowir`，与旧 `fa3_combine.flowir` 的所有计算节点保持相同，仅替换两处地址：

```text
%partial, !ready = flow.load_async @partial
    @Partial[(%REQUEST*%S+%split)*%HQ+%HEAD,0]<1xD>
    by %producer : tensor<1xDxfp32> scope<pipeline> iterate<%split=0 to %S>
%lse, !ready = flow.load_async @lse
    @LSE[(%REQUEST*%S+%split)*%HQ+%HEAD,0]<1x1>
    by %producer : tensor<1x1xfp32> scope<pipeline> iterate<%split=0 to %S>
```

其余保持 `M=max(LSE_s)`, `weight_s=exp(LSE_s-M)`, `O=Σ weight_s*Partial_s/Σ weight_s`，最终 BF16。**所有 Partial/LSE 的真实 global store/load 只访问有效 R 行**；不得制造 64 个 Q rows 的 HBM 访问。

## 5. 调度：PackGQA 必须与 Split 联合搜索

**以下仅为 Qwen3-235B 测试配置**：`B=1,HKV=4,G=16,HQ=64` 时 `N_CTA=4*ceildiv(16,R)*S`。示例：

| R (Q heads/CTA) | S=4 | S=16 | S=32 | S=64 |
|---:|---:|---:|---:|---:|
| 16 | 16 | 64 | 128 | 256 |
| 8  | 32 | 128 | 256 | 512 |
| 4  | 64 | 256 | 512 | 1024 |

132 SM 下，`R=16,S=4` 仅 16 CTA，不能用于证明 PackGQA 性能不好；反过来 S 增大过多也会让单个 split 太短、combine 更贵。

严格分两阶段：

1. **Correctness sweep**：R=16；S=1、4、16、32（S=1 用对应 direct path 或明确标记未支持）；BN=64、128；L=1、17、127、128、129、255、256、257、10000、10001、100000、200000。测试非整除 split、尾块、不同 batch/context。
2. **Performance sweep**：R∈{4,8,16}、S∈{4,8,16,32,64,128}、BN∈{64,128}，过滤非法/空 split。在 Qwen 评测中首选考察 `R=16,S=32/64` 和 `R=8,S=16/32`，不将这些数值写入通用库。测 `FA3_split + combine` 之和，不仅测 split kernel。保留未优化的 `R=1` legacy 基线；时间由实测/仿真给出，不预设最优 split 数；自适应启发式只能在 sweep 验证之后引入。

`R=16,S=32` 在该 **Qwen/H100 benchmark** 下 CTA 数为 128，约覆盖 132 SM 一轮；不意味着一定最优。不同 context 长度独立选择最佳 S，不得把 batch=1 的 heuristic 推广成高 batch 默认方案。

## 6. 验证体系（没有这些证据不能合并）

### 6.1 数学 Oracle

用 PyTorch CPU/GPU 构建两个独立参考：完整 FP32 稠密 GQA `softmax(QK^T/sqrt(D))V`；和 BF16 probability 在 PV 之前转换的 TileLang 风格 Online Softmax oracle。计算所有 split partial 和 LSE 后用 `logsumexp` 合并；核对 `S=1` 与多 split 的等价性及各维度 tail。报告 max abs / relative error；容差须根据 FP32/BF16 两种参考分别设定，不能用模糊的“近似一致”。GTSim 时间模拟器不计算实际 Transformer 数值，因此使用独立数值 oracle，而不是把 timing DAG 当成数值证明。

### 6.2 内存与 CTA 精确性

- 至少覆盖两组**非 Qwen 维度**（如 HQ/HKV/D 不同、G 不被 R 整除的尾 block）以及 Qwen 回归参数，确认编译/地址生成对参数变化敏感，无源码常量残留。
- CTA 数应严格等于 `B*HKV*ceildiv(G,R)*S`；每 `(b,h,s)` 恰好有一个 producer；相邻 split 范围无重叠且覆盖全部 `L`。
- K/V 每 `(b,kvh,qblock,split,k)` 各读取一个真实 tile；不再按单个 Q head 分别读 K/V。同一 group 内复用一次 load；注意不同 splits 对应不同 KV 范围。
- Full group `R=G` 时全体 kernel 的 **逻辑 KV 读取量** 相对 legacy per-head 应接近 `1/G`（相同 K/V tile 与 S 的前提）；这并不保证真实 HBM bytes 恰好降至 1/16，因为 L2 已能命中大量重复请求。不可把 cache hits 当成 HBM 读取。
- 真实 Q/O/Partial/LSE 访问不得包含物理 GEMM padding Q rows；验证全覆盖无重复、每个 HBM global address 正确。
- 保持 Q/K/V/O 以及 weights 的 HBM/HBF 原始 placement，不允许为了图表变好重写 GPU 核心或 memory controller。
- 新 layout 要对 combine 的所有 input index 做精确 audit。

### 6.3 Native opcode / DAG 检查

- `qk` 和 `pv` 必须降低到目标平台真正的 MATRIX/Tensor Core 类（SM90 实例化为 WGMMA），不是 `MUL+REDUCE_SUM` 的 SIMT 路径；证明两者消耗的物理 matrix fragments、warpgroup 数、barriers、SMEM slots。
- 依赖验证：允许 K(k+1)、V(k+1) 提前 issue；只有对应 stage slot 的 K/V(k+STAGES) 重用必须等 QK/PV(k) 消费完成；无读写覆盖冲突。
- 检查 `Q` 的 `R_VALID` 个有效行到目标后端物理矩阵 M fragment 的映射、padding 初始化与 final masking；P@V 的 `QKV_DTYPE` 概率输入 dtype 与 FP32 acc 元数据；所有 `m/ell/acc` 是**逐 query row** 的状态，而不是全部 packed heads 共享一个 max/sum。
- 用原已有 `test_prog_to_flow.py`/`test_ast_parser.py` 和新增测试确认解析+lowering，无“kernel 名字改了实际 opcode 仍旧”的假优化。

### 6.4 H100 官方 CUDA 性能对照（具体 workload 实例，不是库常量）

以下 Qwen 维度**只属于测试 JSON 和 benchmark invocation**，绝不能写入 `.algoir/.flowir`。在真实 H100 SXM 上用 `hopper/flash_attn_interface.py::flash_attn_with_kvcache` 的 BF16 contiguous KV 输入，`seqlen_q=1`, `num_heads=64`, `num_heads_k=4`, `head_dim=128`, `batch=1`，对 10K/100K/200K 分别测：

- `pack_gqa=True, num_splits=0` 自动 heuristic；
- `pack_gqa=True, num_splits=4,8,16,32,64`（API 是否允许的细节以实际版本为准）；
- `pack_gqa=False, num_splits=1` 参考；
- warmup + CUDA event 多轮 latency、官方两个 kernel 总时间、Nsight Compute HBM/L2 bytes 和利用率。

真实 GPU 对照是独立实测；不能拿官方论文中大型 Prefill TFLOPS 当 Decode BW 参考。配置必须匹配 GPU 型号、Dtype、KV layout、batch、length，避免显存不足时悄悄更改设备或测量范围。对于 `CONTEXT=200000` 的 KV cache，独立 BF16 K+V 至少 `409.6 MB`，理论 HBM 读取时间下界约 `132 μs@3.1 TB/s`，**不是性能目标**。

### 6.5 模拟器性能归因

对 legacy/packed 每一套：(1) `fa3_split`、`combine`、总 attention latency；(2) per-kernel HBM read bytes、L2 served bytes/hit/miss、TMA issued bytes、各单元利用率；(3) CTA resident/active SM；(4) k-iteration critical path。保留 `L2 BW↑ 不变、L2 BW↓ 恶化` 作为已有观察：说明 L2 有阈值但不足以证明默认带宽饱和。不要为了让 HBM BW 接近 3100 GB/s 而改动性能模型；目标是**真实 CUDA latency/traffic 校准**。

## 7. 推荐实施步骤、提交边界和交付

- **Commit 1 / oracle**：补 GQA packed index、split bounds、online max/sum/acc、LSE combine 数值 oracle；先通过边界与不等长 B>1 的所有测试。
- **Commit 2 / IR math & layout**：新增两对**参数化** `.algoir/.flowir`、新 split-major layout、Program 选择分支；验证所有有效 global ranges 和输出覆盖。此阶段若未启用实际 MATRIX lowering，不得汇报为 FA3 Hopper 完成。
- **Commit 3 / target lowering**：复用 `fa3_reference` MATRIX 模板，在 SM90 specialization 实现真实 `qk/pv` WGMMA 分片、SMEM Q padding、按参数生成 K/V stage slots、warpgroup barriers 和最小依赖；native op audit + resource feasibility + deadlock tests 全通过。
- **Commit 4 / tune & calibrate**：矩阵扫描 R/S/BN，独立跑 H100 官方 CUDA 对照，记录来源/commit/hash、所有 kernel 周期和 memory counters；择优添加自动 split heuristic。
- **Commit 5 / integration**：把调度开关接入现有 FlashAccel 的 FullHBM/4F1M 两种路径并复测原 HBF 研究曲线；legacy 保持 bit-for-bit 配置相同的回归。

Agent 最终只需汇报：实现文件与 commit、数学/地址/Native IR 检查结论、参数绑定与通用性测试、Cast 降低证据、每个 benchmark context 的最佳 `{R,S,BN,stages}`、真实 H100 与模拟 cycles/traffic 对照、剩余模型误差与已知缺陷。不要生成大量无关文档；**禁止通过修改 HBM/L2/TMA 参数或关闭成本节点来追逐低延迟**。凡无法用当前 parser 表达的 SSA fragment/物理 padding，要显式补最小 IR 支持与测试，而不是 silently fall back 到 SIMT。

