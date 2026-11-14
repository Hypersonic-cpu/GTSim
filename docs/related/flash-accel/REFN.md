# GTSim × FlashAccel：Tensor Partition、异步 Prefetch 与 MegaPlane 复现计划

> **供 Coding Agent 直接执行的实现规范（不要求再次自主设计）**
> 代码基线：`Hypersonic-cpu/GTSim` `sim-validation@a2d918204c9163b85e559ac6480f29805f8a2fa7`；HBFine 子模块基线：`abda9f3b946c073a5a1037f9ef963845f2e5325a`。
> 论文依据：用户提供的 *FlashAccel: Leveraging High-Bandwidth Flash for High-Throughput LLM Inference*，`2607.10186v2.pdf`，主要是 §V-A（Weights Layout）、§VI-B3 / Fig. 12（Prefetch）、§VII-A（LLMCompass）。
> **只实现本 Plan 的功能，不进行额外性能搜索或启发式优化。** Tile-Major 暂不检查/扩展；原有 SRAM Fill 队列的 HOL Blocking **本轮保持原状**。

## 0. 设计决策与论文边界

| 层次 | FlashAccel 论文表述 | 本项目确定的做法 |
|---|---|---|
| Weight / Tensor Partition | 大于 SRAM 的对象被分成可驻留的较小 Tile；每个 Tile Prefetch → GEMM → Release（Fig. 12a） | Frontend 的独立 **Partition Pass**，为 GEMM 构建 Sub-Tensor **View** 与对应子 GEMM Kernel；先实现按输出 **N 维**的容量约束分区 |
| GEMM Tiling | LLMCompass 搜索 GPU GEMM Mapping（§VII-A） | **不在 GTSim 中做搜索**；从已有 C++ GEMM Validation 中固定选一组有效参数，通过参数替换传给 Frontend 的计算 Schedule |
| Prefetch granularity | 一个 `SramPrefetch(vaddr, size)`，底层可跨多个 MegaPlane，以 HyperPage 组织读取（§VI-B3） | Prefetch **Request Range** 与 GEMM 的 `BM/BN/BK` **解耦**。一个 Sub-Tensor 可展开为多个连续物理 Page/MegaPage 传输；计算不依赖是否完成 |
| Prefetch 顺序 | 从计算图抽出 Prefetch，按**计算拓扑顺序**进入队列，异步运行（Fig. 12b,c） | 由**分区后的 FlowIR** 和真实首用顺序生成 QueuePrefetch；SRAM-side 一个独立 HardwarePrefetchEngine 管理 FIFO、容量/释放、Fill |
| SRAM 满 | 从队首不断尝试，直到无空间；可并发处理已派发 Prefetch | 维持 **FIFO + 当前 HOL**；不能偷偷改为跳过阻塞队首，也不要引入复杂预测/priority policy |
| Miss | 未 Prefetch 的区域保留 HBF 原址；Compute **不等待** Prefetch | Demand HBF fallback，新增**可配置** `cancel_queued_on_demand_miss`；默认 `false`（更接近论文）；只能安全地取消**未进入媒体设备的排队工作** |
| MegaPlane | M 个 Plane 共用一条 NAND 命令，MegaPage = M × 4KiB；权重按计算先后轮询布局 (§V-A) | `planes_per_command_group=32` 默认；`1` 视为关闭；物理命令分组和软件权重 Address Mapping Policy 分离 |

**明确非目标**：预测未来 Expert、预取顺序 ML 优化、跨 Stream Kernel 并发、GPU/L2/SM/Scheduler 重写、Tile-Major、自动 GEMM Tiling Search、改变原 SRAM Region 的 HOL 规则、增添兼容性的旧 Plan/旧 Prefetch 路径。

论文中“分区”不代表一定创建新的物理 Tensor：本项目用 `TensorPartitionView(parent, row/col offset, shape, parent strides)`，**不复制原始权重，也不重复分配 HBF 容量**。GEMM 使用这些 View 创建独立的计算 Kernel。Prefetch 物理传输可以比 View 更细。

## 1. 当前代码中的必须修复点（有明确依据）

1. `frontend/src/compile/opt/prefetch.py::split_region()` 当前只是按照 **CTA 区间**贪心切开；**一个 CTA 超出 SRAM 时直接失败**。移除这个不适当的 Partition 实现，按下面的 Sub-Tensor 设计替代。
2. `frontend/src/compile/streaming.py::make_program_plan()` 从**原始 ProgIR Launch 名称**构建命令，但优化后出现 `__tile0/__tile1` 等名称；混合“被切分 + 未切分 Kernel”的完整程序会出现名称与命令不匹配。使用显式 `old_launch_id → [partition_launch_ids...]` 重写命令图。
3. `opt/prefetch.py::Residency.location()`、`HBFMegapageAllocator`、`HBFine::decode_address()` 各自管理一套几何/基址规则。必须合并成**一个数据布局契约**，不要让 Python 猜测和 C++ 独立分配不同物理地址。
4. `include/memory/system.h` 当前管理 `prefetch_queue_`、`queued_prefetches_` 和 admission；`src/memory/sram.cpp` 再管理 `regions_`、`active_regions_`、background fills。重构为 **唯一 SRAM-side Prefetch Engine 的状态机**，不得保留两份并行队列/两套负责发射的状态。
5. `src/memory/sram.cpp::submit_background()` 目前被拒绝的一个 Region 会阻塞后续 Region。**保留这个行为**，不列为本轮 Bug Fix。
6. `src/memory/flashlab.cpp` 当前每次请求调用单独 `run_batch()`，并更新 `origin_ = blocking_finish_ns`。与真正多 Outstanding 的异步媒体时序可能不一致；优先完成 HBFine 后再核实其 Native API，不能通过伪造完成事件使测试变绿。
7. 现有 `configs/memconf/4F1M_eventlab.yaml` / `4F1M_flashlab.yaml` 使用 `hbflex_research`，不是 FlashAccel 的 `grouped_planes`。**统一从 `4F1M_grouped_planes.yaml` 派生**三种 Backend 的 4F1M 验证配置，不用原先的不一致几何做直接对照。
8. `MemRequest/MemResponse` 主要携带**地址、长度和完成状态，没有真实 Tensor payload**；**Cycle 一致、没有错误响应不等于已经验证读出的数据正确**。见 §7 的测试专用 Shadow Payload / CPU Oracle 设计。

## 2. 新的 Frontend 结构（必须按顺序实现）

### 2.1 Pass 位置与输入输出

```text
ProgIR（保留原始 sync / event / launch）
  → compile_program → FlowIR（原始计算语义）
  → PartitionPass（TensorPartitionView + PartGemmLaunch + origin mapping）
  → PrefetchPlanPass（根据 Part 的首用顺序抽取 Preload/Queue/Release）
  → lower_flow → ArchIR（固定 GEMM tiling）
  → ExecutionIRProducer（仍然只按需展开一个 Kernel）
  → 单 Stream ProgramPlan（显式 rewritten launches / prefetch commands）
```

**不要**从 NativeExpander 的已展开 DAG 再决定 Partition；也不要把所有 Partition Kernel 提前展开到 `ExecIR.ops`。Partition/Prefetch Pass 只生成 compact region / view / command metadata。

### 2.2 Class 与接口（仅需很小的数据模型）

建议将 `opt/prefetch.py` 的混杂逻辑拆为 `opt/partition.py` 与保留的 `opt/prefetch.py`，**删除旧 `split_region()` 和与新路径重复的 streaming window 优化器**。`opt/partition.py`：

```python
@dataclass(frozen=True)
class TensorPartitionView:
    parent: str
    row: int
    col: int
    rows: int
    cols: int
    parent_row_stride: int  # 元素单位，不是小 Tensor 自己的宽度

@dataclass(frozen=True)
class PartitionResult:
    flow: FlowIR
    origin_launches: dict[str, tuple[str, ...]]  # old launch → all child launches
    views: tuple[TensorPartitionView, ...]


def partition_flow(flow: FlowIR, policy: PartitionPolicy,
                   layout: PhysicalLayout) -> PartitionResult:
    ...
```

`PrefetchPlanPass` 以 `PartitionResult` 和原始 Graph Topo Order 为输入，返回 `PrefetchPlan`（requests, consumers, enqueue-position, release-position）。同一个 `TensorPartitionView` 的 `parent` 保持原来的 Global Tensor ID 与地址范围，不为每个 View 注册一份完整权重。

### 2.3 固定的 Partition Policy（只实现一种）

策略名：`n_slab_fit_double_buffer`。

1. 仅对 HBF 上的静态、只读 GEMM 权重执行 Partition；Norm 等小权重未超容量时保留原计算 Kernel，只按正常首用加入 Prefetch。
2. 按完整 K、不拆 K 的方式，把权重沿 **输出 N** 维逐列切分，`N_part` 是 **GEMM `BN` 的整数倍**。这样子 GEMM 写入互不重叠的 C 列；**不需要 K 方向 Partial Sum**。
3. 从 `N_part = BN` 开始，按 `BN` 递增，使用**实际物理 Page footprint**（不是只用 `K*N_part*dtype_bytes`）尝试放入一个 Double Buffer Slot 的 Plane SRAM + Base-die SRAM。选择最大可容纳的 `N_part`；剩余尾部不足 `BN` 时保持原 Schedule 所要求的对齐/Mask（不可默默丢列）。
4. 容量计算应按**一个 Slot**：每 Stack `plane_sram / 2` + `base_die_sram / 2`，每 Plane 按现有完整 Page 取整与 Slot 剩余页分配算法计算。两 Slot 总预算不能混用到一个 Request。
5. 一个 `BN` 单元仍不够：本轮明确报告不可分区；**不得隐式改 GEMM Tiling 或偷偷沿 K 分块**。如必须支持则单独实现 K-split + 合法 Accumulate 的另一个 Feature，不得用错误的拼接模拟。
6. 对每个 View 生成 `gemm_parent__part{i}`，Compute Kernel 消费 `B[parent][rows, cols]` 并仅写对应输出列。即便 View 的字节不是连续段（RowMajor N-slab），必须保留 `parent_row_stride`，按逐行 Offset 计算 Page Set 和物理地址；**不能把 N 方向切片错误看作一段连续 HBF 字节**。
7. Partition 前后必须逐元素覆盖相同的 GEMM 输出：C 的输出列不能遗漏/重复；M 小于 BM 时沿用已支持的尾部 Mask / padding。

**特别针对 Qwen MoE**：`W_gate_up` 的逻辑维度为 `(E*H) × (2I)`；每个被选择 Expert `e` 的权重 View 是 `row=e*H..(e+1)*H`；`W_down` 的 View 是 `row=e*I..(e+1)*I`。一个 Expert 的多份 `N_part` 可合成一个完整输出，不要误将不同 Expert 的行基址当成连续的单个 GEMM 权重行。保留 `A_MAP/C_MAP` 对 token/assignment 的映射。

### 2.4 GEMM Tiling 只采用参数替换

**固定示例（可在实现时选仓库现有已验证的同类组合，不自动搜索）：** `BM=64, BN=128, BK=64, stages=2, split=1`，来自 `workloads/common/workload_options.h` 和 `workloads/gemm/gemm_h100.cpp` 的 C++ GEMM 选项；`M=1` 对应已有 BM=64 padding 约定。

- 在 `frontend/` 的 `small_m_gemm` / `grouped_gemm` Schedule 里**显式参数化** BM、BN、BK、stages，或者选一份现有可用的固定 Schedule；**参数必须真正影响 CTA/Load/Matmul/Store 的 Loop 与 Resource Mapping**，不能只改 JSON 字段、实际仍执行 `16x64x16`。
- 只实现一组能运行的参数。若这个 C++ 组合不能直接映射到现有 FlowIR/WGMMA，先据现有合法 FlowIR 选择一组对应经过验证的固定参数，写在报告里；不可为了“通过”使用不支持的空 WGMMA 循环。
- `gemm_tiling` 是用户显式参数，`partition` 仅基于这个 `BN` 做对齐，不搜索更高效组合，也不以 LLMCompass 为运行时依赖。
- Attention/FA3 和 GPU L2 以上模型不变。

## 3. Weight Physical Placement、MegaPlane 和配置

### 3.1 单一几何事实来源

`--memory configs/...yaml` 定义物理事实：`stacks, dies_per_stack, planes_per_die, planes_per_command_group, page_bytes, sram_bytes_per_plane, base_die_sram_bytes_per_stack, address_layout.hbf_base`。Frontend 从**同一个 Memory YAML**取得这些值，`--flashaccel-config` 只定义 Partition/Tiling/Prefetch Policy；**不再输入第二份重复的几何 JSON**。

建议的物理/软件配置（示例）：

```yaml
# memory.yaml — 物理架构
hbf:
  backend: hbfine
  controller:
    geometry: grouped_planes
    planes_per_command_group: 32  # 1 代表 MegaPlane OFF
    enable_read_sram: true
  address_mapping_policy: megapage_round_robin
  prefetch_hyperpage_order: true
  prefetch_engine:
    enabled: true
    queue_capacity: 1024          # 描述符数；不等于 NAND request capacity
    admission_policy: fifo_hol    # 唯一实现，本轮不修改 HOL
    cancel_queued_on_demand_miss: false
```

```json
// flashaccel-policy.json（示意；实际 JSON 不写注释）
{
  "partition_policy": "n_slab_fit_double_buffer",
  "buffer_count": 2,
  "gemm_tiling": {"bm": 64, "bn": 128, "bk": 64, "stages": 2, "split": 1},
  "prefetch_granularity": "hyperpage_coalesced"
}
```

不要新增多个同义的 `enable_megaplane` 标志。物理 `planes_per_command_group = 1` 即关闭 MegaPlane；保留 `geometry=grouped_planes` 使映射退化一致。`address_mapping_policy` 与 M 是两个正交参数：**改变 M 不等于选择新的静态布局策略**。本轮只需 `megapage_round_robin` 与朴素 `linear_page` 两种明确 Policy，默认采用论文的 Round Robin。

### 3.2 Placement Policy 必须与使用顺序吻合

论文 §V-A 是**静态权重按模型计算拓扑顺序**，切成 MegaPage 大小、横跨 MegaPlane 轮询排列；该顺序是模型结构顺序，并非每个 Batch 的动态路由预测。

- 静态权重放置顺序：原始模型拓扑的**首次消费 Operator**（如 W_qkv 在 W_o 之前；W_router 在 Expert 权重之前），同一 Operator 多 View 依 Partition Index 排序；同一 Expert 权重在静态布局可按 `expert_id` 升序。Placement 与运行时 Route/Batch 不耦合。
- 执行时的 Prefetch 顺序：**当前实际路由与计算划分**的首次消费顺序，见 §4。二者不同，不要混在同一个 `sort()` 里。
- 一个权重对象的逻辑地址和 Tensor ID 不变；只建立唯一 `(Tensor, offset) → HBF physical address` 的 Mapping。`HBFMegapageAllocator` 不再独立按照 Tensor 声明顺序分配，而是使用统一的 Placement Plan；其结果传给 C++ Registry/Backend。
- 给 `Frontend` 提供纯函数形式的 `PhysicalLayout.location(page)`（可共享公式/生成式描述）；与 HBFine `decode_address` 和 SramBackend `locate()` 做**样本 Page 地址双向比对**，包含跨 MegaPage、跨 Stack、非整 MegaPage Tensor 尾部。
- 允许关闭对齐、切换 M=1/M=32，但地址有效范围、Plane SRAM 容量必须保持一致；本轮不研究 Tile-Major。

## 4. Prefetch Plan：严格依照图的拓扑顺序

### 4.1 首用与可提前发射的时刻（明确 Policy）

`PrefetchPlanPass` 在**分区后 FlowIR** 上工作，不需要展开 Native Instruction：

1. 从 `FlowIR.regions` 和其 `after` 关系构造稳定的拓扑顺序，tie-break：`original_launch_position`、`expert_first_use`、`partition_index`。**不要只按 Tensor 声明顺序**，也不能按 HBF 物理地址排序得到计算顺序。
2. 对每个消费 HBF Static Weight 的 Sub-Tensor View 求取 Page 覆盖，按 HyperPage/MegaPage 地址分组成 1 个或若干 `PrefetchRange`。**Range Bytes** 和 GEMM 的 `BM/BN/BK` 解耦：计算一个 Sub-Tensor 可以发多个物理 Prefetch 请求；Allocator 统计其全部 Page Footprint。
3. Prefetch 先在逻辑顺序 `PrefetchQueue = stable_topo_order(requests)` 中排队。默认在**合法且 SRAM 有容量的最早时间**尝试 admission，不等 Consumer Kernel 被请求生成才开始。允许 Kernel A 运行期间开始为后继 Kernel B/C 填充 SRAM。
4. 对**静态已知**的 QKV/O/Norm Weight，可在该 Step 开始时将其按拓扑顺序入队（受 Double Buffer 容量和 FIFO HOL 限制）。对**依赖当前 Router 输出**的 Expert，只在 Router 结果可用/Trace 被明确固定后加入对应的准确 Expert；**不得利用未来未知 Route 当作完美预测**。
5. `QueuePrefetch` 只表示 enqueue，**不等待 NAND/SRAM Ready**；Compute Kernel 照常被调度。对应 ReadAlias 初始仍指向 HBF；只对**已经完成填充的 Page**启用 SRAM Resident 路径，未 Ready 的 Page 必须 HBF Demand Fallback。最后消费者完成后 `Release`；若仍有后台填充，则延迟实际释放，避免 UAF。
6. `ProgramPlan` 从变换后的 Launch 顺序构建。将原始 `Launch X` 替换成它的 `[X__part0, X__part1,...]`；`RecordEvent/WaitEvent/Synchronize` 在**最后一个 Part 完成**后才产生原来的完成语义。旧 Launcher、PrefetchPass、ExecIR 不得再并存另一条完整路径。
7. `enqueue_at` 必须是合法 Command/Launch 边界（并区分数据依赖门控），不是编译器做 NativeExpansion 的时刻。不要在每个计算 Kernel 入口额外 `memory_->idle()`。

### 4.2 Qwen3-30B-A3B：具体权重消费顺序

直接以 `frontend/asm/qwen3/flash-accel/one_block.progir` 和 `frontend/lib/grouped_gemm.flowir` 为准：

```text
W_input_norm
 → W_qkv (QKV projection)
 → W_q_norm / W_k_norm (Norm/RoPE)
 → FA3（KV HBM）
 → W_o (Attention O projection)
 → W_post_norm
 → W_router (Router projection)
 → router_softmax_topk / assignment ready
 → W_gate_up[selected experts] (Gate + Up fused GEMM; many partitions)
 → SwiGLU
 → W_down[selected experts] (Down GEMM; many partitions)
 → weighted sum → residual
```

注意：**当前 Qwen IR 是融合的 `W_gate_up`，不是三个独立的 Gate、Up、Down 权重 GEMM。** `GateUp` 的 Output 在 `SwiGLU` 前完整就绪，`Down` 必须等待 `SwiGLU`；不要为了复刻论文示意 Fig. 12 而强行拆成三条不同的原生 Operator。

`grouped_gemm.flowir` 通过 `ORDER=argsort(IDS)`、`RUN_LENGTHS`、`TILE_OFFSETS` 决定 CTA 实际处理顺序。对于 B=1 / 现有 `b1_ctx512.json` 的 `ROUTE_IDS`，按计算首次访问可得到所选 Expert 集合按 ID 排列（示例 `[10,66,77,98,103,107,122,124]`），**不是输入列表原顺序**。应从**最终 Partitioned GEMM Schedule** 确定真正 first-use，而不是对字符串名字或用户 `ROUTE_IDS` 直接排序。相同 Expert 内再依其 Part 的真实计算顺序，先 GateUp 后 Down。多个 Token/重复 Expert 合并相同 `(tensor, expert, partition)` 的重复 Prefetch，但保留所有消费者。

关于额外文献：MoE Expert 预测式提前加载（如 SpecPrefetch、Pre-Attention Expert Prediction）属于**另一种策略**，本复现不采用；仅使用已知路由/固定 Trace 和正确的执行拓扑。FlashAccel 论文只要求拓扑顺序，不指定同一拓扑级内各 Expert 的优先级；因此我们的稳定 tie-break 是**实现约定**，不是论文明确要求。

## 5. 独立 SRAM-side Hardware Prefetch Engine

### 5.1 **唯一归属**和接口

新增 `include/memory/prefetch_engine.h`、`src/memory/prefetch_engine.cpp`（或等价两文件）。**每个 GTSim MemorySystem / 模拟 GPU 只创建一个 Engine，位于 SRAM-side `SramBackend` 内部**；`HBFine`、`EventLab`、`FlashLab` 的 Raw Backend 不拥有 Engine，也不复制其队列。

```cpp
struct PrefetchRequest {
    uint64_t token;
    uint64_t address, bytes;
    std::optional<PrefetchBuffer> buffer;
};

enum class PrefetchState : uint8_t {
    Queued, Admitted, Filling, Ready, Released, Canceled
};

class HardwarePrefetchEngine {
public:
    uint64_t enqueue(PrefetchRequest request);
    void step(int cycle);                    // FIFO admission + submit Fill
    void on_fill_completion(const MemResponse& response);
    bool page_ready(uint64_t token, uint64_t physical_page) const;
    void on_demand_miss(uint64_t physical_address, uint64_t bytes);
    void release(uint64_t token);
    // 紧凑 counters()；不额外建 Statistics Framework。
};
```

**建议所有权结构**：`SramBackend` 持有 `unique_ptr<MemoryBackend> raw_media`、`ReadSramAllocator`、**一个** `HardwarePrefetchEngine`；Engine 持有队列、Region、已提交 Fill ID→Region 和 Ready Page 状态，向 SRAM Allocator 请求空间，向 Raw Backend 提交 `ReadPath::Prefetch`。`MemorySystem` 只负责把 Tensor+Offset 解析成 Physical Range 并转发 `enqueue/release/query`；**删除**现有 `MemorySystem::prefetch_queue_` 等重复队列和 `SramBackend::active_regions_` 的**重复 Owner**，不能新旧两套都保留。

### 5.2 Tick 与状态机（必须保留异步）

```text
Queued --SRAM admission succeeds--> Admitted --> Filling --> Ready
  |                                 |                    |
  | (optional cancel on miss)       | (in-flight cannot  | Release after last
  v                                 |  be blindly killed)| consumer + Fill
Canceled                             +--------------------+--> Released
```

- `engine.step(cycle)` 在 `MemorySystem::step(cycle)` 内推进，每一个 GPU Cycle 都可执行，不依赖当前是否正在运行预取目标 Kernel；**不能只在 KernelBegin/KernelEnd 调用**。
- FIFO admission：由队首开始，SRAM 够就继续 admission，直到头部因 SRAM 容量阻塞；已 Admission 的多个 Region 可以并发 Fill。
- Fill dispatch：**保持当前 `submit_background()` HOL 行为**：如果媒体 Backend `try_submit` 拒绝了当前 Region 的一条 Fill，当次停止后续 Region 的提交。**本轮不得修复/绕过此 HOL。**
- 未 Ready 的 Demand Read 从 HBF 原址读取；Ready 的 Page 从 SRAM 路径读取；禁止因为 Prefetch 尚未完成而阻塞 Compute。
- Request ID 和 Token ID 必须分域；已提交 Fill 每个仅完成一次；不存在“释放了 Region 但在 Backend 中仍有悬挂 Completion”情况。
- Double Buffer 的 Slot 占用由 Engine 的 Allocator 维护；某 Partition 所有消费者结束后，才允许复用相应 Slot。异步填充尚未完成的 Region 应延迟实际 Reclaim。
- `MemorySystem` 的 `pending_background_requests()` 与 Stats 从唯一 Engine 取得，不再通过两层不同计数拼凑；FullHBM 不创建/运行 HBF Engine。

### 5.3 配置式 HBF Miss → Prefetch Queue Removal

新开关 `hbf.prefetch_engine.cancel_queued_on_demand_miss`，默认 `false`；测试 `false/true` 两个值。

- **false**：完全保留队列，Demand Read HBF Fallback 不影响将来 Prefetch。
- **true**：Demand 读一个属于 `Queued` 请求的 HBF Page 且这页尚未 Ready 时，**从尚未 Admission 的队列删除这个目标 Prefetch Descriptor**，更新 canceled 计数。后续访问继续从原 HBF 地址读；不得返回空/过期的 SRAM Alias。
- **不可取消已经 `Admitted`/`Filling` 的媒体请求**（没有原生取消 API）；保持跟踪直至 Completion，之后根据最后 Consumer 的生命周期安全释放。不能把“删除队列项”误实现为删除正在被 Raw Backend 使用的 Region。
- 对一条 Prefetch Request 有多页的情况，首个相关 Demand Miss 触发**整条尚未 Admission Descriptor**的取消，是本轮确定的简单 Policy；不做部分 Page Cancel/重排优化。
- 取消与 Release/Consumer Completion 相互幂等，避免 Queue 死节点和重复释放；默认关闭时不改变论文型调度语义。

## 6. Backend Adapter 分工与最小修改

| Backend | 必须做 | 不允许混入 |
|---|---|---|
| **HBFine（先完成）** | 正确的 MegaPlane M=1/32；统一 `locate/decode`；异步多 Outstanding NAND Read/Response，和 Engine 的 Fill ID 一一对应；维护 Group Sense 等统计 | 不在 HBFine 内创建第二个 Prefetch Engine 或复制 SRAM Queue |
| **EventLab（后完成）** | 使用同一 `grouped_planes` 配置，正确映射每 Stack/Die/Plane；Native EventLab 接受预取/Resident ReadPath 并保持多 Request 并发，不能“配置成功但忽略路径” | 不为 EventLab 另造 SRAM Residency 或 `QueuePrefetch` |
| **FlashLab（最后处理）** | 检查 `run_batch`/`blocking_finish_ns` 是否造成串行化。必要时最小化修改 Adapter 或 Native API，使其支持与当前 Cycle 兼容的多 Outstanding Work；不能把每个 Request 强制跑到 Blocking Finish 再接受下一个 | 不靠固定超长延迟或假的 completed response 通过验证 |
| **FullHBM simple** | `configs/memconf/full-hbm.yaml`，Naive HBM；保持同一 Partitioned Compute 图；Prefetch on 是无 HBF 的**明确 No-Op Control** | 不得在 FullHBM 创建 HBF SRAM Engine，也不声称测到了 HBM→SRAM Prefetch |

仅在必要时改动第三方原生 Backend，优先改 GTSim Adapter；自己的 HBFine 可以直接修改。更改 `3rdparty/` 时必须说明原因、锁定子模块提交和附带对应 Native Tests。不要顺便清理 EventLab/FlashLab 中与这个任务无关的接口。

> **一个重要的事实**：当前 `4F1M_eventlab.yaml` 与 `4F1M_flashlab.yaml` 仍使用 `hbflex_research`。最终要复现 FlashAccel，请由统一 `4F1M_grouped_planes.yaml` 模板生成 `backend: hbfine/eventlab/flashlab` 三份等几何配置；不能用不同几何/Stack 布局产生的周期差异证明优化效果。三个 Backend 的物理时序实现可能不同，不要求彼此 Cycle 相同。

## 7. “读数据正确”必须是真实可验证的结果

当前 GTSim Backend 的 `MemResponse` 没有数值 Payload，**单凭 `completed=true` 或 Cycle 相等不能证明数据正确**。本轮需提供两个互补、但不侵入 GPU 计算模型的验证：

1. **Partition 数值 Oracle（C++ 测试）**：生成一组很小的确定性 BF16/FP32 输入与权重（含非对齐 N 尾部、多个 Expert、重复 Expert），使用 C++ CPU 参考 GEMM 验证未 Partition 与多个 `N-slab` Part 合并后的数值一致；比较逻辑输出矩阵元素（允许浮点舍入误差）。测试 K/N/M 的地址偏移、每一列是否恰好写一次，以及 `GateUp → SwiGLU → Down` 的计算依赖。不因存储介质变化而改变期望数值。
2. **Memory Read Shadow（仅 Validation Mode）**：使用确定性字节函数/Shadow Page Data 在 HBF 媒体源产生数据，Engine 在 Fill Completion 时记录实际复制的 Page 内容或 Hash；Demand HBF Fallback 和 SRAM Resident Read 均按**逻辑 Tensor ID + Offset**核验相同字节内容，覆盖 Partial Page、并发 Fill、Release、Miss Cancellation。至少在**小规模 Probe 中逐字节比对**，Qwen 全量实验可采用流式强 Hash + 区间/长度一致性检查以避免多 GiB Shadow 存储。不要把单纯的 `Request ID` 一致称作“读数据一致”。

输出到每个 `result.json` 的明确字段（Agent 实现并保持 Bash 可检测）：

```json
{
  "data_check": {
    "checked_reads": 1,
    "checked_bytes": 128,
    "mismatched_bytes": 0,
    "logical_read_digest": "hex-digest",
    "shadow_payload_mode": "hash_or_bytewise",
    "partition_oracle_pass": true
  },
  "prefetch_stats": {
    "queued": 0,
    "admitted": 0,
    "completed": 0,
    "resident_read_bytes": 0,
    "fallback_read_bytes": 0,
    "canceled": 0
  }
}
```

上面数字**只是 Schema 示例，不是预计结果**。必须从实际执行读取并校验，不能写死。`logical_read_digest` 是逻辑需求的**多重集合**摘要（无视发射时间/物理地址的顺序差异），可在 on/off 和四种 Memory 模式之间对比；Payload/Shadow 校验则负责证明 HBF 与 SRAM 实际数据来源一致。若不实现 Shadow Payload，必须明确标记 **DATA_NOT_VERIFIED**，不得声称“读数据通过”。

**实施门槛**：先使 **HBFine 4F1M** 在 `prefetch=off` 和 `prefetch=on` 下都满足 `mismatched_bytes=0`、实际检查字节数大于 0、Partition Oracle 通过、逻辑需求摘要一致；`prefetch=on` 必须至少一次成功 Fill 和 SRAM Hit（否则不能证明 Prefetch Engine 已工作）。然后再处理 EventLab/FlashLab。最后 FullHBM 作为数值/逻辑读基准。

## 8. Qwen3-30B-A3B：统一的并发验证方案

- Workload：`frontend/asm/qwen3/flash-accel/one_block.progir`；参数 `frontend/asm/qwen3/b1_ctx512.json`（B=1, E=128, TopK=8, S=4），固定 `ROUTE_IDS`；不要换成只测一个 Expert 的 `--focus-launch`。
- Hardware：4 HBF + 1 Naive HBM 的 `configs/memconf/4F1M_grouped_planes.yaml`；派生 HBFine/EventLab/FlashLab 仅更换 Backend 类型，保持**相同 geometry、SRAM、address policy**。FullHBM 为 `configs/memconf/full-hbm.yaml` + Naive HBM。不要使用 Ramulator2 做本次 HBM 基准。
- 每个 Backend 进行 `prefetch=on/off` 两组。FullHBM 的 `on` 明确无 HBF Prefetch（应等于 `off`），这是程序开关的空操作自检，不是预取收益测试。总共 8 个 Case。
- `on/off` 必须使用**同一计算 Partition、GEMM Tiling、Tensor 物理 Placement（同一 Backend 内）**，只有 Prefetch Engine 的启用状态不同；否则不能对比 Prefetch 的效果。
- **总体墙钟时间上限 30 分钟**，启动全部 Case 的独立进程并发执行，未结束的进程在 Deadline 被标记 `TIMEOUT/SUSPECT_LOOP`。禁止只人工逐条运行命令。若单机并发导致内存不足，记录并采用受控 `MAX_PARALLEL` 重新执行，但不能删除失败记录，也不能默默改模型规模。
- 记录每组 `Cycles`、`Wall`、Python/C++ 单独 CPU Time、Python/C++ 各自 Peak RSS、同时峰值 RSS、Prefetch counters、Data Check，输出简单 Markdown/CSV。

### 8.1 交给 Agent 创建的 Bash 脚本（必须是纯 Bash；不得嵌入 Python 代码）

在仓库中新增 **`scripts/validate_flashaccel_qwen.sh`**，内容如下。可以调用仓库**已有的 Python Frontend CLI**，但脚本本身不能使用 `python -c`、here-document Python、或额外的小 Python Runner。脚本假设 Linux、GNU `timeout`、`setsid`、`jq`、Bash 4+（本轮不要求跨平台）。

```bash
#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."
for tool in timeout setsid jq sed grep; do
  command -v "$tool" >/dev/null || { echo "missing: $tool" >&2; exit 2; }
done

OUT="${OUT:-results/flashaccel-qwen-$(date +%Y%m%d-%H%M%S)}"
RUNNER="${RUNNER:-build-backends/gtsim_program}"
POLICY="${POLICY:-configs/flashaccel/qwen30b_repro.json}"
PARAMS="${PARAMS:-frontend/asm/qwen3/b1_ctx512.json}"
SOURCE="frontend/asm/qwen3/flash-accel/one_block.progir"
ARCH="configs/v0.1.0/h100-sxm.yaml"
HBF_BASE="configs/memconf/4F1M_grouped_planes.yaml"
HBM_BASE="configs/memconf/full-hbm.yaml"
MAX_CYCLES="${MAX_CYCLES:-20000000}"
PER_CASE_TIMEOUT=1750  # 30 min global budget, spare 50 s for checks/report
mkdir -p "$OUT/configs"

# One physical geometry for all three HBF backends; only the backend differs.
grep -q '^  backend: hbfine$' "$HBF_BASE"
cp "$HBF_BASE" "$OUT/configs/hbfine.yaml"
sed 's/^  backend: hbfine$/  backend: eventlab/' "$HBF_BASE" > "$OUT/configs/eventlab.yaml"
sed 's/^  backend: hbfine$/  backend: flashlab/' "$HBF_BASE" > "$OUT/configs/flashlab.yaml"
cp "$HBM_BASE" "$OUT/configs/fullhbm.yaml"

pids=()
labels=()
started=$SECONDS

for backend in hbfine eventlab flashlab fullhbm; do
  for mode in off on; do
    name="${backend}-${mode}"
    dir="$OUT/$name"
    mkdir -p "$dir"
    memory="$OUT/configs/${backend}.yaml"
    cmd=(python3 frontend/tools/run_qwen_simulations.py
      --source "$SOURCE" --params "$PARAMS" --runner "$RUNNER"
      --architecture "$ARCH" --memory "$memory" --out "$dir"
      --flashaccel-config "$POLICY" --prefetch "$mode"
      --verify-memory-reads --max-cycles "$MAX_CYCLES")
    printf '%q ' "${cmd[@]}" > "$dir/command.txt"
    printf '\n' >> "$dir/command.txt"
    # Each invocation is independent. GNU timeout terminates its command
    # on deadline; setsid groups Frontend and its GTSim child for cleanup.
    setsid timeout -k 10s "${PER_CASE_TIMEOUT}s" "${cmd[@]}" \
      > "$dir/driver.log" 2>&1 &
    pids+=("$!")
    labels+=("$name")
  done
done

failed=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  name="${labels[$i]}"
  if wait "$pid"; then
    echo "PROCESS_OK $name"
  else
    code=$?
    echo "PROCESS_FAILED $name exit=$code (124/137 usually timeout)" >&2
    tail -n 30 "$OUT/$name/driver.log" >&2 || true
    failed=1
  fi
done

elapsed=$((SECONDS - started))
if (( elapsed >= 1800 )); then
  echo "GLOBAL_DEADLINE_EXCEEDED elapsed=$elapsed" >&2
  failed=1
fi

# Never substitute success of process for correctness of memory contents.
for name in "${labels[@]}"; do
  result="$OUT/$name/result.json"
  if ! [[ -s "$result" ]] || ! jq -e '
      .returncode == 0 and (.timing.total_cycles > 0) and
      (.data_check.checked_reads > 0) and
      (.data_check.checked_bytes > 0) and
      (.data_check.mismatched_bytes == 0) and
      (.data_check.partition_oracle_pass == true) and
      ((.data_check.logical_read_digest | type) == "string")
    ' "$result" >/dev/null; then
    echo "DATA_OR_CYCLE_FAILED $name" >&2
    failed=1
  fi
done

# Compare demand-read semantics across on/off within each backend.
for backend in hbfine eventlab flashlab fullhbm; do
  a="$OUT/${backend}-off/result.json"
  b="$OUT/${backend}-on/result.json"
  if [[ -s "$a" && -s "$b" ]]; then
    lhs=$(jq -r '.data_check.logical_read_digest' "$a")
    rhs=$(jq -r '.data_check.logical_read_digest' "$b")
    if [[ "$lhs" != "$rhs" ]]; then
      echo "READ_DIGEST_MISMATCH $backend" >&2
      failed=1
    fi
  fi
done

# Prefetch must actually do work on the first fully validated backend.
if [[ -s "$OUT/hbfine-on/result.json" ]] &&
   ! jq -e '(.prefetch_stats.completed > 0) and
             (.prefetch_stats.resident_read_bytes > 0)'
     "$OUT/hbfine-on/result.json" >/dev/null; then
  echo "HBFINE_PREFETCH_NOT_EXERCISED" >&2
  failed=1
fi

printf 'case,cycles,wall_seconds,checked_bytes,mismatched_bytes\n' \
  > "$OUT/summary.csv"
for name in "${labels[@]}"; do
  result="$OUT/$name/result.json"
  if [[ -s "$result" ]]; then
    jq -r --arg name "$name" '[ $name,
      (.timing.total_cycles // 0), (.elapsed_seconds // 0),
      (.data_check.checked_bytes // 0),
      (.data_check.mismatched_bytes // -1) ] | @csv' "$result" \
      >> "$OUT/summary.csv"
  fi
done

echo "FINISHED elapsed=${elapsed}s failures=$failed output=$OUT"
exit "$failed"
```

**Agent 必须完成的脚本配套接口**：为 `run_qwen_simulations.py` 增加 `--flashaccel-config`、`--prefetch {on,off}`、`--verify-memory-reads`；最终 `result.json` 必须包含上面声明的 `data_check` 与 `prefetch_stats` 字段。脚本只**调用**这个 CLI，不允许把编译/预取算法重新写一遍。若某 Backend 超时或返回未验证数据，脚本必须退出非零并保留日志。

> 若要求 30 分钟内全部运行，应在 Agent 执行时先建立并行资源预算。上面脚本默认同时运行 8 个 Case，适合有足够 CPU/RAM 的服务器；机器不足时，可对脚本增加 Bash 原生并发上限（仍须多进程），不能把错误归咎于死循环而忽略 OOM。模拟耗时尚未实测，**不能承诺一定在 30 分钟内成功完成**。

## 9. Agent 的明确实施阶段（设置门槛，避免乱改）

1. **锁定基线**：保留 Git Worktree 与参数、配置 Hash。读取论文 §V-A/§VI-B3/Fig.12。先跑少量基础用例验证现有程序入口，不要一开始就启动 8 个大 Case。
2. **Frontend Partition**：完成 `TensorPartitionView`、N-slab 容量策略、GEMM Tiling 固定参数、`origin_launches` 重写、跨 Part 的计算依赖和 CPU 数值 Oracle。**这一步独立于 Prefetch 后端验证**。
3. **地址映射统一**：同一个 Memory YAML 驱动 Frontend budget/placement 与后端 decoder；做 M=1/32、linear/megapage Round Robin、尾部、不连续 N-slab 的地址测试。
4. **Prefetch Pass**：根据新的分区后拓扑发射 Queue/Release，满足 QKV/O/Router/GateUp/Down 顺序、正确 Expert 集与首用顺序；测试“一个 Program 中有被切分与未切分 Kernel”的完整执行，不得再有 old/new Launch 命名错位。
5. **唯一 SRAM HardwarePrefetchEngine**：迁移并删除重复 `MemorySystem` / `SramBackend` 队列状态；保持 FIFO/HOL；实现 per-page ready、Fallback、Release 与 `cancel_queued_on_demand_miss` 的开/关。写 Engine 的小规模独立状态机测试。
6. **HBFine 优先完成**：`prefetch on/off` 和小型真实 Shadow Bytes Probe 均正确，再跑一次**Qwen 单层完整程序**；必要时更改自有 HBFine 以保证多 Outstanding 的时序和 MegaPlane 统计。**此 Gate 未通过前，不要扩展修改 EventLab/FlashLab。**
7. **EventLab、FlashLab 适配**：同一 grouped 4F1M Geometry；优先改 GTSim Adapter，只有证明 Native API 缺失时才改第三方；尽量不碰 Native 实现。逐个跑 on/off 与 Shadow Read 检查。
8. **8 Case Bash 矩阵**：只使用 `scripts/validate_flashaccel_qwen.sh` 发起并发任务；整体 Deadline 1800s；到 Deadline 根据超时和日志做一次死循环/资源耗尽排查，不要只报告 “still running”。
9. **清理与交付**：删除旧 `split_region`、重复的 PrefetchWindow 生成路径、旧 Manifest Adapter、重复队列与无调用者 API。C++ clang-format、Python ruff format、Debug assert / Release Build；报告 C++/Python LOC 净变化与 CPU/RSS/Cycle。**不新增兼容层，不保留旧实现；逻辑尽量单一所有权、组合优于多级继承，模板仅在真正有重复类型逻辑时使用。** Commit 并 Push 到 `custom`（先核对 remote/branch，不强推）。

### 完成定义

- [ ] 复现 Fig. 12 的 **Sub-Tensor Partition → Prefetch → Partial GEMM → Release → 跨 Kernel 重叠**，与计算 Tiling 严格区分。
- [ ] `BM/BN/BK` 由固定参数控制，不进行 GEMM Auto-Tuning；一个 GEMM 的 N Partition 不会丢数据、重复输出或改变 reduction 语义。
- [ ] 4F1M 的 M=32/M=1 与 Address Mapping Policy 可配置，Frontend 与 Backend 对地址/容量一致。
- [ ] 计算按真实 Qwen 首用顺序 Prefetch，Expert GateUp/Down 按实际 Route 和 CTA 顺序，不引入预测式预取。
- [ ] 一个 SRAM-side Engine，Raw HBF Backends 不含第二个 Engine；FIFO/HOL 保持原状。
- [ ] Miss Cancel 有开关且默认关闭；取消排队项不影响已提交 Fill 或 Demand 正确性。
- [ ] HBFine `on/off` 的真实读数据验证（不是只测 Cycle）首先通过；EventLab、FlashLab 和 FullHBM 均有测试结果与明确状态。
- [ ] Bash 启动多进程的完整 Qwen3-30B-A3B 一层 8 Case；30 分钟 Deadline 内汇总成功、失败、超时，全部输出留存可复核。
- [ ] 新代码没有旁路的 Legacy 路径、重复 IPC/Prefetch Queue、过度防御/Exception Framework；生产代码质量与复杂度优先于兼容性。

## 10. 来源与实现参考

- FlashAccel v2: `docs/related/flash-accel/2607.10186v2.pdf`；公开摘要：<https://arxiv.org/abs/2607.10186>。重点 §V-A, Fig.7 / §VI-B3, Fig.12 / §VII-A。论文**没有**规定 MoE 各 Expert 在同拓扑级的唯一 tie-break，也**没有**规定使用 GTSim 的 M/N/K 内部具体切块参数。
- 当前 GTSim：<https://github.com/Hypersonic-cpu/GTSim/tree/a2d918204c9163b85e559ac6480f29805f8a2fa7>，重点 `frontend/src/compile/opt/prefetch.py`、`frontend/src/compile/streaming.py`、`frontend/lib/grouped_gemm.flowir`、`src/runtime/kernel_prefetch_session.h`、`include/memory/system.h`、`src/memory/sram.cpp`、`src/memory/{hbfine,eventlab,flashlab}.cpp`。
- HBFine 自有实现：<https://github.com/Hypersonic-cpu/HBFine/tree/abda9f3b946c073a5a1037f9ef963845f2e5325a>，重点 `include/hbfine/address.hpp`、`src/controller.cpp`。
- Qwen3 MoE fused GateUp/Down 语义：<https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py>。
- 预测式 Expert Prefetch 是**不同研究方向**、本计划明确不采用：<https://arxiv.org/abs/2607.24787>、<https://arxiv.org/abs/2511.10676>。
