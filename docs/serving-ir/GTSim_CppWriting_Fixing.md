# GTSim C++ Memory / Execution IR — 破坏性精简审计与执行计划

**审计对象**：`Hypersonic-cpu/GTSim`，`sim-validation@29bc0add93407b368dcf5305820909797e0798b5`（2026-11-14）。

**修订说明（v2）**：沿用原版 P0-01～P0-07、D01～D16、H01～H24（共 47 项）的代码审计与优化设计；**仅调整错误处理和断言策略**：Debug 用标准 `assert`，Release 不要求诊断性崩溃，不新增 `abort` / `fatal` / `GTSIM_CHECK`。

**范围修订（v3）**：L2 cache 及 GPU 片上执行模型（包括片上 SRAM、Scheduler、Subpartition、SM/TMEM/NoC）不在本轮实现范围内。此前针对这些文件的实验性修改必须保持撤销；本轮只允许修改 Frontend/Execution-IR/Streaming 接口和 Memory Backend 侧代码，不得改变片上模型、地址发射、Cache line 或 Cycle 行为。

**证据范围**：已扫描仓库当前 `include/`、`src/` 和 `workloads/` 下主要 C++ 源码，并重点展开检查 MemorySystem/L2/SRAM、Naive HBM/HBF、HBFine、EventLab、FlashLab、Ramulator2、Simulator、Scheduler、SM/Subpartition、Execution IR / IPC。以下为**发现的可操作问题及代码级建议**，不宣称已经通过 Profile 证明每项均能提速，也不将对外部依赖 `3rdparty/` 的修改纳入范围。

## 0. 不可误读的设计约束

- **目标不变**：最少的执行路径、最少的重复数据、最少热路径动态分配、最少源代码，同时保持模拟行为。允许破坏旧 API / 统计格式；仅保留直接 C++ 构图和 Python→C++ Streaming 两条入口。
- **本轮实现边界**：不得修改 L2 cache、片上 SRAM、Scheduler、Subpartition、SM、TMEM 或 NoC 的实现；若旧版 P0/H 项与此边界冲突，以本条为准，仅将相应条目保留为后续建议。
- **仅 Debug 断言**：直接使用 `<cassert>` 的 `assert(expr)`；Release 编译中的 `-DNDEBUG` 会移除这些断言，**不要求 Release 遇到内部不变量错误时一定崩溃**。禁止新建 `GTSIM_CHECK`、`fatal`、`abort` 包装或全项目错误处理框架。
- **检查前移且只做一次**：配置、Tensor/地址布局、CTA/SM/Warp 分配、静态 Dependency、Opcode/Backend 能力在配置装载、Graph Builder 初始化或 Graph Attach 等外层阶段检查。若只是简单的调用约定（例如非负 SM ID），不要在各下游函数重复检查。固定尺寸 C 数组的合法长度/索引尽量在建立数组时 `assert` 一次。
- **容器访问**：索引若在上游已证明合法，热路径优先 `vector[index]`、数组访问、预先缓存的指针/迭代器；若访问缺失仍需被 STL 检出（尤其是稀疏 key、第三方回调返回的 ID），可以直接使用 `vector::at` / `map::at` / `unordered_map::at`，**不必再包一层 `find` + 自定义 `throw`**。注意 `.at()` 本身是**边界检查并抛 `std::out_of_range`，并不会产生 UB**，在热路径需考虑其额外检查；绝不为省检查使用会无意插入新元素的 `unordered_map::operator[]`。
- **不再要求 Release 的显式 fail-fast**：内部不变量违例在 Debug 用 `assert` 定位；Release 使用正常 STL 语义即可，对未被验证的输入不承诺安全恢复或立即崩溃。**但不得新增 silent fallback、错误成功响应或破坏模拟正确性的未定义状态转移**；必要时在上层返回错误状态并结束执行。
- **正常硬件行为不得删**：`try_submit() == false`、队列满、资源不足、争用、依赖尚未完成和重试，是模拟结果而非非法输入。必须保持其返回值及时间行为。
- **顺序语义必须保留**：Prefetch 发行顺序、TB 派遣/退休顺序、DRAM/NoC 先后顺序、统计时间序列顺序可能改变 Cycle。`std::map` **不能一律**替换成 `unordered_map`。
- **特殊合法负数**：直接 C++ workload 中 `sm_id == -1` 有指定映射语义；`MemoryRoutingHint.consumer_node == -1` 可以代表无关联节点/Writeback。不要笼统断言所有 ID 非负。
- **对外边界保持必要限制**：IPC frame 长度上限、文件/数据读取成败、内存分配尺寸溢出、Backend 返回的 `success/error_code` 等会影响实际行为或资源安全的条件，仍在唯一入口处理；可以通过明确状态返回、Debug 断言或既有 STL 异常自然终止，而不是在每一层 `throw`/`catch`。

## 1. 第一优先级：数据结构和明显的热路径算法问题

### P0-01 `Memory::step` 每 Cycle 线性扫描等待队列，还给单节点分配 `vector<int>`

**位置**：`include/memory/cache.h:25–141`。当前 `memory_queue` / `memory_write_queue` 都是 `vector<tuple<vector<int>, bytes, completion_cycle>>`；`Memory::enqueue(node_id)` 为一个 ID 创建 `vector<int>`；`step()` 每 Cycle 使用 `remove_if` 扫描整个队列。

**实施**：将元素替换为 `struct Pending { int node_id; int complete_cycle; };`，使用两个 `std::deque<Pending>`；读写完成时间分别由 `memory_occupied_until` 保证不递减，因此 `step` 可 `while (front.complete_cycle <= now) pop_front()`。保留原有读先/写后、带宽、银行冲突计时与完成顺序。`enqueue_read/write` 合并为单个带 `is_write` 的简单辅助函数，删除无用 `bytes` 记录和多节点 vector（若当前外部有多节点调用，先用源码搜索确认并兼容等效单节点展开）。

**收益类型**：热点 O(queue size)→O(completions)，移除每请求内层堆分配；可显著减代码。

### P0-02 全局元素地址计算对每个元素重新排序维度与检查边界

**位置**：`include/memory/cache.h:438–504` 的 `compute_global_element_addr()`，特别是 `:469 std::stable_sort`；以及 SRAM 地址计算 `:210–257` 的 `compute_dim_order()` 每次调用。

**实施**：新增 `AddressPlan`（可作为当前 DAGNode 的紧凑预计算字段，或者在构图/发射一次性生成并作为参数）：包含 `ndims`、`order[Tile::kMaxDims]`、`strides`、`elem_bytes`、tile offsets、tile-major 标志；对常见 2D row-major contiguous、2D column-major、tile-major 分别写直接算式。其它 layout 只在初始化时建立小固定数组 order，无需每个元素 sort。Tensor rank/shape/Tile 范围在 Builder/Graph Attach 阶段统一确认，Debug 用 `assert`；每元素路径不再 `throw`、不构建错误字符串，Release 不需额外兜底分支。固定 C 数组的 `ndims` 上限只在初始化 `AddressPlan` 时 `assert`。

**验收**：用现有 GEMM/FA3、tile-major 和 SRAM bank-conflict 用例比较每个 subop 生成的 line 地址与 Cycle；不是只比较总 Cycle。

### P0-03 SRAM subop helper 每次创建整条 `vector<int>` 并计算不需要的其他 subop

**位置**：`include/memory/cache.h:265–385` 的 `compute_sram_subops` / `compute_sram_subop_bytes` / `compute_sram_subop_conflict`，在 `src/subpartition.cpp:402,417` 的 Pipeline 完成热路径调用。

**实施**：`subop_bytes` 直接通过 `total_elements - subop_index*elems_per_subop` 算当前 subop，O(1)；Conflict 计算只处理**当前 subop**，不重复从 0 扫描到尾。消除 per-subop `std::unordered_map<uint64_t,int> word_counts` 的反复堆分配：用固定上限 small scratch/fixed-capacity 哈希表或由 Scratch 保留容量；先比较线性小数组与复用哈希的 CPU/RSS，保证冲突计数一致。对 Swizzled 继续 0 conflict。

### P0-04 L2 Subop Lines 反复逐元素排序、哈希去重、重复展开

**位置**：`include/memory/cache.h:507–568`（每 subop 创建 `unordered_set<uint64_t> seen`）、`:655–682`（`get_subop_lines` 创建 vector + coalesce 扩展）、`:609–622`（`get_transaction_key` 每次从头循环）、`:655–660`（`get_total_transaction_count` 对所有 subop 再次构造 Lines），调用分别在 `src/scheduler.cpp:388` 与 `src/subpartition.cpp:427`。

**实施**：
1. 使用 P0-02 的已编译 `AddressPlan`。
2. `emit_subop_lines(node, subop, callback)` 为单一无堆分配生成逻辑；缓存行去重使用固定容量 scratch 或复用的小缓冲。当前 subop 独有的物理行顺序、coalesce-factor 复制必须严格保持。
3. `get_total_transaction_count()` 使用无输出计数回调，不必构造多份 vector；计算发生在 Node 首次 Issue，一次即可。
4. 实际发射 subop 调用相同生成器并写入 L2 队列或进入重试队列；需要临时 Lines 时使用复用缓冲。**不要**预存整 Kernel 的全部 line 列表，否则以内存换时间违反目标。
5. `get_transaction_key(index)` 只在 `include/tma.h:46` 等少量 fallback 场合使用，改成直接获取第一个有效行/预先确定 Base Key；禁止反复从 0 遍历。若接口不再需要，删除。

### P0-05 Scheduler 每 Cycle 排序产生临时容器

**位置**：`src/scheduler.cpp:617–626`：`std::vector<int> ranked_warps = resident_warps; std::sort(...)`；在 GTO Greedy 没有成功时频繁执行。

**实施**：增加 Scheduler 私有复用的 `ranked_warps_scratch_`，每次 `assign(resident_warps.begin(),...)`，不跨 Kernel 持有节点；继续用原 comparator 与 tie 行为排序以保证调度 Cycle 相同。只有 Profile 证明值得且有逐 Cycle trace 对比时，才改为更复杂的 O(W) 选择算法。

### P0-06 SRAM 预取容量判定每次全量复制资源数组

**位置**：`include/memory/sram.h:41–70`：`auto planes = slot.planes; auto stacks = slot.stacks;` 对每次 reserve 复制完整状态。

**实施**：使用“事务性差量预留”：先在小型 `delta_plane` / `delta_stack` 累积本次请求修改，检查 `used + delta + page_bytes <= limit`；全部成功后才写入 `slot`；失败则不产生副作用。把 `slots_` 从 `std::map<int, Slot>` 改成 `std::vector<Slot>`，按 `buffer.count` 一次性 resize。保留按输入页顺序的 plane/stack 回退规则、epoch owner、Live、peak 的更新顺序；可将 Reservation 的每页记录压缩为相同 unit 的计数（后续可选）。

### P0-07 命令流异常链膨胀且重复校验

**位置**：`src/runtime/streaming_program_runner.cpp:80–528`（约 46 处 throw/catch）；`src/execution_ir/kernel_graph_builder.cpp:60–244`（约 23 处 throw）。`stream_protocol.cpp` 还有约 9 处协议异常路径。

**实施**：
- `ProgramBegin/Manifest`、`KernelBegin`、`KernelEnd`、IPC decode 作为**唯一协议边界**，保留 frame size、解析返回值、版本/消息类型等真正必要的检查。异常输入在边界返回失败状态（由顶层进程报告），或 Debug `assert`；不写新的 `fatal`/`abort` 函数。
- `KernelGraphBuilder::begin` / Graph Attach 一次确认 Kernel 的 Tensor/CTA/SM/warp 绑定和依赖顺序；`add_instruction` 不再重复做 dtype/opcode/shape/资源合法性长分支。对连续 ID 和已经构造的依赖使用 Debug `assert`；对应的稀疏索引如需检查可直接 `.at()`。
- 删除 `send_error` / `catch(...) { builder.abort(); detach_graph(); throw; }` 等仅用于异常传播的兼容层。正常完整运行依靠 `unique_ptr`、`detach_graph` 管理资源；外部 IPC/Glaze/第三方库若抛异常，不额外建立复杂 catch/rethrow 链。若当前 Python 需要收到失败消息，只在**最外层**集中返回错误状态，避免重建逐层 Error Protocol。
- 删除 `ProgramReady/KernelBegin` 的重复字段检查与 `46` 处“检查→构造字符串→抛异常”模式；不要求每一个旧 `throw` 都替换为对应的 `assert`：**可由上游不变量保证的，直接删掉检查**。

## 2. 第二优先级：容器选型与不必要的状态

| ID | 位置 | 已观察到的结构 | 明确替换方式与注意点 |
|---|---|---|---|
| D01 | `include/execution_ir/kernel_graph_builder.h:56–65` | `map<int,int>` ×4、`set<int>` ×3、`map<int,uint64_t>` | Frontend Tensor/CTA ID 为本地连续编号时：`vector<int>` 存 Tensor binding、SM、warp base、CTA warps；`vector<uint8_t>`/bitmap 标记 read/write/prefetch consumer；`vector<uint64_t>` 存 output area。对通用输入只需一次检查连续性，或对非连续情况在 Graph Begin 统一重编号。不要将稠密整数 map 简单换成 unordered_map。 |
| D02 | `src/execution_ir/kernel_graph_builder.cpp:22–46` | 热路径的 `static std::map<string,string> supported_ops` | 用 `std::unordered_map<std::string_view, PipelineKind>` 的常驻表，或利用现有 `OpType` → Pipeline enum 直接查表。避免每条 Instruction 做有序字符串查找与重复 `parse_dtype`。 |
| D03 | `include/runtime/streaming_program_runner.h:52–56` | `launches_`, `prefetch_regions_`, `aliases_`, `events_` 为字符串 key `std::map` | 全部按精确 ID 查询，改成 `unordered_map`；迭代 Launch 顺序以 `commands_` 为准，不靠哈希遍历顺序。Manifest 的 `sequence` 用连续数组/计数而非 `set`。 |
| D04 | `src/runtime/kernel_prefetch_session.h:33` | `std::map<int,int> ids(bindings...)` | 连续 Local Tensor ID → `vector<int>`，窗口 consumer 的 `owners_` 已是 vector，应保留。 |
| D05 | `include/memory/system.h:334` | `std::map<uint64_t,ReadAlias> queued_prefetches_` | 已有 `prefetch_queue_` 明确顺序，map 仅通过 ID 精确查找；改成 `unordered_map` 不影响队列遍历。 |
| D06 | `include/memory/sram.h:115` | `std::map<int, Slot> slots_` | Buffer slot 范围是 `[0,count)`，改 `vector<Slot>`；详见 P0-06。 |
| D07 | `src/memory/sram.cpp:168` | `Region::ready_pages` 使用 `std::set<uint64_t>` 仅 insert/contains | 连续预取区间使用相对于 `address/page_bytes` 的位图 `vector<uint64_t>`，O(1) readiness 检查；区间按页向下取整，保留边界页。低风险过渡方案 `unordered_set`。 |
| D08 | `src/memory/eventlab.cpp:18,165–180` | `pages_` `std::set<uint64_t>` 仅 contains/insert | 改 `unordered_set` 或紧凑 bitmap；预加载本身的页迭代顺序来自显式 `for(page=first;...)`，不依赖 set 排序。 |
| D09 | `include/subpartition.h:43`, `src/subpartition.cpp:334,369,444,469` | `unordered_map<int,string> pending_memory_ops` 只 insert/erase/empty | 改 `unordered_set<int>`。无需存每个 Pending 节点的 Opcode 字符串；进一步可用 Node ID 位图但需规划生命周期。 |
| D10 | `include/dag_node.h:690–766` | 同时持有 `nodes`, `nodes_dict`, `nodes_by_name` 三套节点索引 | 对新执行路径及 C++ workloads 确认可强制 `node->id == nodes.size()`；随后以 `nodes[id]` 代替 ID→Node 哈希表，删除 `nodes_dict`、`nodes_by_name`（仓库内主要使用者是 DAG 本身），`add_issue_dependency` 和 `build_dependency_graph` 直接按 ID 查；保留 `name` 仅用于日志。Graph Attach 时 Debug `assert` 一次 `dep_id < nodes.size()`；后续直接按连续 ID 查找，必要的非稠密 lookup 用 `.at()`。|
| D11 | `include/sm.h:43,46`, `src/sm.cpp:25–55,192–216` | `set<int> all_tb_ids/resident_tb_ids` | **不能改 unordered_set**：Set 的升序迭代决定 TB 派遣/退休顺序。若需要简化和去堆分配，改为排序去重 `vector<int> all_tb_ids`，小规模有序 `vector<int> resident_tb_ids` + `lower_bound` 插入/移除，保持精确升序。|
| D12 | `include/memory/naive.h:163` | `map<uint64_t,uint64_t> read_bandwidth_bins_` | **不能直接换 unordered_map**，`bandwidth_samples()` 按时间顺序输出。仅在确认 bin 时间索引单调、分布合适时改为基于基准 bin 的 `vector<uint64_t>`；否则保留 map。|
| D13 | `src/memory/sram.cpp:210`, `:164–198` | `map<uint64_t,Region> regions_` 并在每 Cycle 从头遍历提交后台请求 | **不能直接换 unordered_map**：Token 升序遍历就是 Prefetch 先后顺序。若消除树节点开销，使用保持相同 token 优先级的有序活动队列 + token 查找 HashMap；只扫描尚未完全提交的 Region，跳过已提交或未启动的 Region。 |
| D14 | `include/memory/base.h:60–68`、`include/memory/system.h:257–278` | `map<string, uint64_t>` 指标与两级 MemoryTier 快照 | 这些主要是**低频输出**，不是热点。为确定性 JSON/CSV 可保留；若简化二级 TierSnapshot，使用 `array<Snapshot,2>`，边界再形成字符串 map。不要为了统一 unordered_map 牺牲结果的稳定顺序。|
| D15 | `include/arch_conf.h:74–156` | `DuplicateCheckingMap` 继承 `std::map`，YAML parse 及 normalize | 一次性 Cold Path，优先保留确定性和对错误配置的可定位失败。可以破坏性删旧版 YAML 别名、normalize 和重复 key 校验，但必须决定唯一受支持 Schema，且必须先运行配置加载 smoke tests；收益主要是 LOC，不是仿真 CPU。|
| D16 | `include/memory/system.h:332, 225–231` | 每个 Node ID 持有 `shared_ptr<vector<ReadAlias>>`，完成时再查 map | 若读别名由 Kernel Prefetch Window 唯一拥有，可用 `vector<int> node_to_window` + `vector<WindowAlias>`，只存小整数索引，避免 per-node HashMap/shared_ptr 复制；需保证后台请求与已退休 Window 不引用失效对象，不能简单使用悬挂 `span`。|

### 不能盲目替换为 HashMap 的实例

`SramBackend::regions_`、`SM::all_tb_ids`/`resident_tb_ids`、`HBMModel::read_bandwidth_bins_`、`read_aliases_`（已排序并 `upper_bound`）、NoC 请求队列、Ramulator 写完成 priority_queue。这里的顺序具有物理或输出语义。**排序要求明确时，优先选择“排序 vector/平坦数组”，其次 tree map；不得为 O(1) lookup 改变调度。**

## 3. 第三优先级：其他可验证的热点和多余检查

| ID | 位置 | 现状 / 问题 | 明确修改 |
|---|---|---|---|
| H01 | `include/memory/cache.h:630–647` + `include/memory/system.h:432–463` | L2 `enqueue_transaction_key` 已 `validate_access`，MemorySystem `route_request` 再验证同一次请求 | 在最上游完成 Tensor placement/Access 验证；下游删除重复校验，在 Graph Attach 处用 Debug `assert` 一次，Release 热路径直接发射，不在两层重复 `validate_access`。注意 writeback / FullHBM override 要保持 tier 语义。 |
| H02 | `include/memory/cache.h:795–839` | Cache Hit 找一次哈希，`touch_lru` 又找一次 | `touch_lru(iterator)` 直接复用 `cache_it`；插入处同理复用已有迭代器。 |
| H03 | `src/subpartition.cpp:35–65` + `:393–405` | `get_pipeline_throughput` 在每个 subop 完成时按字符串线性查找 Pipeline | 初始化 Subpartition 时缓存 LD/ST 等 Pipeline 带宽整数或 Pipeline 指针；以 `OpType` 分发，消除 repeated string compare。 |
| H04 | `src/subpartition.cpp:270–515`, `src/sm.cpp:125–216` | 每 Cycle `completed`, `all_completed`, `completed_tbs`, deferred 列表创建并复制临时 vector | 优先复用成员 scratch (`clear` 保留 capacity)，或使用 `std::span`/输出缓冲，保证完成事件顺序不变。勿无证据全面重写 API；按 Profile 优先。 |
| H05 | `include/memory/system.h:45–65,115–145,648–699`, `include/noc.h:94,191,348` | 逐 Cycle 返回多个临时 `vector<Request/Response>` | 按组件引入复用 scratch、批处理 callback 或输出参数，避免有结果的周期反复分配；空周期 vector 默认构造通常便宜，不要误认为每次都会分配。只在确认 Profile 热点时做。 |
| H06 | `src/memory/eventlab.cpp:87–130` | 每次 `try_submit` 构造 `batches(stacks_)` + 每 Stack `per_channel(channels_)`，背压重试会重复分配和组包 | EventLabBackend 成员中保留按 Stack/Channel 的 Scratch（每次调用开始 clear），重用容量；或者将容量 preflight 与实际构造合并成一次通过，先保证跨 Stack admission 原子性。 |
| H07 | `src/memory/sram.cpp:164–198` | 每个 Core Cycle 扫描所有 Region，重复 `regions_.at`、`pending_.contains`、`background_.contains` | 活动 Region 有序队列；背景 ID 使用与 Foreground ID 保证不相交的高位区间，生成时不需要双 HashMap contains while（配置一次验证 ID 范围）。发射、背压与 Prefetch 顺序必须精确一致。 |
| H08 | `include/memory/clock.h:15–34`, `src/memory/ramulator2.cpp:65,95`, `src/memory/hbfine.cpp:143` | 每次时钟转换/提交都 throw if negative or reversed time | Graph/Simulator 的计时不变量在初始化/单一 `step` 边界确认；热路径删除重复动态错误构造；第三方 Clock/Callback 的不变量在转换边界使用 Debug `assert`；若第三方返回显式错误码则依照上层状态返回机制处理，不新增 `abort`。 |
| H09 | `src/memory/naive.cpp:20–24`, `src/memory/hbfine.cpp:24–33`, `src/memory/ramulator2.cpp:65–77`, `src/memory/sram.cpp:29–40` | 每 Backend 对上游已确认的请求又检查大小、类型、地址、请求唯一性 | 将规则集中在 Graph/MemorySystem 接口层；Backend `try_submit` 只负责 capacity/physical scheduling；第三方回调状态不变量 Debug `assert`，必要的错误码通过上层状态返回，不得误报成功。 |
| H10 | `include/memory/cache.h:735–756`, `include/memory/system.h:535,593,682`, `src/memory/*` callbacks | Completion 未对应 Request/Route/MSHR 时 throw；这类是真正内部数据损坏 | Debug `assert` 核对关联关系；对未能静态证明存在的 key 使用 `.at()` 直接访问。**不得解引用 `end()`，也不得返回默认数据掩盖错误**；不保证 Release 崩溃方式。 |
| H11 | `src/subpartition.cpp:128–158,283–392` | 对不存在的 Scheduler Node ID 多次 silent `return/continue`；易掩盖未知完成事件 | 明确跨 SP 可预期的 forwarded completion，与本地必然存在的 node 区分；后者在初始化后直接 dense lookup（Debug `assert` ID 范围）或使用 `.at()`，不能静默丢失完成；删除不必要的 `find`。 |
| H12 | `src/sm.cpp:62` 左右 `is_tb_complete` | 不存在的 TB 当前视作“已完成” | 初始化建立真实 TB，合法情况下无需分支；Debug `assert(present)`，Release 直接访问经构造期确保有效的 dense count；不要设计未知 TB 时返回 `true` 的静默成功分支。 |
| H13 | `src/subpartition.cpp:164–220` | NoC 跨 Cluster 访问警告后**悄悄按本地 SRAM 回退** | 目标是清晰、不可意外的架构模型：不支持跨 Cluster 时在构图/绑定阶段 Debug `assert(noc->same_cluster(...))`；运行时不再警告后回退到本地 SRAM。如果正式研究要保留 fallback，必须是显式配置，不允许自动。 |
| H14 | `include/tmem.h:242–245`, `reset():16–30` | 四个跨 CTA Pair 状态是 `inline static unordered_map`，全局跨 Simulator/SM 共享，Reset 会清空别的实例的状态 | 引入 GPU 实例所有的 `TMEMPairCoordinator`（4 Map 移入其中），所有 SM 的 `TMEMUnit` 持有对同一 Coordinator 的引用；GPU 生命周期内创建，Kernel attach/detach 时由 GPU 统一 reset。**不能移动到每个 TMEMUnit 自己的实例成员**，否则跨 CTA/跨 SM 的配对握手失效。 |
| H15 | `src/simulator.cpp:420–1430` 和 `run():375–415` | `print_results` 大量根据 Node Name/Workload 解析并反复扫描 DAG；`run()` 结束时无条件调用 | 与数值执行核心分离；默认仅保留简洁 `Summary/KernelTiming/MemoryStats`，并在运行过程中累积必要统计；旧按名字推导的 debug 报表可破坏性删除或放进显式编译/运行开关，不能默认扫描整张 DAG 多次。确保外层 `Program timing JSON` 的 Cycle / HBM-HBF counters 不丢。 |
| H16 | `include/dag_node.h:690` | DAG 同时有自有 `tensor_registry` 和非自有 `tensor_registry_view` | 对新程序继续借用 Program Registry，对直接 C++ workloads 只保留单个拥有所有权的 WorkloadGraph 容器；如果重构会触及过多 Workloads，可暂时保留这一个小耦合，不要为了它增加 Adapter 类。 |
| H17 | `src/runtime/stream_protocol.cpp`、`include/runtime/stream_protocol.h` | Socket Frame 错误路径 throw 多处、Glaze decode 单独报告错误 | 提供最小 `read_frame()/send_frame()`，在 framing 入口处理 EOF/读取失败与非法 Frame：保留 1MiB 大小限制，返回失败状态给最外层入口或在 Debug `assert`；不引入 `fatal`。Linux/macOS 两套编译验证 Apple Clang 类型链接修复。 |
| H18 | `workloads/{gemm,fa2,fa3}/*.cpp`, `workloads/common/workload_options.h`, `workloads/serve/program_runner.cpp` | CLI parse 和 Kernel 约束中 throw/catch、重复的参数检查 | 只在最上层参数解析处保留必要错误检查，失败由 `main` 直接打印并返回非零退出码（或 Debug `assert`）；直接构图内部只接受已验证 options，去掉重复 `throw`。把非常长的 workload-specific CLI switch 仅合并真正共享逻辑，不加类层次。 |
| H19 | `src/runtime/kernel_execution_session.cpp:7–40` | Kernel 每次退休扫描所有 SM/SP/Queue 检查完成不变量 | 这是**非热路径**（每 Kernel 一次），不应优先删掉；Debug 保留一次 `assert` 完成状态检查，直到可靠的 Pending Counter 取代扫描；若释放 DAG 的前置条件决定指针是否仍被正常使用，**不要仅靠 Debug assert 来完成实际同步/等待**。Retire 的真实状态机及必要等待逻辑在 Debug/Release 都必须保留。 |


| H20 | `src/memory/flashlab.cpp:54–68,82,109–112` | `try_submit` 对每次请求再次验证大小/地址、时间前沿以及每次 Native Completion 检查 | 数据形状和有效 Tier 在上游确定；删除普通请求的冗余检查；**native session completion 个数/时间有效性属于第三方返回契约，在调用边界 Debug `assert`，或按现有失败状态上报；不在子流程新增 `throw`。**`run_batch` 是固有模型调用，不应为了减少 LOC 改掉。 |
| H21 | `src/memory/eventlab.cpp:27,73,87–91,143,161,176` | 配置、请求、completion callback 和 preload 分散抛异常 | 请求格式校验上移；Unknown Child ID 使用 `.at()` 或 Debug `assert`；EventLab Engine 停滞/失败、Preload Failure 在上层统一检查结果状态，不新增 `abort`。不能把失败 callback 当成成功以缩短模拟。`std::vector` 重复组包参考 H06。 |
| H22 | `include/memory/base.h:89–119` | `MemoryBackend` 默认虚函数的“不支持功能”行为通过 5 处 `throw` 实现 | 对不支持的预取/Locate/Preload 功能，先按 Backend 能力在**构造/Program 初始化**决定是否合法，Hot Path 不再探测能力；不支持的默认虚函数若必须存在，`void` 返回者可以使用 `assert(false && "unsupported backend operation")`；**有返回值的函数不得只写 assert 而在 Release 缺少 return**。优先移除不适用的虚接口或使用 pure-virtual + 明确的后端实现；有正常“不支持”返回语义时直接返回 `false`/`0`，不新建检查宏。不要删除 Runtime 可选择 Backend 的虚分发。 |
| H23 | `include/hbm_address.h:27–57`, `include/memory_layout.h:42–60` | 解码/分段访问路径包含多处范围与栈数检查 | Stack 数和 Segment Layout 在配置阶段检验一次；常用地址解码使用前置的有效地址不变量。对 Backend/NoC 外部引入的无法静态证明有效的地址，在单一接收边界用 Debug `assert` 或合适的 `.at()`，无需逐层再校验。尤其不能把越界 decoder 返回合法 Channel。 |
| H24 | `src/memory/base.cpp:19–46`, `src/memory/system.cpp:9–40` | Factory/Topology/Placement 在多个入口做配置支持检查，部分是同义判断 | 创建 `MemorySystem` 前用唯一配置验证阶段确定合法组合；Backend Factory 只分发 enum、创建对象，非法枚举在配置解析/Factory 入口 Debug `assert`（或顶层返回失败状态）；删除重复的逐 Tensor HBF 权重 readonly 检查，前提是 Program TensorRegistry 创建时已经验证。该路径是 Cold Path，优先级低于 Cache/Queue。 |

## 4. 错误处理机制的唯一推荐方案（v2：标准 assert）

**不新增** `include/check.h`、`GTSIM_CHECK`、自定义 `fatal()`、`abort()`、通用异常层次或 Status Framework。直接使用标准 C++：

```cpp
#include <cassert>

// 建立对象 / Graph Attach 等冷路径：断言一次即可。
assert(config.num_sms > 0);
assert(node->id == static_cast<int>(dag.nodes.size()));
assert(ndims > 0 && ndims <= Tile::kMaxDims);

// 后续热路径不再重复断言已建立的不变量。
const auto* node = dag.nodes[id]; // ID 已在 attach 阶段统一验证

// 稀疏 key 若必须检查是否存在，直接使用 STL 的 .at()：
const auto& request = pending.at(request_id);
```

注意：`assert` 在 `-DNDEBUG` 下会被移除；`.at()` 会抛 `std::out_of_range` 而**不是产生 UB**。允许 `.at()` 使用标准库自身的异常语义，但**不要写额外的 `if (...) throw ...` 来包装它**。对于 `vector[]` 或 C 数组，必须能够由前置条件保证索引合法；不要依赖越界 UB 作为错误处理策略。

| 所在层 | 新的处理规则 | 热路径是否执行额外校验 |
|---|---|---|
| Config / CLI / JSON / IPC | 只确认解析成败、frame 大小上限、必要资源上限；无效输入在最外层报错并结束，其他简单参数按合法处理 | 否 |
| Tensor / AddressPlan / Graph Attach | 检查一次维度、内存布局、静态 ID 与映射不变量；在 Debug 使用 `assert` | 否 |
| Scheduler / SM / L2 / MemorySystem | 信任已经固定的配置和节点结构；对真正稀疏查找可直接 `.at()` | 基本没有 |
| Backend / Native Callback | 已保证的参数不重复检查；未知 ID `.at()`，Native 状态/错误码只在边界处理 | 最少 |
| Graph Retire | **必须保留**实际的等待/完成状态机；Debug assert 用于验证最终清理条件 | 每 Kernel 一次 |

### 需要保留的真实控制流（不是“过度检查”）

- `try_submit()` 的容量拒绝、NoC/Memory Queue backpressure、Prefetch 的资源争用与重试。
- IPC 的 frame 长度上限（避免异常长度导致大规模分配）、读取失败和协议解析失败的**一次性**处理。
- Native Backend 的明确失败结果不可当作成功请求计入统计。
- 前一个 Kernel 退出后，只有当底层 Pipeline、Callback、Outstanding Request 已完成/正确解绑，才可销毁 DAG。`assert()` 只能验证，不负责等待或资源回收。

### 应删除或迁移到 Debug 的检查

- 同一 Tensor/访问在 Builder、L2、MemorySystem、Backend 四处重复校验。
- 每个元素地址计算都检查 Tensor Rank、Tile 边界、溢出，并构造包含 Node Name 的异常字符串。
- 每个 Prefetch/DRAM Clock Tick 检查固定配置值和已经在构造时建立的 Channel/Stack 数目。
- 后台 Callback 查找时先 `find`、检查 `end`、再抛异常；可以直接 `.at()`，或对已建立的连续 ID 使用预分配 vector。
- 每层重复检查 Kernel/Stream metadata 和 Controller 能力；由最外层一次决定合法组合。

## 5. 精确实现顺序（Agent 不应自主重新设计）

1. **冻结可复现对照**：Git worktree 固定 `29bc0add...`，Release 构建；保留当前五个实验 T1–T5 的 Cycle、User+Sys CPU（Python/C++ 独立）、Peak RSS（独立和同时）及配置哈希；再挑一组小的 SRAM/HBFine、EventLab/FlashLab、Ramulator2、FA3、NoC/跨 CTA 原生 C++ 用例。每组 ≤900 秒；缺少可选第三方 Backend 时必须报告 skipped，不应声称通过。
2. **清理异常与断言边界**：**不新增 check.h 或任何自定义错误宏**；删除 `StreamingProgramRunner`、`Builder` 的冗余 `throw`/`catch`、重复检查，把必要验证移到配置/Graph Attach/协议入口；内部使用标准 `assert`（Debug）或已证明有效的直接访问、必要时 `.at()`。
3. **本轮不改片上热点**：P0-01～P0-04 涉及片上 SRAM/L2/address emission，全部冻结为后续建议；本轮只修复接口/Backend 侧可独立验证的结构性问题，不能用 CacheLine、bank conflict 或 Cycle 变化作为优化手段。
4. **改容器**：D01–D09，优先稠密数组和有界 scratch；D10 删除 DAG 重复索引前必须确认直接 C++ workload ID 连续；D11 使用保持排序的 vector 而不是 HashSet；D12/D13 明确记录顺序约束；D14–D16 根据 Profile 与使用者范围决定。
5. **简化 GPU/SM/Scheduler 和 TMEM**：P0-05、H03/H04/H11/H12/H14，保持逐 Cycle issue/complete trace；TMEM Coordinator 必须支持跨 SM 配对状态共享。
6. **精简 backend**：H07–H10、H16–H19，删除重复检查、仅以 Debug `assert` 验证内部不变量，同时保留真实容量拒绝；EventLab 的 Scratch 优化、RamulatorCallback 必须有小规模独立验证。
7. **瘦身 Simulator 诊断层**：H15 删除超长的 name-based 汇总/重复全 DAG 扫描，保留精确的 `ProgramTiming`/TierCounters/必要波形和可复现实验报告。不要以直接删统计为手段假装加速仿真：要求单独给出 `run()` 和报表 CPU 耗时。
8. **格式化**：对修改的 C++ 使用仓库 `.clang-format`，Linux Clang/GCC 和 macOS Apple Clang 构建（若可用）；`git diff --check`。删除死成员、无用 Header、过时注释，避免添加通用抽象框架。
9. **性能和代码量检查**：T1–T5 用 Release 测 CPU/RSS/Cycle，另用未定义 `NDEBUG` 的 Debug 构建完成轻量 C++ workload、Kernel Attach、Memory Backend smoke test，确保初始化/边界 `assert` 未触发；Debug 与 Release 的 CPU Time 不混用比较。生产 C++ 源码按模块统计新增/删除/净减少；大目标净减少 800 行以上，但**不可为了数字删除必需的设备状态**。提供主要热点的 `perf`/Instruments 证据或微基准，而非仅证明 Cycle 一致。
10. **产物和 Git**：一份精炼 Markdown 报告，列出变更文件、每项旧/新 CPU、RSS、Cycle、请求顺序、Backend 测试结果；自动 Commit；在验证 `custom` remote 与当前分支后推送，不 force、不推 origin。

## 6. 验收用例与必须观察的行为

- T1: two GEMMs FullHBM 1 SM，**38,948** Cycles。
- T2: two GEMMs FullHBM 132 SM，**14,914** Cycles（6,824 + 8,090）。
- T3: 128×256×64 GEMM，HBF 4F1M，Streaming Prefetch，**11,769** Cycles。
- T4: 512×512×512 GEMM，HBFine 4F1M，Streaming Prefetch，**25,964** Cycles。
- T5: Qwen3-30B-A3B 单层 `one_block.progir`，`b1_ctx512.json`，FullHBM、no HBF / no Prefetch，**397,765** Cycles。
- 另外：Cache-line request ID/address 序列、`subop_bytes`、SRAM conflicts、MSHR merge/response、TB 发行/退休次序、RAM/HBF DRAM transaction 时间序列必须与原版相同或明确说明修正原因。
- C++ 与 Python 的 CPU Time、Peak RSS 独立采集；同时峰值 RSS 用采样的 `max_t(RSS_py+RSS_cpp)`，不可用各进程 Peak RSS 相加代替。
- **不能用正常 backpressure 的减少/硬失败来人为缩短执行**；不得把异常判定删除成 silent success；已知稀疏 key 使用 `.at()`，连续索引必须有前置不变量。Release 不保证诊断性崩溃，但不能用未定义行为来掩盖模型错误。

## 7. 最小交付格式

- `commit`、`branch`、`custom` 推送目标和结果。
- `C++ Core` / `C++ Backend` / `Execution IR` 三块 LOC 新增、删除、净变化，另报告删掉的类/API。
- T1–T5 与 Memory/NoC/TMEM 核心测试：Cycles、Python/C++ CPU、Python/C++ Peak RSS、Combined Peak RSS；所有差异需说明。
- 分离“已实现/已测量”和“后续建议”，不得把没有实际执行的性能优化写成已获得收益。


