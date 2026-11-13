# GTSim 第二轮重构计划：破坏性精简与性能优化

代码基线： `sim-validation` 分支，提交 `fc122648`（2026-11-13）。

总体目标：删除旧执行架构，而不是继续在旧架构旁边增加新架构。 最终只保留两种 DAG 来源：

1. `workloads/` 中的 C++ 代码直接构造 DAG。
2. Python Frontend 按需生成 Execution IR，通过 IPC 交给持久运行的 GTSim。

两种入口共用同一个 DAG 执行核心。不再支持 FIFO、`.gtsir` ProgramExecutor、旧的 TileIR 执行入口。

重构的优先级明确为：

模拟正确性与性能 → 运行时内存 → 代码质量及代码量 → 兼容性。

兼容性不再是约束。允许删除旧 API、旧测试、旧工具、旧格式及其文档。不得为了保留旧功能而引入额外适配层。

## 一、预期的最终架构

C++ Workloads

直接构造 DAG

Python Frontend

紧凑 ArchIR → Instruction Batches

IPC + KernelGraphBuilder

共享 Simulator / DAG Execution Core

Attach → Execute → Retire → Detach

Persistent GPU / SM / L2 / HBM / HBF / Clock

对于 Python 路径，仍采用第一阶段的 一个 Kernel 完整接收后执行、完成后释放；不在这次重构中引入 Kernel 内 Chunk 调度。

## 二、明确删除的代码

下列文件是优先删除对象，不应再次通过 wrapper 或 compatibility class 保留。

| 文件                                                          | 处理                               |
| ----------------------------------------------------------- | -------------------------------- |
| `src/runtime/program_executor.cpp`                          | 删除旧 ProgramExecutor              |
| `src/execution_ir/graph_materialize.cpp`                    | 删除旧 DAG 物化器                      |
| `src/execution_ir/graph_codec.cpp`                          | 删除旧 `.gtsir` 读取器                 |
| `src/tile_ir_parser.cpp`、`src/tile_dag_builder.cpp`         | 删除旧 TileIR 执行链                   |
| `include/tile_ir.h`                                         | 删除旧 TileIR API                   |
| `workloads/gemm/gemm_ir.cpp`、`workloads/serve/serve_ir.cpp` | 删除依赖 TileIR 的执行入口                |
| `frontend/src/compile/export_runtime.py`                    | 删除旧 `.gtsir` 导出                  |
| `frontend/src/compile/arch_to_exec.py`                      | 删除旧完整 ExecIR 展开路径，必要的通用逻辑移到现有生成器 |
| `frontend/tools/run_qwen_simulations.py`                    | 删除 FIFO 分支和旧文件生成代码，保留简洁的流式入口     |
| `tests/fixtures/*.gtsir`                                    | 删除不再使用的旧执行格式 Fixture             |

同时修改 `CMakeLists.txt`、`frontend/src/compile/__init__.py`、`frontend/src/compile/__main__.py`、`frontend/src/compile/run_text.py`，清理所有已删除符号的引用。

只保留有实际用途的早期 IR 解析、检查和文本输出，不再允许任何生产路径先构造完整 Program ExecIR。

删除旧路径前必须先保存可执行的基线版本，用于后续性能比较。 此步骤不算保留兼容性：基线只存在于独立 Git Worktree 和 Benchmark 结果中，不进入最终代码。

## 三、C++ 核心重构：具体接口与实现

### 3.1 `Simulator`：消除两套 GPU 初始化代码

修改： `include/simulator.h`、`src/simulator.cpp`

当前两个构造函数重复创建 GPU、NoC、MemorySystem。这部分应该统一为一个实现。

保留以下接口：

```
class Simulator {
public:
    // 唯一的设备初始化实现
    Simulator(const SimulatorConfig& config,
              const TensorRegistry& tensors);

    // C++ workload 的便捷入口，委托给上面的构造函数
    Simulator(DAG* dag, const SimulatorConfig& config);

    void attach_graph(DAG& dag);
    bool run_active_graph(int max_cycles);
    void detach_graph();

    int64_t current_cycle() const;
};
```

具体要求：

- 第二个构造函数使用 delegating constructor，不得复制 GPU 初始化代码。
- `attach_graph()` 只初始化当前 DAG 所需的 Scheduler、CTA、Warp、Scoreboard 状态。
- `detach_graph()` 释放所有 DAG 引用，但保留 GPU、Cache、MemorySystem 和全局时间。
- 兼顾原 C++ workload 的 `sm_id == -1` 等映射语义，不能因为合并构造函数改变它们的模拟结果。
- `TensorRegistry` 在 Python 执行路径中由 Program 持有；DAG 不复制全局 Registry。

### 3.2 `KernelExecutionSession`：只保留必要的状态转换

修改： `src/runtime/kernel_execution_session.cpp`

现在 `detach()` 重建 Scheduler，而下一个 `attach_graph()` 又会重建一次。

改成：

```
class KernelExecutionSession {
public:
    static void attach(Simulator& sim, DAG& dag);
    static void verify_retired(const Simulator& sim);
    static void detach(Simulator& sim);
};
```

具体要求：

- `attach()` 创建或重置 Scheduler，一次即可。
- `detach()` 清除旧图的引用及局部状态，不重新分配即将被下一次 `attach()` 替换的 Scheduler。
- 若 Scheduler 支持复用，优先实现 `reset_for_graph()`，清除旧节点列表、Ready Queue、Scoreboard 绑定和 CTA 状态。
- 不因 Kernel 切换而重建 GPU。
- 禁止通过直接 `clear()` 未完成的设备请求队列来掩盖生命周期错误。正常路径应先验证必要设备操作已经完成，再解除指针引用。

不要求为此开发一个通用 Session Framework。

### 3.3 `KernelGraphBuilder`：删除重复依赖图及拓扑检查

修改： `src/execution_ir/kernel_graph_builder.cpp`、对应 Header。

这是本轮最重要的内存优化之一。

目前 Builder 持有：

```
std::unique_ptr<DAG> dag_;
std::vector<std::set<int>> deps_;
std::vector<std::vector<int>> children_;
```

随后又执行：

```
auto pending = deps_;
// Kahn topological sort ...
dag_->build_dependency_graph();
```

而 `KernelExecutionSession::initialize()` 还会再次执行 `build_dependency_graph()`。

这些全部应当精简。

设计依据：新的 Frontend 已保证 Instruction 按 ID 顺序输出，且每个依赖 ID 都小于当前 Instruction ID。

因此：

\\[ \forall (u\rightarrow v),\quad u.id < v.id \\]

这已经保证图不存在有向环。

不需要再保存第二份完整依赖图，也不需要 Kahn 拓扑排序。

Builder 改为：

```
class KernelGraphBuilder {
public:
    KernelGraphBuilder(const TensorRegistry& tensors,
                       const SimulatorConfig& config);

    void begin(const KernelHeader& header);
    void add_cta(const CTA& cta);
    void add_instruction(const Instruction& instruction);
    void add_prefetch_window(PrefetchWindow window);

    std::unique_ptr<DAG> finish(const KernelEnd& end);
    void abort() noexcept;
};
```

`add_instruction()`：

```
void KernelGraphBuilder::add_instruction(const Instruction& op) {
    const int id = static_cast<int>(dag_->nodes.size());

    if (op.id != id)
        throw std::invalid_argument("invalid instruction order");

    for (int dep : op.after_complete)
        if (dep < 0 || dep >= id)
            throw std::invalid_argument("invalid completion dependency");

    for (int dep : op.after_issue)
        if (dep < 0 || dep >= id)
            throw std::invalid_argument("invalid issue dependency");

    // 检查必要的 CTA / Tensor / Opcode 参数。
    // 直接创建 DAGNode。
    // DAGNode 自身保存依赖 ID。
    // 不建立 deps_、children_、pending 等副本。
}
```

`finish()` 只做轻量收尾：

- 检查 Instruction、CTA 数量。
- 检查必要的输出 Tensor 定义。
- 返回 DAG 所有权。
- 不再执行拓扑排序。
- 不再构建依赖图。

只在 `attach_graph()` 时调用一次 `DAG::build_dependency_graph()`。

另外明确删除：

- `instruction_names_`：Instruction ID 已是唯一标识，名称只是调试信息。
- `opcodes_`：Prefetch consumer 直接访问 `dag_->nodes[id]->op`。
- `deps_`、`children_`：完全删除。
- `collectives_` 和基于 `std::regex` 的 WGMMA 名称分组校验：交给可信 Frontend，必要时只在调试模式检查。
- `output_regions_` 的 O(N²) 重叠检查：删除。必要的输出面积只用每 Tensor 一个整数计数。
- 其他仅为重复进行结构验证而创建的大型容器。

\*\*保留必要的轻量验证：\*\*协议版本、Instruction ID 顺序、依赖 ID 合法性、Tensor 边界、CTA 资源限制、Opcode 与 Pipeline 类型。

这里的原则不是把所有检查都删掉，而是删除在可信生成器、Builder 和 Scheduler 之间重复进行的检查。

### 3.4 Kernel Bridge：简化单 Stream 的依赖状态

删除： `include/runtime/dependency_tokens.h`，将必要功能合并到执行器。

当前 `DependencyTokens` 用 `unordered_map<string, Entry>` 和引用计数实现 Token 生命周期。但第一阶段只有一个顺序执行的 Stream，且下一 Kernel 一定在上一 Kernel 完成后才装载。

因此不需要维护通用 Token Registry。

使用一个轻量 Stream Completion Fence：

```
struct StreamState {
    uint64_t completed_launches = 0;
    int64_t last_completion_cycle = 0;
};
```

每个 Kernel 的元数据保留顺序编号、Stream ID、Launch ID。执行器通过完成状态确认其前驱已经结束。

这仍然等价于我们此前讨论的 DAG Completion Bridge：

\\[ \operatorname{Complete}(K_i)\Rightarrow \operatorname{Ready}(K\_{i+1}) \\]

区别只是两个 Kernel DAG 不会同时存在，所以不用物化 Barrier 节点，也不用一个完整的依赖 Token 容器。

显式 `RecordEvent`、`WaitEvent` 仍可使用小型事件时间表实现。第一阶段不支持跨 Stream 并发，遇到不支持的跨 Stream 操作应明确报错。

### 3.5 统一 C++ Program 执行器

修改： `streaming_program_runner.cpp`。

旧 `ProgramExecutor` 删除后，新的执行器就是唯一的 Python Program 运行入口。可以将 `StreamingProgramRunner` 简化命名为 `ProgramExecutor`，不再保留旧类。

它只负责：

- 读取 Program 元数据。
- 按顺序执行 Launch、Prefetch、Event、Synchronize 命令。
- 请求、构造、执行并释放一个 Kernel DAG。
- 汇总 Timing 和 Memory Statistics。

不要再建立第二套 Kernel 执行机制。

`KernelTiming`、Prefetch 统计和 Memory 统计应使用现有结构，不复制一套新的数据类型。

## 四、Python Frontend 重构

### 4.1 保留唯一的 Execution IR 生成路径

主要修改： `frontend/src/compile/streaming.py`、`expand_native.py`。

删除旧的：

`lower_arch → 完整 ExecIR → export_runtime → 巨大 JSON`

保留：

`ArchIR → NativeExpander.expand_to → IPC Instruction Batches`

`NativeExpander` 只保留一种指令生成实现，不要同时维护 eager 和 streaming 两套逻辑。

接口建议：

```
class NativeExpander:    def __init__(self, region, resolver, tensors):        ...    def expand_to(self, emit_cta, emit_instruction) -> None:        ...
```

原先兼容 `self.nodes`、`self.ctas` 的双写分支应删除。

`compile/run_text.py` 若仅用于统计节点数量，改为给 `expand_to()` 传递计数回调，而不是创建完整 ExecIR。

### 4.2 修复 Prefetch 索引错误

修改： `frontend/src/compile/streaming.py`

目前：

```
for _, _, node_id, tensor_id, tile in eligible:    ...    current_wave = waves[_]
```

改成：

```
for cta_id, iteration, node_id, tensor_id, tile in eligible:    ...    current_wave = waves[cta_id]
```

要求增加一个包含多个 CTA、多个 Pipeline Iteration 的针对性测试，确保没有把 iteration 当成 CTA ID。

四组基准中的 HBF Prefetch 实验必须重新验证，不能因为历史周期一致就忽略这个错误。

### 4.3 删除旧 Manifest Adapter

目前 `make_program_plan()` 仍从旧 `gtsim.exec.program/1` 风格的字典读取元数据。

改成直接从源 `ProgIR`、编译后的 `FlowIR/ArchIR` 和 Prefetch Plan 生成紧凑 `ProgramPlan`。

不要再通过旧 `export_runtime.py` 获取 `PIPELINES` 等常量；把 Opcode→Pipeline 的映射留在 Python Execution IR 生成模块中，且只定义一次。

`run_qwen_simulations.py` 的 FIFO 写文件、FIFO Worker、旧结果转换等代码全部删除。

保留一个清晰的命令行入口，公共逻辑集中在 `compile/` 中，不在实验脚本里重复实现编译器。

## 五、IPC 优化：批量传输 Instruction

目前每条 Instruction 独立编码成 JSON 并发送，通常还需要多次系统调用。对于百万节点级 DAG，这会显著增加 CPU 时间。

本次直接改成有界批量传输，避免后续再重构一次协议。

建议保留 Unix Socket 和长度前缀协议，但将消息单位改为：

```
ProgramBegin
TensorDef / LaunchDef / CommandDef
ProgramReady

RequestKernel
KernelBegin
CTA
InstructionBatch
InstructionBatch
...
KernelEnd

KernelDone
```

其中：

```
struct InstructionBatch {
    std::vector<Instruction> instructions;
};
```

具体约束：

- 每批最多 512 条 Instruction。
- 每批 JSON 编码后不超过 256 KiB；达到任一限制就发送。
- IPC Frame 最大值为 1 MiB。
- Python 缓冲区只持有当前 Batch。
- C++ 将当前 Batch 解析成临时 `vector<Instruction>`，逐条加入 `KernelGraphBuilder`，然后立即释放 Batch。
- 不允许 C++ 累计多个 Batch 的 `Instruction` 容器。
- CTA 可以单独传输；遇到下一 CTA 时，先发送尚未发送的 Instruction Batch。
- `KernelEnd` 不触发额外全图复制。

这是传输批处理，不是 Kernel 内 Chunk 执行。C++ 仍在完整 DAG 构建完成后才开始模拟。

先继续使用 JSON，不在本次引入 Protobuf、自定义二进制 Instruction ISA 或复杂的 Buffer Pool。若性能证据显示 JSON 仍是瓶颈，再单独优化。

## 六、基准实验：先冻结旧版，再修改代码

本次必须报告 Simulation Cycles、CPU Time、Peak Memory Footprint。

先建立不可修改的 Git Worktree：

```
git worktree add --detach ../GTSim-baseline fc1226489474b452430d1b1b53a1aa2465adde8b
```

在该 Worktree 中保留旧 FIFO 与当前 Socket 版本，并独立编译 Release 版本。正式清理只在开发分支执行。

因此报告区分三者：

- `Old FIFO`：旧 `.gtsir` / FIFO 路径。
- `Previous Socket`：当前 `fc122648` 流式版本。
- `Refactored Socket`：本次重构后的版本。

前两者作为固定基线，最终代码不保留其执行路径。

### 6.1 必须保留的五组实验

| 编号 | Workload                          | 配置                             | 预期总 Cycle |
| -- | --------------------------------- | ------------------------------ | --------- |
| T1 | `two_gemms.progir`                | FullHBM，1 SM                   | 38,948    |
| T2 | `two_gemms.progir`                | FullHBM，132 SM                 | 14,914    |
| T3 | `example.progir`，GEMM 128×256×64  | HBF 4F1M，Streaming Prefetch    | 11,769    |
| T4 | `example.progir`，GEMM 512×512×512 | HBFine 4F1M，Streaming Prefetch | 25,964    |
| T5 | `qwen3/one_block.progir`          | Qwen3-30B-A3B，FullHBM，1 层      | 由基线运行确定   |

T1–T4 必须复用历史实验的准确参数、配置文件及 Prefetch 选项，不得为了得到对应 Cycle 人为修改时序参数。T2 分层应为 6,824 + 8,090。

T5 使用现有的：

- `frontend/asm/qwen3/one_block.progir`
- `frontend/asm/qwen3/b1.json`
- `configs/v0.1.0/memory_full_hbm-0F5M.yaml` 或经过确认的等效 Naive HBM 配置。

T5 不启用 HBF，也不启用 Prefetch，所有 Tensor 放在 HBM。测试完整的一层 MoE Block，不允许通过 `--focus-launch` 偷换成单个 Kernel。

为了满足执行时间上限，建议新增一个仅用于 Benchmark 的参数文件：复用 `b1.json`，但将 `CONTEXTS` 固定为 `[512]`，保留 `B=1`、`S=4` 和其他模型结构参数。不得修改原 `.progir`。若这个配置仍超时，可以统一降低上下文长度，但旧版和新版必须使用完全相同的参数，并在报告中注明。

### 6.2 时间限制

每组实验的旧/新成对比较必须在 15 分钟内完成，设置硬超时 900 秒。

编译时间不计入单组实验的执行时间。

超时必须终止 Python 和 GTSim 整个进程组，并记录失败，不能留下后台进程或使用不完整数据。

性能比较至少先进行一次完整成对运行。只有在预算允许时进行额外重复，不能为了重复测量突破时间上限。

### 6.3 正确统计两个进程的 CPU 与 Memory

不能只统计 GTSim，也不能把 Python 和 GTSim 的 Peak RSS 直接相加后声称是实际同时峰值。

建议统一 Benchmark Harness，在 Linux 上分别收集：

| 指标              | Python Frontend       | C++ GTSim             |
| --------------- | --------------------- | --------------------- |
| User CPU Time   | `ru_utime`            | `ru_utime`            |
| System CPU Time | `ru_stime`            | `ru_stime`            |
| CPU Total       | User + System         | User + System         |
| Peak RSS        | `ru_maxrss` / `VmHWM` | `ru_maxrss` / `VmHWM` |
| Wall Time       | 整体实验                  | 可额外记录 C++ 存活区间        |

可以通过 GNU `/usr/bin/time -v` 分别包装两个进程，或使用正确的 `wait4()`/`getrusage()` 实现。注意 Python 的 `RUSAGE_SELF` 不包括其 GTSim 子进程的 CPU 使用量。

另外，若需要报告同时驻留的总内存峰值，应监控两个 PID 的瞬时 RSS：

\\[ M\_{\text{combined peak}} = \max_t\left(\operatorname{RSS}\_{Python}(t) +\operatorname{RSS}\_{GTSim}(t)\right) \\]

这与两个进程各自 Peak RSS 之和不是同一个指标。可使用较低频率 `/proc` 采样作为辅助指标，分别保留各进程的 Peak RSS 作为权威结果。

每组实验至少输出：

| Case  | Version           | Cycles | Python CPU(s) | C++ CPU(s) | Python Peak RSS(MiB) | C++ Peak RSS(MiB) | Wall(s) |
| ----- | ----------------- | ------ | ------------- | ---------- | -------------------- | ----------------- | ------- |
| T1–T5 | Old FIFO          | …      | …             | …          | …                    | …                 | …       |
| T1–T5 | Previous Socket   | …      | …             | …          | …                    | …                 | …       |
| T1–T5 | Refactored Socket | …      | …             | …          | …                    | …                 | …       |

其中 T1–T4 的 Cycle 不允许发生无依据的变化；T5 必须与固定基线一致。

## 七、代码量与性能验收标准

除了模拟正确性，本次必须把代码量减少作为交付条件。

建议明确以下目标：

| 指标                                 | 要求                                   |
| ---------------------------------- | ------------------------------------ |
| 生产代码总 LOC                          | 净减少，目标至少 800 行                       |
| 旧 FIFO / `.gtsir` 执行代码             | 完全删除                                 |
| C++ Program 执行器                    | 仅一套                                  |
| C++ DAG Materializer               | 仅一套                                  |
| Python Native Instruction Expander | 仅一套生成逻辑                              |
| Builder 的完整依赖副本                    | 0                                    |
| DAG dependency graph construction  | 每 Kernel 仅一次                         |
| GPU 初始化代码                          | 单一实现                                 |
| 同时驻留的完整 Kernel DAG                 | 最多 1 个                               |
| CPU 性能                             | 不应相较 Previous Socket 明显恶化；超过 5% 必须分析 |
| Peak RSS                           | 不应相较 Previous Socket 恶化；必须报告原因及热点    |

LOC 统计只包含生产 C++/Python 代码，排除文档、生成文件、测试数据和 Benchmark 输出。分别报告新增、删除、净变化，并且提供各个主要模块的变化。

如果新增的框架性代码抵消了大量删除收益，应优先合并类和函数，而不是继续引入抽象。

### 不建议使用的设计

本轮明确禁止为精简而再次引入：

- 新的 `ExecutionManager` / `RuntimeManager` / `SessionManager` 多级继承体系。
- 一套用于旧路径的 Adapter/Bridge/Facade。
- 通用插件式 Protocol Serializer。
- 与单 Stream 需求无关的 DAG Scheduling Framework。
- 需要复制整个 DAG 的校验或优化 Pass。

只在确实存在多个实现、且共享逻辑足够多时使用继承。 这里更适合组合、普通函数、委托构造、局部模板和 RAII，而不是复杂的类层次结构。

## 八、Agent 执行步骤

1. 冻结基线与实验配置

   建立基线 Worktree，保存 T1–T5 的运行命令、配置哈希、Cycles、CPU Time 和 Peak RSS。先确保可以重复运行，再开始破坏性修改。
2. 删除所有旧执行路径

   清理 CMake targets、旧 ProgramExecutor、TileIR 执行链、旧 Materializer/Codec、FIFO 路径及其专用测试。仅保留真正直接生成 DAG 的 C++ Workloads。
3. 统一 Simulator 生命周期

   使用委托构造函数移除重复设备初始化；重构 KernelExecutionSession，避免每个 Kernel 重复分配 Scheduler 和复制状态。
4. 精简 Builder

   删除 `deps_`、`children_`、拓扑排序和重复 Graph Construction。只保留一个 DAG 所有权及必要元数据。修复生命周期和异常清理。
5. 精简 Python Frontend

   去除完整 ExecIR 的旧路径，收敛 `NativeExpander` 和 `ProgramPlan`；修复 Prefetch 索引错误。
6. 优化 IPC

   改为有界 `InstructionBatch` 传输。确保批次内存固定有界，降低 JSON 编解码和系统调用开销。
7. 完成测试和性能对比

   先运行较小的 T1–T4，再运行 T5。输出严格一致性结果、CPU/RSS 对比及 LOC 报告。需要时只修复当前两条入口相关的问题，不恢复旧兼容层。
8. Formatting、Commit、Push

   按以下交付流程执行。

## 九、Formatting、提交与推送

在最终代码正确、性能报告生成后：

1. 对修改过的 C++ 文件使用仓库 `.clang-format`，优先使用项目已有版本 `clang-format 23.1.3`。
2. 对修改过的 Python 文件使用 `ruff format`，优先使用项目已有版本 `ruff 0.14.4`。
3. 执行 `git diff --check`，清理未使用头文件、Import、死代码和失效的 CMake target。
4. 重新 Release 编译并运行最终验证，确保格式化没有改变功能。
5. 保存最终 Benchmark Markdown、JSON 和精简后的设计说明。
6. 自动 Commit，并推送到 `custom` remote。

建议提交信息：

```
git add -A
git commit -m "refactor!: consolidate streamed DAG execution and remove legacy paths"
```

推送时必须先验证：

```
git remote get-url custom
git branch --show-current
```

然后推送当前工作分支到 `custom` 的对应分支，例如：

```
git push custom HEAD
```

禁止误推到 `origin`，禁止强制推送，禁止覆盖其他人的提交。若 `custom` 不存在或推送失败，应保留本地提交并明确报告原因。

## 十、最终交付要求

Agent 最终只需报告以下内容，不要编写大量过程性 Markdown：

- 代码变化：新增/删除文件数、C++/Python LOC 净变化、删除的旧执行路径。
- 正确性：T1–T5 新旧 Cycle 对照、Prefetch 索引修复测试。
- 性能：每组实验各自 Python/C++ CPU Time 与 Peak RSS、整体 Wall Time。
- 架构结果：证明只剩两条 DAG 入口、一个共享执行核心、一次依赖图构建、一个活跃 Kernel DAG。
- Git 结果：Commit SHA、分支、`custom` remote 推送结果。
