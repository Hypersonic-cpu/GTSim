# GTSim 第一阶段重构：Kernel 粒度的流式 Execution IR 与连续 DAG 执行

> **用途**：供 Coding Agent 直接实现的设计规范。  
> **代码基线**：`Hypersonic-cpu/GTSim` 的 `sim-validation` 分支（2026-11-13）。  
> **本阶段范围**：`frontend/` 的 Execution IR 生成、Python↔C++ 传输、GTSim 的 DAG 生命周期与 Kernel/Stream 元数据。**不重写硬件执行模型。**

## 0. 决策摘要

采用 **Persistent GTSim + On-demand Kernel Generation + One Active Kernel DAG**：

1. Python 常驻并保存未展开的 `ProgIR/FlowIR/ArchIR`；**禁止提前展开整个 Program 的 ExecIR/DAG**。
2. C++ GTSim 进程、GPU、SM、L2、MemorySystem、HBM/HBF、全局 Tensor 分配和仿真时钟持续存在。
3. C++ **仅在前一个 Kernel DAG 已退休并释放之后**，向 Python 请求下一个 Kernel；Python 收到请求后才展开该 Kernel，并逐条传输 Instruction。
4. C++ 接收 Instruction 时直接构造该 Kernel 的 `DAGNode`；一旦收到 `KernelEnd`，验证并执行整个 Kernel DAG。**第一阶段不做 Kernel 内 Chunk 级执行/回收**。
5. Kernel 只提供 `launch_id / stream_id / tensor binding / resource lifetime / completion` 等元数据；DAG Scheduler 不需要理解 GEMM/FA3 等 Kernel 算法语义。
6. 同一 Stream 的 Kernel 顺序使用**通用 Completion Token（逻辑 Kernel Barrier）**表示；无需虚拟 GPU 指令，也无需在每两个 Kernel 之间插入 `StreamSynchronize`。
7. 保留旧 `.gtsir` 文件执行路径作为兼容和回归基准。

### 硬性内存目标

\[
M_{\text{peak}} = M_{\text{persistent device}} + M_{\text{compact IR}} + O(N_{\text{largest kernel}})
\]

这里的 `O(N_largest kernel)` 允许一定的常数倍开销，例如 Python 的 Kernel-local Event 索引、C++ 的 DAG 与校验表；但**不允许**：

- Python 在整个程序范围内累计已经展开的指令；
- 多个 Kernel DAG 同时驻留；
- 同一个 Kernel 同时保留完整的 `ExecIR.ops`、转换后的 `ops` 字典、整块 JSON 字符串与 C++ `Kernel.ops`、`DAG.nodes` 等多份完整表示；
- C++ 完成 Kernel A 之前，Python 就预先展开 Kernel B；
- 全局 Scoreboard 等数组随**历史累计**指令 ID 单调增长。

本阶段不承诺整个 Python+C++ 双进程的 RSS 恰好等于单份 DAG 的字节数；要求的是**有界、无额外完整拷贝、与 Program 总展开节点数无关**。

---

## 1. Kernel、Stream 与 DAG 的精确定义

### 1.1 Kernel 是元数据和生命周期边界

原有 GTSim 已根据 `DAGNode` 中的 `op/pipeline/warp/CTA/tensor/dependencies` 模拟执行，不需要理解“这是一段 FA3”。重构后保持此原则。

Kernel 元数据仅负责：

- `launch_id`：身份、统计和日志；
- `stream_id`：执行顺序所属 Stream；
- `tensor_bindings`：局部 Tensor ID 到 Program 全局 Tensor ID 的映射；
- Launch overhead，以及 CTA/warp 资源的建立和释放；
- 入口外部依赖（`wait_tokens`）和完成信号（`signal_token`）；
- 输出契约、必要的 PrefetchWindow 元数据。

**不要给每个 DAGNode 增加必须依赖 KernelContext 的调度逻辑。**

### 1.2 Kernel Barrier 是普通的 Completion Dependency

同一 Stream 的 `QKV → FA3 → O Projection` 等价于：

```text
QKV: 所有终点指令完成 ─> Token(QKV.done) ─> FA3: 所有根指令允许进入 Ready
FA3: 所有终点指令完成 ─> Token(FA3.done) ─> OProj: 所有根指令允许进入 Ready
```

注意：

1. **不能仅依赖前一个 Kernel 的最后一条“列出的”指令**：其他 Warp/CTA 可能更晚完成。
2. **不能只阻塞下一个 Kernel 的第一条指令**：不同 Warp 的其他根节点可能抢先执行。
3. Barrier 是**零 Cycle 的控制状态**，不占用 SM、Warp、Pipeline，也不访问内存。
4. 物理内存中不需要同时存在两个 Kernel 的 DAG。A 完成、释放后，仅保留 `QKV.done = COMPLETED`，然后生成 B。入口 Token 可理解为 Barrier→B 所有根节点的**压缩表示**。

**同 Stream 的自动顺序 ≠ 显式 `cudaStreamSynchronize()`**。显式 Sync 是 Program Command；不应在每个 Kernel 边界自动生成。

### 1.3 Kernel 的完成条件

需要同时满足：

- 已收到 `KernelEnd`，没有尚未生成的指令；
- **全部** DAGNode 已完成，而非某个单独的尾节点；
- CTA 与相关流水线/异步操作已退休，后继 Kernel 所需的写入可见；
- Scheduler、Pipeline、SM、TMA/TMEM、Memory 回调均不再引用待销毁的节点。

Kernel 完成**不意味着**清空 L2、重建 HBM/HBF、归零时钟、释放全局 Tensor，也不意味着等待无关后台预取。第一阶段可以暂时保留当前 `Simulator::run()` / `KernelExecutionSession::retire()` 中较保守的 `foreground_idle()` 检查，保证结果兼容；**不要额外在每次 Kernel 后调用整个 `MemorySystem::idle()`**。

### 1.4 第一阶段只支持一个有序 Stream

当前 C++ `ProgramExecutor` 限制一个 Stream，前端 `prog_to_flow.py` 也按单有序 Stream 编译。本阶段保持此约束；所有元数据仍保留 `stream_id`，便于以后扩展。遇到多 Stream 并发或无法实现的 Cross-Stream Wait，必须显式报错，**不能悄悄用全局串行执行替代**。

---

## 2. 当前代码中的关键耦合（必须处理）

| 文件 | 现状 | 第一阶段改动 |
|---|---|---|
| `frontend/src/compile/__main__.py` | 对整个 ArchIR 调用 `lower_arch()` | 新增流式执行入口，不构造全 Program ExecIR |
| `frontend/src/compile/prog_to_flow.py` | 将多个 Invocation 合并为 Region；`SYNCHRONIZE` 被消费而未写进 FlowIR | 单独保留原始 ProgIR 命令，生成紧凑 `ProgramPlan` |
| `frontend/src/compile/arch_to_exec.py` | `lower_arch()` 遍历全部 Region，构造完整 `ExecIR.ops`；还用上一个 Region 的尾指令 ID 连接下一个 Region | 仅为被请求的 Region 生成指令；跨 Kernel 依赖改成 Token |
| `frontend/src/compile/expand_native.py` | `NativeExpander` 不断追加 `self.nodes`，`Event` 索引随展开增长 | 增加逐条输出接口；Event 索引只存在当前 Kernel |
| `frontend/src/compile/export_runtime.py` | 先创建每个 Kernel 的 `ops` 字典列表，再 `json.dump()` 整块输出 | 新路径逐条序列化到 IPC，不构造完整 JSON |
| `src/execution_ir/graph_codec.cpp` | 读完整文件字符串，解析完整 `Kernel.ops` | 新路径逐条解析有界协议消息 |
| `src/execution_ir/graph_materialize.cpp` | 从完整 `Kernel.ops` 创建 DAG，含所有校验 | 提取可复用的 Validator，并实现增量**构图**的 Builder |
| `include/dag_node.h` | DAG 拥有节点，同时带有 `tensor_registry` | 新路径 Tensor Registry 改为 Program 生命周期；DAG 仅持有当前 Kernel 的节点 |
| `include/simulator.h`, `src/simulator.cpp` | 构造函数依赖 `DAG*`，GPU 根据 `dag->tensor_registry` 初始化 | 增加与 DAG 无关的 Simulator 构造和 Graph attach/detach |
| `src/runtime/kernel_execution_session.cpp` | `initialize/rebind` 强绑定全 DAG 与调度状态 | 拆分 graph attach、完成验证、detach 清理 |
| `src/runtime/program_executor.cpp` | `read_kernel → materialize → bind_kernel → run`；`registry_owner` 特别维持首个 DAG 的生命周期 | 新增持久 IPC Runner；取消新路径中的 `registry_owner` workaround |
| `include/scheduler.h` / `src/scheduler.cpp`、`include/sm.h` / `src/sm.cpp` | 保存大量 `DAGNode*`、warp/CTA 关联 | 在释放 DAG 之前彻底清理这些指针与 Kernel-local 状态 |

**特别注意：**`arch_to_exec.py` 中 `tails[region]` + `predecessors` 的跨 Region 依赖不能原样进入新协议，否则新 Kernel 会引用已经删除的前一 Kernel 的节点 ID。Kernel 内 `after_issue / after_complete` 则必须完全保留。

---

## 3. Python 端 Class / Interface

建议增加目录 `frontend/src/compile/streaming/`，不要为了本次任务重写 Parser、静态求值器、Hopper Lowering 或高层 IR Grammar。

### 3.1 `ProgramPlan`：保留 Program 命令语义

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class LaunchPlan:
    launch_id: str
    stream_id: str
    arch_region_index: int
    tensor_bindings: tuple[tuple[int, int], ...]  # local tensor -> global tensor
    wait_tokens: tuple[str, ...]
    signal_token: str

@dataclass(frozen=True)
class ProgramPlan:
    stream_id: str
    tensors: tuple                 # 只含全局 Tensor 描述
    commands: tuple                # 有序命令，不含展开的 Instruction
    launches: tuple[LaunchPlan, ...]
```

**来源规则：**

- 输入是 `ProgIR`：在 `compile_program()` 合并 Region 之前，从**原始** `ProgIR.commands` 提取命令；同时继续构造对应 FlowIR/ArchIR。不要指望从 ArchIR 还原已经丢弃的 Sync/Event 信息。
- 输入是 `FlowIR / ArchIR`：根据 `regions` 和单 Stream 元数据合成按序 Launch Plan。
- 保留既有 `Prefetch / QueuePrefetch / Release / RecordEvent / WaitEvent / StreamSynchronize` 语义；不能支持的命令明确拒绝。尤其不能默默删除显式同步。
- `ProgramPlan` 引用 Region 索引和命令，不含 DAGNode、展开的 `ops` 或多份 AST。

### 3.2 `ExecutionIRProducer`：按请求展开

```python
class ExecutionIRProducer:
    def __init__(self, arch_ir, plan: ProgramPlan, options):
        self.arch_ir = arch_ir          # 紧凑 ArchIR
        self.plan = plan
        self.options = options
        self._active_launch = None

    def emit_program_manifest(self, writer) -> None:
        """发送 Program、Tensor、Commands；不包含展开指令。"""
        ...

    def emit_kernel(self, launch_id: str, writer) -> None:
        """仅展开被请求的一个 Kernel；按记录发送并释放其临时状态。"""
        if self._active_launch is not None:
            raise RuntimeError("concurrent kernel expansion is forbidden")
        self._active_launch = launch_id
        try:
            # 查找一个 ArchIR Region
            # 发送 KernelBegin、CTA、Instruction、PrefetchWindow、KernelEnd
            ...
        finally:
            # 不保留本 Kernel 的展开节点与 EventIndex
            self._active_launch = None
```

### 3.3 改造 `NativeExpander` 的具体要求

现在 `NativeExpander.expand()` 将所有 `ArchInstruction` 追加到 `self.nodes`。新路径实现类似：

```python
expander.expand_to(
    emit_cta=lambda cta: writer.send_cta(cta),
    emit_instruction=lambda op: writer.send_instruction(op),
)
```

实现要点：

1. 维护 **Kernel-local** 的自增 `next_node_id`，不依赖整个 Program 的 `len(nodes)`。
2. 依赖查找继续使用 `EventIndex`，但保存**必要的节点 ID、绑定值和 producer 信息**，不持有完整的过去指令对象。Kernel 结束后释放整个索引。
3. 每生成一条指令立即传输；**不要**再同时生成 `ExecIR.ops`、导出 dict 列表和完整 JSON 字符串。
4. 当前 `lower_region()` 可用于理解单 Region 的现有语义，但新路径不要以返回完整 `ExecIR` 再整体导出作为最终实现。
5. 统一保持 `after_issue` / `after_complete` 及 WGMMA/TMA 依赖的现有语义。
6. `output coverage`、`partial_outputs`、PrefetchWindow 的消费节点等 Kernel 级契约可保留小型校验/索引结构，不必保留完整指令列表。

**协议限制：**每个 Kernel 内 Instruction ID 从 0 开始连续编号；依赖只能引用已生成的**同一 Kernel**的较小 ID。当前 `lower_arch()` 本就检查 forward dependency，故流式路径应当显式验证和沿用此约束，而不是静默忽略前向边。

---

## 4. Python ↔ C++ IPC 协议

### 4.1 通信实现

推荐 Linux/Unix 的全双工 `socketpair(AF_UNIX, SOCK_STREAM)`。Python 通过 `subprocess.Popen(pass_fds=...)` 启动 C++ GTSim 子进程，使用专门 FD 收发协议，日志走 stderr。**两个进程均持续运行，不为每个 Kernel 重新起进程。**

建议使用简单的版本化 Length-Prefixed JSON：

```text
[uint32 little-endian payload_length] [UTF-8 JSON message]
```

每条 Instruction 是一帧；禁止把整个 Kernel 的指令放在一个 JSON 数组里。为单帧设上限（例如默认 1 MiB，可配置），检查部分读写、异常 EOF、版本不匹配、消息顺序和字段错误。传输层应独立，以便后续换 Protobuf/二进制时不动 GTSim。

### 4.2 消息列表

| 方向 | 消息 | 主要字段与含义 |
|---|---|---|
| Python → C++ | `ProgramBegin` | 协议版本、Stream ID、Program 标识 |
| Python → C++ | `TensorDef` | 全局 Tensor shape、dtype、tier、stride、layout、placement 等 |
| Python → C++ | `CommandDef` | 程序内有序 `LaunchKernel/Prefetch/Release/Event/Sync` 等命令；不含展开 DAG |
| Python → C++ | `ProgramReady` | 全部紧凑 Program 元数据发送结束 |
| C++ → Python | `RequestKernel` | 需要的 `launch_id`、`stream_id`；仅在旧 DAG 销毁后发送 |
| Python → C++ | `KernelBegin` | Kernel Tag、Stream、`wait_tokens`、`signal_token`、Tensor Bindings |
| Python → C++ | `CTA` | 一个 CTA 的 SM/warp/shared/register 资源 |
| Python → C++ | `Instruction` | 一条指令的 opcode、pipeline、ID、CTA、warp、tile、tensor、依赖等 |
| Python → C++ | `PrefetchWindow` | 可选的一个窗口；消费者使用当前 Kernel 的局部节点 ID |
| Python → C++ | `KernelEnd` | `op_count`、`cta_count`、输出与 partial output 契约 |
| C++ → Python | `KernelDone` | 完成信号、Kernel Tag、可选摘要统计 |
| Python → C++ | `ProgramEnd` | 全部 Program 命令执行完毕 |
| 双向 | `Error` | 结构化错误码和描述，触发安全清理 |

完整单次 Launch 的时序：

```text
Python                          Persistent C++ GTSim
   | ProgramBegin / TensorDef / CommandDef |
   |------------------------------->       |
   | ProgramReady                   |       |
   |------------------------------->       | 创建全局 Tensor Registry / GPU
   |                                       |
   |           RequestKernel("QKV")        |
   |<-------------------------------       |
   | KernelBegin / CTA / Instruction*      |
   | PrefetchWindow* / KernelEnd           |
   |------------------------------->       | 构造、校验并执行 DAG(QKV)
   |                                       | retire → detach → destroy DAG
   |                KernelDone("QKV")      |
   |<-------------------------------       |
   |           RequestKernel("FA3")        |
   |<-------------------------------       | 再展开下一个 Kernel
   | ...                                   |
```

**Flow Control：**第一阶段固定 `max_inflight_kernels = 1`，由 C++ `RequestKernel` 提供反压。Python 禁止 speculative lookahead。C++ 接收完全部当前 Kernel 记录并执行完毕，才请求下一个。Python 生成耗费的 CPU Wall Time **不计入 GPU 仿真周期**（除非未来专门建模 Host Launch Delay）。

---

## 5. C++ 端 Class / Interface 设计

以下名称可按当前代码风格调整，但职责、所有权与调用顺序必须一致。

### 5.1 `FramedTransport`

建议：`include/runtime/stream_protocol.h`、`src/runtime/stream_protocol.cpp`。

```cpp
class FramedTransport {
public:
    explicit FramedTransport(int socket_fd);
    Message receive();                 // 逐帧读取并校验
    void send(const Message& message); // 一次发送一条有界记录
};
```

`Message` 使用强类型的 variant/record decoder，避免大量无类型动态 Map 进入 Simulator 热路径。协议错误/中断时关闭当前 GraphBuilder 并回收节点，不留悬挂指针。

### 5.2 `DependencyTokens`：通用 DAG Bridge

建议：`include/runtime/dependency_tokens.h`。

```cpp
using TokenId = std::string;

enum class TokenState : uint8_t { Pending, Completed };

class DependencyTokens {
public:
    void declare(TokenId id);
    bool all_completed(std::span<const TokenId> ids) const;
    void complete(const TokenId& id);
    void release_if_unreferenced(const TokenId& id);
};
```

行为：

- 当前 Kernel 在**全部完成、退休且完成必要设备操作后**标记其 `signal_token = Completed`。
- 下一个 Kernel 的所有根节点的逻辑前驱是 `wait_tokens`。在 `attach_graph()` 之前检查 `all_completed()`；未满足时不能调度任何根节点。
- 单 Stream 下通常只需要一个前驱 Token；保留通用 Token 机制支持将来扩展。
- Token 无 Pipeline/SM/Latency，不要把它做成一个必须分配 warp 的虚假 `DAGNode`。
- 已无消费者的 Token 应释放，不能永久缓存全部历史 Launch。

### 5.3 `KernelGraphBuilder`：边接收边构图，但仅在完整 Kernel 后执行

建议：`include/execution_ir/kernel_graph_builder.h`、`src/execution_ir/kernel_graph_builder.cpp`。

```cpp
struct KernelHeader {
    std::string launch_id;
    std::string stream_id;
    std::vector<TokenId> wait_tokens;
    TokenId signal_token;
    std::vector<std::pair<int, int>> tensor_bindings;
};

struct BuiltKernel {
    KernelHeader header;
    std::unique_ptr<DAG> dag;
    std::vector<ExecutionIR::PrefetchWindow> prefetch_windows;
};

class KernelGraphBuilder {
public:
    KernelGraphBuilder(const TensorRegistry& global_tensors,
                       const SimulatorConfig& config);
    void begin(KernelHeader header);
    void add_cta(const ExecutionIR::CTA& cta);
    void add_instruction(const ExecutionIR::Instruction& op);
    void add_prefetch_window(ExecutionIR::PrefetchWindow window);
    BuiltKernel finish(const KernelEnd& end);
    void abort() noexcept;
};
```

**核心实现细节：**

1. `add_instruction()` 完成字段校验和 Tensor ID 映射后，**立刻创建一个 `DAGNode`**。不要先积累 `std::vector<Instruction>` 再创建节点。
2. 保留现有 `DAG::nodes / nodes_dict / nodes_by_name`。这仍是单 Kernel 的完整 DAG，**不是 Phase 2 的运行中任意节点插入**。
3. `finish()` 校验 `op_count/cta_count`、Dependency ID、CTA 资源、目标 SM、Opcode↔Pipeline、Tensor 区域、输出覆盖/重叠、WGMMA collective 完整性和有向无环性，然后构建 `children/issue_children`。尽量复用 `graph_materialize.cpp` 的实际校验逻辑，不要做出行为不一致的第二套 Validator。
4. Kernel 内 ID 从零开始，每个新 Kernel 可重用 ID。所有 `node_id_to_sp / global_scoreboard` 等容器都只根据**当前** Kernel 分配。
5. `PrefetchWindow` 随 Kernel 构造，仅保留当前 Kernel 所需的窗口；保持原有 `KernelPrefetchSession` 行为。
6. `finish()` 返回后 Builder 不继续保存另一份完整 Kernel 信息；`abort()` 与异常路径必须能释放部分生成的节点。

### 5.4 `Simulator`：硬件持久，Graph 可 Attach/Detach

对现有 Simulator 增加接口，**保留旧构造函数**供传统 DAG/文件执行路径使用：

```cpp
class Simulator {
public:
    // 新增：Program 级构造，不依赖首个 DAG
    Simulator(const SimulatorConfig& cfg, const TensorRegistry& global_tensors);

    void attach_graph(DAG& graph);
    bool run_active_graph(int max_cycles, bool verbose = false);
    void verify_graph_retired() const;
    void detach_graph();
};
```

需要拆分的对象生命周期：

| Program 级常驻 | Kernel-local，必须释放 |
|---|---|
| `GPU / SM` 硬件对象 | 当前 `DAG / DAGNode` |
| `L2` 内容及替换状态 | Scheduler 内 `scheduler_warp/ready_warp/node_dict` |
| `MemorySystem`、HBM/HBF 后端 | `global_scoreboard`、`node_id_to_sp/sm` 的当前节点映射 |
| 全局 `TensorRegistry` / Global Address | SM 的 `tb_to_nodes`、CTA residency 及相关预约状态 |
| 全局模拟 Cycle 和设备统计 | 当前 Kernel 的回调、PrefetchWindow Session、Kernel-local 临时状态 |
| 可跨 Kernel 持续的预取/后台队列 | 当前 Kernel 指令和结束后无用的依赖表 |

**Graph 释放顺序必须是：**

```text
run graph
  → 全部指令/必要设备操作完成
  → verify_graph_retired()
  → 清空 node_completion_observer 及其捕获的 Session
  → detach_graph()：清理 Scheduler/SM/Pipeline 等全部 DAGNode* 引用
  → DAG.reset()：真正释放节点
  → signal_token.complete()
  → RequestKernel(next)
```

当前 `KernelExecutionSession::initialize(..., replace_schedulers=true)` 会重新创建部分 Scheduler，这是允许的：**Kernel-local 调度器可以重置，GPU/L2/MemorySystem 不可以重建**。不要为了删除 `bind_kernel()` 强行让已完成 Kernel 的节点常驻。

当前 `Simulator::run()` 使用 `completed_node_count == dag->nodes.size()` 判断结束；此阶段仍可用于**当前完整 Kernel**，但不能误认为这是一个长生命周期 Program DAG 的完成状态。第二阶段再引入 `input_closed` 与动态节点计数。

### 5.5 `StreamingProgramRunner`：唯一的 DAG 生命周期控制者

建议保留 `ProgramExecutor` 的非流式功能，新增 `StreamingProgramRunner`：

```cpp
class StreamingProgramRunner {
public:
    StreamingProgramRunner(SimulatorConfig cfg, FramedTransport& transport);
    ProgramTiming run();

private:
    void receive_program_manifest();
    void execute_command(const ProgramCommand& command);
    void execute_launch(const LaunchPlan& launch);
    void retire_and_release_active_graph();

    TensorRegistry global_tensors_;         // Program 生命周期
    std::unique_ptr<Simulator> simulator_;    // 只创建一次
    std::unique_ptr<DAG> active_graph_;       // 同时最多一个
    DependencyTokens tokens_;
    FramedTransport& transport_;
};
```

`execute_launch()` 必须按这个顺序：

1. 断言 `active_graph_ == nullptr`；旧 Kernel 的所有 Node 指针已被清理。
2. 发送 `RequestKernel(launch_id)`。
3. 逐条接收 `KernelBegin → CTA / Instruction / PrefetchWindow → KernelEnd`，使用唯一 `KernelGraphBuilder` 直接生成 DAG。
4. 验证 Launch/Stream ID、Tensor Binding、输出契约与入口 Token。
5. 在**同一个 Simulator** 中 `attach_graph`，根据原有模型处理 Launch overhead、Prefetch Session，然后运行该完整 DAG。
6. 完成后统计 per-kernel timing；检查必要设备操作完成。
7. 清回调、`detach_graph`、销毁 `active_graph_`；仅保留 Program 级状态与紧凑统计。
8. 标记 `signal_token`，发送 `KernelDone`；继续执行 Command 或请求下一个 Kernel。

**一定要拆掉新路径中的 `registry_owner` 生命周期 workaround。** 当前首次 DAG 中的 `tensor_registry` 被用于 GPU 初始化，新构造函数必须改用 Program 级 Tensor Registry，避免第一个 DAG 无法释放或在每个 DAG 中复制完整注册表。

### 5.6 Sync、Event、Prefetch 的处理

- **同 Stream Kernel 顺序**：Completion Token 自动保证；不自动生成 `StreamSynchronize`。
- **显式 `StreamSynchronize`**：只有原始 Program 明确出现时才执行；不能由前端丢失。此处表现为 Host 等待语义/显式命令，在单 Stream 的 GPU 执行序列中可能不增加额外调度约束。
- **`RecordEvent / WaitEvent`**：保留已有单 Stream 语义；不支持的跨 Stream 版本明确报错。
- **`Prefetch / QueuePrefetch / Release`**：原有 HBF SRAM 容量、Read Alias、Epoch、延迟和后台请求语义必须保持，不得因 Kernel 结束就取消、丢弃或强制排空无关 Prefetch。
- **`ProgramEnd`**：可以执行已有 Program 结束时的最终同步/统计逻辑；不能误把每个 Kernel End 都当作 Program End。

---

## 6. 推荐实施顺序

### Step A — 固化基准行为

- 阅读并记录现有 `export_runtime.py → graph_codec.cpp → graph_materialize.cpp → program_executor.cpp → KernelExecutionSession → Simulator::run()` 路径。
- 为至少两个 Kernel、两个 CTA 的测试建立旧路径基准，尤其保证**最后列出的指令不一定最后完成**。
- 保存 GEMM、FA3、HBF Prefetch 的周期、访问量、输出覆盖与资源占用基准。

### Step B — 拆出 Program Tensor 与 Persistent Simulator

- 新增不依赖 `DAG*` 的构造函数；GPU 仅构造一次。
- 实现 `attach_graph / verify_graph_retired / detach_graph`。
- 清理 Scheduler、SM、Pipeline 的所有当前 DAG 指针；确保前一 DAG 删除后不会 UAF。
- 支持后一个 Kernel 重新使用 `node_id = 0`，同时保留 L2 和 HBF/HBM 状态。

### Step C — Completion Token 和紧凑 ProgramPlan

- 保留原 `ProgIR.commands`，尤其显式 `SYNCHRONIZE` / Events。
- 将跨 Region 尾指令依赖从新流式路径剥离，改成 `wait_tokens/signal_token`。
- 单 Stream 下验证 QKV→FA3→OProj 的顺序，不生成伪 GPU 指令或隐式 Stream Sync。

### Step D — C++ 接收流并直接构造 DAG

- 实现 `FramedTransport`、消息校验和 `KernelGraphBuilder`。
- 一条 Instruction 对应一次 `add_instruction`；不要在 C++ 新路径产生完整的 `Kernel.ops` 备份。
- Builder 只在 `KernelEnd` 后提交执行，避免触及 Phase 2 的动态 Scheduler 改造。

### Step E — Python 按需生成

- 建立 `ExecutionIRProducer` 和 `NativeExpander.expand_to()`。
- 任何时刻只展开 C++ 当前请求的 Kernel；逐条发送，不落地庞大的 `.gtsir` 文件。
- Kernel 结束后释放生成器、EventIndex、临时绑定与输出校验数据。
- 保留旧 `lower_arch()` / `export_runtime()` 供校验，但新路径不得调用整个 Program 的 `lower_arch()`。

### Step F — 测试与收尾

- 测试全部在HBM(simple model)上做, 不需要涉及ramulator2和HBF. 
- 可以先利用 two gemm 的 .progir 运行两个小型 GEMM 看看是否正确.
- 运行 Qwen 30B-A3B (复用之前的 .progir)（QKV Projection→FA3→O Projection，或等价可用 Fixture）。
- 对比旧路径：Instruction 完成顺序、Kernel 周期、Memory Bytes/Requests、Prefetch、Tensor 地址、L2 内容与累计统计。
- 连续运行大量小 Kernel 与单个大型 Kernel；代码结构上保证不会留下历史 DAG、完整 JSON、Program-wide EventIndex 或不断增长的 Node-ID 数组。**此阶段不要求强制进行 RSS 数值基准测试**。
- 验证坏数据、半包、IPC 中断、非法依赖、CTA 错误与不支持的 Stream 命令均安全报错、清理资源。

---

## 7. 验收清单（不可妥协）

### 功能

- [ ] 新路径 Python 不创建整个 Program 的展开 `ExecIR`。
- [ ] C++ 只有一个 Persistent GPU/SM/L2/MemorySystem 与连续全局仿真时钟。
- [ ] 上一 Kernel DAG **释放之后**才发送下一 `RequestKernel`。
- [ ] 同 Stream 的执行顺序由所有前驱指令完成后触发的 Completion Token 保证。
- [ ] 没有隐式插入 `StreamSynchronize`、`DeviceSynchronize` 或 L2 Flush。
- [ ] 原有指令依赖、CTA 映射、Prefetch、Tensor 可见性与必要设备操作都正确。
- [ ] `ProgramTiming` 和既有 Benchmark 的结果与旧路径一致，差异需有明确原因。
- [ ] 旧 `.gtsir` 路径及现有测试仍然可用。

### 内存和所有权

- [ ] 不产生 Program 级全部展开 DAG，也不产生巨大的 JSON 数组。
- [ ] 新 C++ 路径没有同时存在完整 `Kernel.ops` 与完整 `DAG.nodes`。
- [ ] 新 Python 路径不同时保留完整 `ExecIR.ops`、`ops` dict 数组和 JSON 文本。
- [ ] 不会同时保存两个已展开 Kernel DAG。
- [ ] DAG 删除前已经解除所有 Scheduler/SM/Pipeline/Memory 的悬挂指针。
- [ ] Tensor Registry、GPU 和 Cache 不依赖第一份 DAG 存活。
- [ ] Scoreboard、Node Map 和 Event 索引不随**累计**已执行的指令数无限增长。

### 严格不做

- [ ] 不做多 Stream 并发调度（后续阶段）。
- [ ] 不重写 Hardware Scheduler、HBF Backend、Compiler Grammar。

---

## 8. 给 Agent 的实现提醒

1. **“按 Kernel 生成”不等于“先生成所有 Kernel 文件，然后逐个读取”。** Python 展开也必须是 on-demand。
2. **“单 Kernel DAG”不代表 Kernel 是 Scheduler 的基本语义。** 它只是第一阶段的物理内存窗口；执行层仍处理普通 DAGNode。
3. **“Kernel Barrier”不要实现成一个需要 Warp 资源的假指令。** Completion Token 足以表达当前单 Stream 的 DAG 桥接。
4. **不能仅等待某一条尾指令，也不能仅约束下一 Kernel 的某一条根指令。**
5. **Kernel 回收时删除的是图和调度引用，不是 MemorySystem。** 必须保留 Cache Residency、全局 Tensor 地址和合法的后台 Prefetch。
6. **`prog_to_flow.py` 当前不会在 FlowIR 中保留显式 Sync。** 必须用原 ProgIR 创建 sidecar ProgramPlan；不允许丢同步语义。
7. **`Simulator` 当前从 `dag->tensor_registry` 构造 GPU。** 必须解除这个所有权耦合，否则无法正确释放第一个 DAG。
8. **`graph_materialize.cpp` 包含许多实际安全/一致性校验。** 新 Builder 必须共享或完整继承这些校验；不能因为流式处理而降低正确性。
9. **本阶段只要求一个 Kernel 完整接收后再开始执行。** 不要偷偷实现节点级增量运行，避免把第二阶段的复杂度提前带进来。

**Definition of Done：**Python 根据 C++ 的请求逐个生成 Kernel 的 Execution IR；C++ 单个进程中保持 GPU/Memory 常驻，逐个构造、运行和释放完整 Kernel DAG，借助通用 Completion Token 实现同 Stream 的跨 Kernel 依赖。内存规模与最大单 Kernel 相关，而非 Program 的累计展开规模，并保持原 GTSim 的执行语义与回归测试。

