# GTSim Memory Backend Refactor & HBF Simulator Integration

## 目标与约束

基于 `sim-validation` 分支，使用现有 submodule 固定版本，不修改第三方源码、不升级 submodule。

- 新增 **HBF-EventLab**、**FlashLab HBFSim** 两个可选择的 HBF 后端。
- 保留现有 Naive、HBFine、Ramulator2 的能力和行为。
- 统一 Backend 抽象，重构 `include/`、`src/` 的 Memory 相关文件组织。
- 仅使用原生 C++ GEMM；不构建或运行 TVM、自动调优、Serving 等额外任务。
- GEMM 固定 `A=HBM`、`C=HBM`，仅允许 `B=HBM/HBF`。
- 执行四组实验：FullHBM 5H、CoLocated 4F1H × 三种 HBF 后端。
- 结果保存到 `validation/results/1112-hbf-backend-gemm/`。

## Step 1 — 重构 Memory 模块

建议组织：

```text
include/memory/
  base.h          # MemRequest/Response、MemoryBackend 接口
  config.h        # Backend 类型、配置、校验
  stats.h         # 公共统计结构
  clock.h         # GPU cycle 与物理时间转换
  system.h        # MemorySystem 对外接口
  address.h       # 公共地址转换约定

src/memory/
  base.cpp        # Backend factory、公共辅助逻辑
  naive.cpp       # 原有 HBMModel / HBFModel
  hbfine.cpp      # HBFine adapter
  ramulator2.cpp  # Ramulator2 adapter
  eventlab.cpp    # HBF-EventLab adapter
  flashlab.cpp    # FlashLab HBFSim adapter
  system.cpp      # MemorySystem 相关实现
```

要求：

1. 迁移现有 `memory_backend.cpp`、`hbfine_backend.cpp`、`ramulator_backend.cpp`，并将 `memory.h` 中属于 Backend/MemorySystem 的实现适当拆分。
2. 保留**一个公共 `MemoryBackend` 抽象基类**。HBM/HBF 均通过同一请求接口访问；必要时增加轻量的 HBF 专用能力接口，不建立不必要的多层继承。
3. 公共接口统一使用 `MemRequest`、`MemResponse`、GPU cycle、请求 ID 和通用统计结构。第三方类型仅出现在对应 Adapter 的 `.cpp` 或私有实现中。
4. 现有预取扩展属于可选能力，使用显式 capability/unsupported 语义；不得让 EventLab/FlashLab 静默假装支持 HBFine SRAM Prefetch。
5. 移除通用 Backend 接口对 `hbfine::Statistics` 的直接依赖，提供可比较的公共指标；保留必要的 HBFine 专属诊断。
6. 原 `include/memory.h` 等路径可暂时保留轻量兼容头，避免修改无关模块；不要保留两套独立实现。
7. 使用统一 Factory 按配置创建 Backend。使用 RAII、`std::unique_ptr`、单一状态所有权，不使用全局单例。

**重构过程中不得修改 GPU pipeline、L2/MSHR、DAG 调度或现有 D2D 路由语义，除非新后端接入确实要求。**

## Step 2 — 统一 Backend 契约

沿用现有：

```cpp
class MemoryBackend {
public:
    virtual ~MemoryBackend() = default;

    virtual bool try_submit(const MemRequest&, int cycle) = 0;
    virtual std::vector<MemResponse> step(int cycle) = 0;
    virtual std::size_t pending_requests() const = 0;
};
```

保留原有统计、预取、初始化扩展，并完善以下语义：

- 拒绝提交时，请求所有权仍属于 GTSim。
- 每个成功接收的请求恰好返回一次完成。
- Backend 内部可以拆分请求，但必须正确汇总 Parent completion。
- 完成时间不得早于请求到达时间。
- 正确处理背压、多个 outstanding requests 和非整页请求。
- 统一 clock conversion；禁止逐周期累积浮点误差。
- `preload_region()` 必须确实建立可读取状态；不支持时明确报错，不允许空实现冒充成功。
- 日志和统计不应保存无限增长的逐请求历史。

GPU L2、HBM/HBF 路由和 GTSim D2D link 保持原来的外部接口。明确每个物理链路资源由哪一层计时，**不得在 GTSim D2D 与 HBF 模型内部重复计算同一条 UCIe/HBIO 链路的延迟和带宽**。

## Step 3 — 接入 HBF-EventLab

使用 `HBFEventLab::core` 和 `SimulationSession`，不使用 CLI、TensorAccessBridge 或 Qwen 实验。

- 采用 `try_submit()`、`advance_before(T)`、completion handler 建立在线请求路径。
- 将 GTSim 请求映射到 EventLab 的 Global 地址、64 B 命令和读写状态；支持父请求拆分及完成合并。
- 使用同一个 GPU 时间轴推进，严格遵守 `advance_before(T)` 的排他终点与同 Tick 因果约束。
- 配置四个 HBF stack 对应的 channel、bank、page-buffer 等资源；若使用独立 Session 表示各 stack，确保地址交织及时间推进一致。
- GEMM 的 B 为预加载只读权重。EventLab 若缺乏公开的直接状态初始化 API，可在 setup 阶段通过合法写路径建立初始内容，完全排空后开始测量；setup 时间和流量必须单独记录，不纳入 GEMM。
- 关闭故障注入及不必要的逐事件统计，保留必要的 NAND/请求计数。

不得把未初始化 NAND 的读取错误屏蔽成成功。

## Step 4 — 接入 FlashLab HBFSim

优先直接链接第三方原生 C++ `hbfsim_core`，避免 Python、标准输入文本协议及子进程 IPC。

- 使用 `hbfsim::physical::SimulationSession` 及其 `SimulationTransaction`。
- 选择适合只读权重的 `HbfStatic` / 静态初始镜像路径，避免在 GEMM 测量阶段进行权重初始化和 GC。
- GPU 的 HBF 地址统一转换为 FlashLab 的设备局部地址；配置四个 HBF stacks 和正确的物理资源。
- 将返回的 completion timestamp 转换成 GTSim cycle，并按时刻交付；不能在 `run_batch()` 返回时提前标记完成。
- **重点验证闭环时间语义**：FlashLab 的 batch-relative `issue_ns`、frontier 和未来资源预约，必须能够保持 GTSim 动态请求的真实到达顺序及并发竞争。优先尝试非阻塞 frontier，并验证后续请求不会破坏已经返回的完成时间。
- 若其现有批次 API 无法正确表达动态到达与资源竞争，不允许采用“每批等待全部完成”的近似来掩盖问题；应明确定位并实现最小的原生集成适配，或将该问题标记为阻塞项。

**构建注意**：FlashLab 的 CMake 使用 `CMAKE_SOURCE_DIR` 管理部分路径，优先采用隔离的 CMake 子构建并导入 `hbfsim_core`，避免直接 `add_subdirectory()` 导致路径解析错误。不要构建第三方的实验、报告和测试 targets。

## Step 5 — 配置与 GEMM Placement

扩展原有 `HBFBackend` 配置选择：

```yaml
hbf:
  backend: bank_plane  # 需要改名为 HBFine 相关
  # backend: eventlab
  # backend: flashlab
```

名称需清晰，原有 `bank_plane` 需要和新的保持“平行”。

统一四组配置：

| Case     | Topology  | HBM | HBF | HBF Backend | GEMM Placement     |
| -------- | --------- | --: | --: | ----------- | ------------------ |
| Baseline | FullHBM   |   5 |   0 | 无           | A/B/C = HBM        |
| HBFine   | CoLocated |   1 |   4 | HBFine      | A/C = HBM, B = HBF |
| EventLab | CoLocated |   1 |   4 | EventLab    | A/C = HBM, B = HBF |
| FlashLab | CoLocated |   1 |   4 | FlashLab    | A/C = HBM, B = HBF |

要求：

1. 全部使用相同 H100 GPU 配置和 GEMM 参数。
2. HBM 后端保持相同实现，只按 stack 数量调整容量/带宽。
3. HBF 三组尽可能统一 4 μs read、75 μs program、4 KiB NAND page、stack/channel 数量和物理接口档位；模型无法等价的参数明确记录。
4. 使用已有 `placement: all_hbm` / `gemm_b_hbf`，保证 A/C 始终 HBM。Split-K 中的 PartialC 也必须保留在 HBM。
5. 当前 `make_tensor_placement()` 将详细 HBF 限定为 BankPlane，需推广为所有支持的 HBF 后端；不能把 HBFine 的特定物理布局强加给其他模拟器。
6. 确保 B 的 readonly/preloaded 语义一致，测量阶段 HBF write bytes 应为零。
7. 几组实验可以同时开始跑, 用不同进程. 另外, HBM直接用 simple model 即可.&#x20;

尽量复用现有 `configs/memconf/`，为三种 4F1H 后端分别提供清晰的配置，不重复维护不相关的 GPU 参数。

## Step 6 — 构建与少量验证

优先 Release 构建，只编译 GTSim C++ GEMM 及必要的第三方核心库。

新增的测试限制为：

- **Backend contract smoke**：提交/背压/完成/时间单调性/初始化；三种 HBF 后端复用一套测试逻辑。
- **Placement smoke**：A/C 进入 HBM、B 进入 HBF，FullHBM 下所有请求进入 HBM。
- 运行已有与此次改动直接相关的 Memory routing、HBFine、Ramulator2 测试，保证旧后端未回归。

不运行完整第三方测试集，不运行 TVM、Serving、自动调优或无关测试任务。

## Step 7 — 运行小规模 GEMM 并保存结果

先运行 `M=128, N=256, K=64` 的 C++ GEMM，四组配置各运行一次。若全部稳定完成，可增加一次 `512×512×512` 对照，但不作为必需步骤。

结果目录：

```text
validation/results/1112-hbf-backend-gemm/
  configs/           # 本次运行的有效配置
  logs/              # 四组原始输出
  results.csv        # 对照指标
  results.json       # 机器可读结果与版本信息
  summary.md         # 简短结论、差异和限制
```

统计至少包括：

- GPU 总模拟周期与完成状态
- HBM/HBF read/write bytes、请求数
- HBF 平均请求延迟
- NAND page sense 次数、Page Buffer 命中（如可获得）
- 模拟器自身的 CPU wall time
- Backend 名称、配置、Git commit、submodule SHA
- setup/preload 的时间及流量（与正式 GEMM 分开）

不同 Backend 无法支持的统计字段填 `N/A`，不得伪造零值。允许不同模型产生不同性能结果，但必须核实 A/B/C placement、输入规模、初始化状态和请求完成契约。

## 完成标准

- 新目录和抽象关系清晰，第三方依赖封装在 Adapter 内。
- HBFine、Ramulator2 原有功能不回归。
- EventLab、FlashLab 均通过同一个 `MemoryBackend` 接口接入 GTSim。
- 四组 C++ GEMM 正常结束，结果及真实运行耗时保存在指定目录。
- 没有触发 TVM 或其他无关任务。
- 如 FlashLab 闭环 API 或 EventLab 初始化存在无法解决的问题，记录明确的技术阻塞及复现方式，**不得使用固定延迟、虚假完成或其他简化模型代替目标后端**。

最终只需简短汇报：文件结构调整、两个后端的实际接入方式、构建/测试结果、四组 GEMM 性能表，以及尚未解决的限制。不额外撰写长篇设计文档。

