# A100 3M2F Simple 模型基线（阶段性记录）

配置：`configs/v0.1.0/a100-pcie40-3m2f-grade1-simple.yaml`。
仅用原有固定延迟/带宽队列 HBMModel、HBFModel，不启用 Ramulator2 或 HBFine。

| 参数 | 5M | 3M2F |
| --- | ---: | ---: |
| HBM 读/写上限 GB/s | 1555 / 777.5 | 933 / 466.5 |
| HBM 读/写延迟 cycles | 270 / 270 | 270 / 270 |
| HBF 读/写 GB/s | — | 524.288 / 27.96 |
| HBF 读/写延迟 | — | 4 / 75 μs = 5640 / 105750 cycles |
| HBM/HBF 链路 GB/s | 7680 / — | 4608 / 768 |

频率1410MHz、108SM。HBM带宽/容量/链路按3/5缩放，容量变为24GiB。
写带宽取读带宽一半为继承的模型假设，非官方A100写带宽保证。
HBF每堆栈：8-Hi、256GiB、读262.144/写13.98GB/s；两堆栈聚合。
这是用户确认的研究假设：将原16-Hi Simple速率524.288/27.96减半。
Grade-1接口上限384GB/s/stack，不等于NAND持续读速率，也未冒充实测规格。
HBF读写延迟沿用原模型4/75μs，不随堆栈数缩放。

## 放置与覆盖检查

- GEMM：A（激活）HBM；B（静态权重）HBF；C/partial C HBM。
  已修复原builder把A也标成HBF的问题，新增tier回归断言。
- FA2：Q/K/V、KV cache、O、partial O/LSE全部HBM；该内核没有学习权重。
  QKV/O投影权重属于独立GEMM，不应把KV cache当成权重放HBF。
- GDN语义：卷积状态、递推状态、Q/K/V/Z、门控中间值均应在HBM；
  qkvz/ba/out/MLP/conv/norm及A_log/dt_bias等只读参数应在HBF。
  当前实测覆盖四个大投影；小ba、conv/norm参数尚未纳入混合内存时序仿真。
- 输出/后继输入的tier约定都是HBM。当前是独立kernel进程的冷启动基线，
  尚未组成共享tensor identity、缓存状态及跨kernel依赖的完整decoder图。

Qwen3.5-9B，batch1，BF16，32K，FA2=16Q/4KV、D256、47split；
GDN QKVZ=4096→12288、out=4096→4096，MLP=4096→24576→12288→4096。

| 单算子 μs | 5M | 3M2F |
| --- | ---: | ---: |
| FA2 核心+combine | 118.191 | 173.150 |
| GQA QKV+gate投影 | 55.711 | 170.621 |
| GDN QKVZ投影 | 66.314 | 200.562 |
| 输出投影 | 38.200 | 163.193 |
| MLP gate/up | 131.073 | 391.731 |
| MLP down | 109.056 | 479.221 |

流量日志验证：FA2 HBF读写均0；所有3M2F GEMM的HBF读恰为B的BF16字节数，
HBF写为0；5M HBF流量为0。详见`results/serve-3m2f/operators.json`。
复现：`python3 scripts/experiments/run_serve_3m2f.py`，原始日志在
`validation/results/serve-3m2f/`。4项相关CTest及确定性回归通过。

## 端到端 TPOT 与 HostRAM 等待时间（修正）

此前九柱组件图中的 HostRAM GPU-kernel 求和及“32-layer covered components”
不能反映 CPU offload 的端到端代价；该比较已撤回。
`results/serve-3m2f/components.png` 现在仅作两种内存配置的仿真组件诊断，
不再绘制 HostRAM GPU-only 柱或32层求和柱。历史 covered-components.json
中的 oversubscription_trace_gpu 和 decode_component_sum 仅为诊断数据，不能直接作为 TPOT。

修正后的图：[`comparison.png`](results/serve-3m2f/comparison.png)。

| 整个 decode step | TPOT ms/token | 来源 |
| --- | ---: | --- |
| HostRAM oversubscription | 398.8382 | 客户端逐输出事件，含暴露的传输等待及完整服务路径 |
| 全 HBM / NoOff | 37.3758 | 客户端逐输出事件，完整服务路径 |
| 3M2F | 127.8094 | 以全 HBM 实测值校准仿真比值后的估计 |

两组实测请求内容完全相同：batch1、BF16、32768输入、256输出。
直接重读各自的256个非空 SSE 事件，核对单调时钟端点及 usage，按
`TPOT = (last_text_time - first_text_time) / 255` 复算，与原 trace-metrics.json 一致。
第一个输出由 prefill 产生，TPOT排除TTFT，覆盖其后255个输出间隔。
该定义保留 kernel 之间等待CPU权重/传输/同步的时间，也覆盖LM head、采样、
调度和服务返回的可见延迟。异步复制若与计算重叠，不再把复制耗时重复相加。
HostRAM/NoOff实测比为 **10.6710×**，不是通过乘10拟合得出。

### 3M2F 的 TPOT 推算

沿用3 HBM + 2 HBF，使用两种配置**相同覆盖范围**的仿真时间比值：

```text
T_sim_5M   = 8 × GQA_components_5M   + 24 × GDN_components_5M   = 11.889287943 ms
T_sim_3M2F = 8 × GQA_components_3M2F + 24 × GDN_components_3M2F = 40.656323404 ms
ratio = T_sim_3M2F / T_sim_5M = 3.419575974
TPOT_3M2F_est = TPOT_NoOff_measured × ratio
              = 37.375789765 × 3.419575974 = 127.809352706 ms/token
```

按该推算，HostRAM/3M2F为 **3.1206×**。40.656ms仍是仿真组件和，
图中TPOT柱使用校准后的127.809ms。
该估计假设完整服务TPOT随当前覆盖组件的仿真比值同比缩放；
目前GQA覆盖投影、MLP和FA2，GDN覆盖四个大投影，辅助算子、LM head及host调度
尚未显式仿真，它们的缩放是模型假设。结果标为**校准估计**，不是3M2F真机实测，
也不是已经执行了完整decoder图。将来扩充算子覆盖后，重新计算相同口径的比值即可。
不把HostRAM的传输等待转移到HBM/HBF估计中。

### 层级 wall time 审计

HostRAM按255轮完整decode、每轮24GDN+8GQA逐层核验，匹配顺序为`DDDQ`重复8次。
每层窗口定义为“前一层MLP down结束→本层MLP down结束”，第一层从embedding结束起计。
这样保留入口等待及层内全部kernel/空隙；不会把发生在下一个input norm前的等待删掉。
该入口等待归属是一条明确的记账边界，并不表示预取一定属于该层权重。

| HostRAM | 样本数 | 平均 wall time ms |
| --- | ---: | ---: |
| GQA完整层 + 入口等待 | 2040 | 12.382993 |
| GDN完整层 + 入口等待 | 6120 | 11.205265 |
| 32层block | 255 | 367.990303 |

32层block还没有覆盖层外路径；不能用367.990ms替代398.838ms客户端TPOT。
另取254个可闭合GPU窗口（embedding完成至下一轮embedding完成）核验：
32层平均367.986149ms + 层外平均30.983571ms = 窗口平均398.969720ms。
它与客户端采用不同端点/样本数，保留其差别，不强行配平。
kernel CSV可确定全部等待被保留，但无法单独区分memcpy、host调度、同步、队列等原因。
当前没有NoOff逐层wall导出，图中的NoOff/3M2F层级wall值留N/A，未将组件仿真冒充完整层wall时间。

两次实测都来自共享A100的单次运行，HostRAM/NoOff的KV cache预算分别10/5GiB；
10.6710×是现有两次运行之比，不能视为独立隔离的PCIe传输惩罚。

数据及SHA256来源清单：`results/serve-3m2f/wall-time-audit.json`；
255轮逐步分解：`results/serve-3m2f/hostram-step-wall-times.csv`。
复现（`$TRACE_DATA_ROOT`指NSysTracer的data目录）：

```bash
python3 scripts/experiments/audit_serve_wall_time.py \
  --data-root "$TRACE_DATA_ROOT" \
  --sim-components docs/results/serve-3m2f/covered-components.json \
  --output validation/results/serve-wall-time-audit
python3 scripts/experiments/plot_serve_wall_time.py \
  --data validation/results/serve-wall-time-audit/wall-time-audit.json \
  --output validation/results/serve-wall-time-audit/comparison.png
```

## TVM进展与未完成项

`tvm`已是`serve-sim`祖先，实际执行merge返回Already up to date。
已加载并校验pinned TVM，FA2/GDN功能TE可导出TensorIR；源码位于
`workloads/serve/semantics.py`，不固定tile、warp或流水线。
现有自动搜索后端仅识别一个GEMM reduction与cast输出，使用Hopper WGMMA/TMA；
其M轴必须是64倍数，因此既不支持decode M=1，也不支持FA2多reduction和GDN状态递推。
尚未扩展该lowering，未完成MetaSchedule搜索，没有最优IR或优化后的12柱图。
语义IR不应被标为优化IR，手写C++参数扫描也不应被标为TVM自动优化。

实测最大RSS约2.30GB，但macOS peak memory footprint为3,772,419,264B。
按更保守值超过3GB，后续TVM与仿真最多**1个worker**；详见worker-policy.json。

导出命令：`.tvm-venv/bin/python scripts/tvm/export_serve_semantics.py`。
功能IR及放置清单位于`results/serve-3m2f/semantic-ir/`，状态明确为
`semantic_export_only_not_optimized`。`search-status.json`的`best_ir`为null。
