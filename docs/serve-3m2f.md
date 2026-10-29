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

## 图和TPOT的限制

`results/serve-3m2f/components.png`是九柱**已覆盖组件**图，不是完整layer/TPOT图。
GQA覆盖投影、MLP、FA2；GDN覆盖四个大投影；第三组仅8×GQA+24×GDN。
HostRAM列只计匹配GPU kernel duration，排除了CPU offload等待，不能表示over-subscription的TPOT。
真实客户端TPOT：HostRAM 398.838ms；NoOff 37.376ms。3M2F完整TPOT尚未测得。
不能将40.656ms组件之和与上述客户端TPOT比较后声称加速。

尚需完整GDN/辅助算子/LM head/调度和传输覆盖及正常trace逐核导出，才能形成所要求的TPOT图。

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
