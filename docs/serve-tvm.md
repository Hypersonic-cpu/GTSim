# A100 / 3M2F 的 TVM serving 探索

展示入口：[结果摘要与12根柱子的图](serve-results.md)。

本实验的输入是功能语义，输出是可重放的已调度 TensorIR 和 GTSim Tile IR。
使用真实 Qwen3.5-9B 的维度、batch=1、32K KV；没有读取或模拟 checkpoint 数值对耗时的影响。
数值验证使用独立生成的 BF16/FP32 输入；耗时目标是 **Simple 模型仿真周期**。

## 路径与搜索边界

- 无调度约束的算法：workloads/serve/semantics.py。
- 原生调度规则：scripts/tvm/serve_rules.py，调用 TVM split/fuse/bind/compute_inline/sample_categorical。
- 原生 MetaSchedule EvolutionarySearch、反馈模型、builder/runner：scripts/tvm/serve_search.py。
- 数学 SBlock 到向量图：scripts/tvm/serve_lower.py。
- stable-softmax attention 到在线 FA2 的编译转换：scripts/tvm/serve_fa2.py。
- 显式放置清单：scripts/tvm/serve_memory.py；TIR 属性 gtsim.semantic_tensor_tiers。
- 独立执行已保存在线 TIR 的数值解释器：scripts/tvm/serve_eval.py。
- C++ 图导入器：build/gtsim_serve_ir，读取 Tile IR v2。

MetaSchedule 真实生成候选、调用 GTSim，并用测得周期更新 ridge 代价模型，再指导进化搜索。
保留原生 tuning database、每轮 cost-model 观测、trace、候选 IR、失败原因及资源记录。
这不是 C++ builder 参数扫描，也没有把未优化语义 IR 标为最优 IR。

FA2 的在线 softmax / split-KV 重写由编译器提供，并非 TVM 自动发现新的 attention 算法。
TVM 搜索 KV tile（32/64/128）、split（16/32/47/64/96/128）、warp（1/2/4）和缓存阶段（1/2）。
ISA 固定为 Ampere warp MMA m16n8k16；共享加载、四槽寄存器片段复用和基本布局由编译器实现。
D256 的资源约束会剔除无法容纳的 tile，而不是调整硬件容量。

投影使用 BF16 输入、FP32 归约的直接 GEMV，避免固定进行 M=64 的无效行计算。
TVM 搜索输出 tile（64/128/256）、K tile（32/64/128）、warp（1/2/4）和预取阶段（1/2/3）。
完整 GDN 还提供三个合法 pointwise 融合族；融合指令保留在可重放 TVM trace 中。
独立投影分别搜索，避免强迫所有大小不同的矩阵共用同一 tile。
完整 GDN 搜索则记录一套可运行的连接图；另可将独立搜索最优投影布局组合后单独测量。
这些空间没有覆盖所有可能的布局、重排、融合、跨层预取或算法，最优指“已测候选中最优”。

## 功能语义与放置

Qwen 参数：hidden=4096、FFN=12288；GQA=16Q/4KV、D256；24GDN+8GQA。
GDN=16个D128的key头、32个D128的value头，QKVZ输出12288，b/a输出64。
输出投影4096→4096，gate/up为4096→24576，down为12288→4096。

GDN 功能图包含输入 RMSNorm、QKVZ/b/a、四槽因果卷积与SiLU、Q/K归一化、
gated-delta递推、BF16 attention输出边界、gated RMSNorm、输出投影、residual、
post norm、SwiGLU MLP及最终residual；输出新卷积状态和FP32递推状态。
物理卷积状态为 [B,8192,4] BF16；FP32递推状态为 [B,32,128,128]。
读写新的状态均在HBM，下一次decode可将新状态作为输入；它们不是学习权重。
归约与norm计算采用FP32，模型参数和普通激活为BF16。

3M2F 中，所有 W_*、A_log、dt_bias 明确在 HBF；
输入/输出、Q/K/V/Z、KV cache、归约暂存、卷积状态、递推状态及中间结果均在 HBM。
全 HBM 对照的所有参数均在 HBM。参数角色先写入 TIR 清单，lowering 从清单取放置；
缺少清单、状态进入HBF或拓扑无法提供声明设备时拒绝编译，不静默重映射。
Tile IR 经 TensorRegistry 保留设备身份，再交由既有 Simple 模型路由。
FA2 的 HBF 读写必须为0；整个实验不允许 HBF 写流量。

串联 SBlock 使用全完成边界，不能把前一kernel刚发出的store视为已经可读。
寄存器环和共享环的下一次复用等待此前消费者完成；stage边界使用共享join，
避免用二次规模的all-to-all边膨胀宿主机内存。每个CTA显式声明warp、shared、register预算。
可变warp CTA按每个SM的轮转分配子分区，避免1-warp CTA被固定编号误钉在SP0。

## 配置与复核

两份A100配置均显式设置 SIMD instruction quantum=32、每子分区width=16：
108 SM × 4 SP × 16 FMA/cycle × 2 FLOP/FMA × 1.410 GHz = 19.49184 TFLOPS。
这是对继承的width=32的规格修正；全HBM、3M2F基线与最终搜索在统一配置下重测。
Tensor Core和内存参数没有按trace拟合；HBM/HBF带宽、非对称读写延迟仍按已确认的假设。
始终使用 Simple HBMModel/HBFModel，没有 Ramulator2 或 HBFine。

原来错误固定warp间距的运行保留在 tvm-search/，标为 invalidated，不参与结论。
继承SIMD width=32的探索保留在 tvm-search-sp-balanced/，属于旧模型审计；
最终结果只取 study-a100-fp32/。当前研究入口先串行重测12项基线，再运行搜索矩阵。

预算：FA2 12次、5种投影各8次、完整GDN 16次，共68次候选预算。
预算包含构建拒绝；成功数量与拒绝原因保留在结果中。
先前单次峰值已超过3GB，因此严格 **1个worker**；最终68个候选的最大值为7.778GB（RSS/footprint两者取大）。
CPU线性代数与TVM内部线程也设为1；不并行启动多个仿真。
等待使用逐渐延长的检查间隔，不把宿主机仿真运行时间作为GPU执行时间。

## 结果与 TPOT 口径

最终机器可读汇总、历史、database、反馈模型及最优IR位于：
results/serve-3m2f/tvm/。最优IR放在其 best-ir/ 子目录；
FA2同时保存 scheduled.tir 和在线 native.tir，GDN/投影保存实际已调度功能TIR。
每个搜索最优候选均重放trace并再次仿真，确认周期一致。

TPOT按已确认的方法推算：

    TPOT_hybrid_est = TPOT_NoOff_measured × T_sim_hybrid / T_sim_normal

两种仿真必须覆盖相同组件。主对比继续采用：
GQA的四个大投影+FA2，以及GDN的四个大投影，按8/24层加权。
完整GDN另行报告实测仿真耗时，不把它与旧投影子集混进同一个比值。
这个模型假设未显式仿真的辅助算子、LM head、调度/服务开销也同比缩放；
因此3M2F栏是**校准估计**，不是物理HBF实测或实际完整32层执行时间。

HostRAM使用客户端端到端398.838ms/token，包含等待CPU权重、传输、同步及服务路径。
全HBM使用同请求的客户端37.376ms/token。二者来自共享A100的单次采集，非独占GPU多次统计。

为让12根柱子的层级与整步口径一致，新图的前两组定义为“每逻辑层的摊销TPOT贡献”，
把整步层外开销也分摊进去，并检查每个场景 8×GQA+24×GDN=TPOT。
HostRAM原始完整层wall（12.383/11.205ms）先保留，再按全HBM仿真组件占比分摊剩余层外时间。
全HBM/HBF/TVM层贡献使用统一校准系数。它们不是单kernel时长或NoOff逐层wall实测。
原始层wall仍在 wall-time-audit.json 中，避免用kernel-only数据替代端到端结果。

## 实际完成结果

主预算68个候选全部完成：FA2 12个、5种投影各8个、完整GDN 16个。
7个赢家均完成trace重放、结构一致性检查与仿真周期复测。

| 相同算子范围，μs | 3M2F 原实现 | 3M2F + TVM | 加速 |
| --- | ---: | ---: | ---: |
| FA2 核心 + combine | 173.621 | 157.656 | 1.10× |
| GQA QKV+gate | 170.621 | 166.338 | 1.03× |
| GDN QKVZ | 200.562 | 198.272 | 1.01× |
| 输出投影 | 163.193 | 75.289 | 2.17× |
| MLP gate/up | 391.731 | 390.411 | 1.00× |
| MLP down | 479.221 | 203.289 | 2.36× |

完整GDN：首个可行2273.809μs → 最优1485.908μs（1.53×）；
最优Ntile64/BK128/2warps/1stage/融合族1，三个融合族分别覆盖6/4/6个候选。
FA2最优KVtile32/split16/4warps/2stages；投影最优均为Ntile128/BK128/4warps/3stages。
这些投影使用SIMD GEMV，不能称作TVM发现Tensor Core布局。

TPOT校准结果：5M实测37.375790ms、HostRAM实测398.838189ms、
3M2F估计127.738092ms、3M2F+TVM估计90.346741ms。
模型配置修正、编译路径变化与MS空间内搜索的贡献应区分；
results.json还单列“首个可行候选→最优”，不把所有收益都归因于搜索本身。

固定Ntile128/BK128/4warps，只改变register ring的深度：

| 架构 / μs | 1stage | 2stages | 3stages |
| --- | ---: | ---: | ---: |
| 5M | 40.131 | 30.329 | 30.167 |
| 3M2F | 165.442 | 90.765 | 75.289 |

各次均读取同样的32MiB权重，3M2F的HBF写为0；对照支持预取隐藏高读延迟的解释。
3M2F输出投影权重带宽从原实现205.612升至445.677GB/s，仍低于524.288GB/s；
仅权重读取下界64μs，最优75.289μs没有突破带宽下界。
名义32个CTA的B tile容量随深度为1.049/2.097/3.146MB；
HBF带宽×读延迟为2.097MB。这是容量代理，不能当成实际在途字节测量。

本次主搜索结束后准备了原生rfactor的b/a Split-K扩展；12种split/fusion组合的两步状态
数值与重放检查通过。用户要求收尾后停止额外性能搜索，其未完成候选不参与任何图表或最优结论。
入口为scripts/tvm/hierarchical_serve_search.py，状态见tvm/optional-extension-status.json。

主搜索源码快照、SHA清单及阶段完成记录位于tvm/provenance/。
源快照不含原始NSys trace、checkpoint或第三方TVM源码；第三方版本按pinned manifest单独获取。
用于展示的仅是本实验生成的IR、日志、派生统计和图表，不重新分发原始采集数据。

## 验证与复现

独立NumPy GDN参考检查两次连续decode的输出、卷积状态和递推状态；
三种融合族均通过，并验证TVM trace重放的结构一致性。
直接执行已保存的在线FA2 TIR，检查dense attention参考、尾块、空split及batch1/2。
C++边界测试检查BF16、HBF权重路由、HBM共享/寄存器身份、Ampere MMA及子分区分配。
硬件配置与二进制/源码SHA256、原始日志、内存统计均保留，旧H100确定性参考不变。

运行：

    .tvm-venv/bin/python scripts/experiments/run_serve_tvm_study.py
    .tvm-venv/bin/python scripts/experiments/summarize_serve_searches.py \
      --search-root validation/results/serve-3m2f/study-a100-fp32/search \
      --baseline-operators validation/results/serve-3m2f/study-a100-fp32/baseline/results.json
    python scripts/experiments/plot_serve_tvm.py \
      --data docs/results/serve-3m2f/tvm/results.json \
      --output docs/results/serve-3m2f/comparison-tvm.png

研究入口拒绝覆盖已有目录。完整数据已存在时，直接汇总/重放对应文件；
重新探索应使用新的输出目录。绘图需要matplotlib，不要求CUDA或重新编译TVM。
本机没有执行真实GPU CUDA kernel；数值解释器和静态资源预算也不是ptxas寄存器验证。
SIMD/SFU、寄存器占用和连续KV布局仍是tile级抽象，未复现vLLM paged-cache的所有地址细节。

依据：[Qwen官方配置](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json)、
[Transformers Qwen3.5实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)、
[FA2实现](https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_fwd_kernel.h)、
[TVM MetaSchedule](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html)、
[NVIDIA A100架构](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)。
