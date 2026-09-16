# 按特性逐步叠加的性能对比（2026-09-15）

最新真实early exit与1K–64K叠加结果见[2026-09-16结果与分析](feature_stacking_results_20260916.md)。该批次已按用户要求停止，保留537次完整测量。

本文把结果按“哪一步改变了什么”重排。历史测试清单、功能状态、日志审计和已知问题仍见 [全部测试汇总](testing_summary_20260915.md)。

## 1. 术语：这些开关到底改变什么

| 术语 | 在当前代码里的含义 | 可能影响什么 |
| --- | --- | --- |
| Prefill | 处理输入 prompt，建立 KV，得到首个输出所需状态 | 输入阶段耗时；本轮主表的计时从 prefill 完成后开始 |
| Decode / recurrent core | 持续生成 token；每个 token 可多次通过同一组 Transformer 层 | 本轮主要计时对象；轮数和每轮 batch 大小都会影响吞吐 |
| Context / 并发 | Context 是输入上下文长度；并发是同时活跃的请求数，不是 CUDA 流数量 | 每个 query 读多少 KV，以及每轮能合批多少 query |
| Triton attention | 仓库原有、以正确性为先的自定义分页 attention kernel | 是本次 A 的具体实现；不能把结果推广成“Triton 编程语言慢” |
| FLASH_ATTN / FA4 | 接入官方 FlashAttention；本机 B300 实际运行 FA4 4.0.0b30 | 仍计算普通因果 MHA，改变的是执行 kernel，不改变模型结构 |
| Paged KV / Depth-Aware KV | 页表定位 KV；按请求/轮次选择正确缓存。当前性能采用 LAST-EXITED，未执行深度补齐最后退出深度 | 是整个基础实现的一部分；本轮没有拿掉它做独立消融 |
| S：同步单流 | CPU 调度、GPU 执行、结果回收按同步路径推进 | 基准模式 |
| AS：异步单流 | CPU 可以在上次 GPU 工作未结束时准备后续批次；GPU token 直接送下一次 prelude。GPU 工作使用同一执行流 | 隐藏 CPU 调度/准备时间；单流上的 GPU kernel 仍按序执行 |
| AM：异步多流 | 保留 AS，再用 core、boundary、copy 三条 CUDA 流执行可独立的工作 | 尝试让 core 与 prelude/coda/输入复制重叠；不是多卡，也不保证并行有收益 |
| Prelude / coda | Ouro 的 token embedding / 输出头及采样相关执行阶段 | 与其他请求的 core 可能有重叠机会 |
| Static buffers | 预分配、复用输入 metadata 与状态等设备缓冲区 | 减少分配，但也有管理、复制和复用等待成本；不自动固定 batch shape |
| Padding | 每个 stage 的 batch 向上补齐到 2 的幂，例如 33→64；虚拟行不能写真实 KV 或输出 token | 形状种类减少，但增加无效计算；不是全部补到 token budget=2048 |
| CUDA graphs | 把一组 GPU 操作捕获后重放，减少逐算子 CPU 提交开销 | 尚未实现、未开启；static/padding 不等于 graphs |
| fixed4 | 每个 decode token 固定执行四轮 | 用于固定计算量对照 |
| mixed234 | 用阈值/最小深度控制不同请求分别在 2/3/4 轮退出；prefill 首输出仍为四轮 | 改变工作量和 stage 分布；不是模型质量已验证的自适应早退 |
| Lookahead gate | 提前给出后续轮次退出决定，让 CPU 有时间准备 batch | 目前没有蒸馏权重；本表不衡量训练后 lookahead 的收益或精度 |
| tok/s / p95 ITL | 每秒交付的输出 token 数 / 主机观察到的 token 间隔第95百分位 | 前者越高越好、后者越低越好；均不是 GPU idle 百分比 |

## 2. 叠加顺序和缺失组合

| 阶段 | 保留的基础 | 本步唯一变化 | Attention | Async | 多流 | Static / pad / graphs | 实测状态 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E0 基础 | 当前修复后的队列调度、depth-aware paged KV、退出控制等 | 起点 | 自定义 Triton | 关 | 关 | 全关 | 已测 |
| E1 更换 attention | E0 | Triton 换为官方 FA4 | FA4 | 关 | 关 | 全关 | 已测 |
| E2 加 async | E1 | 开启 CPU prepare / GPU forward pipeline | FA4 | 开 | 关 | 全关 | 已测 |
| E3 加多流 | E2 | core/boundary/copy 分流 | FA4 | 开 | 开 | 全关 | 已测 |
| E4 再加 static | E3 | 开启静态缓冲区复用 | FA4 | 开 | 开 | static 开，其余关 | 没有正式性能数据；只有相关功能 smoke |
| E5 再加 padding | E4 | batch 向上补齐 | FA4 | 开 | 开 | static/pad 开，graphs 关 | 没有正式性能数据；只有相关功能 smoke |
| E6 再加 CUDA graphs | E5 | 捕获/重放 GPU 操作 | FA4 | 开 | 开 | graphs 开 | 未实现，按用户要求暂不做 |

这是 **S → AS → AM** 的可测叠加链，没有“同步多流”这一中间档。早退策略在每条链内保持不变；mixed234 单列为另一种负载。
页式 KV、depth-aware 布局、队列调度已经在 E0 中，不能给它们填写本轮不存在的单项收益。这里也不拿早期有 batch 碎片化的版本作 E0。

## 3. 性能大表：E0 → E1 → E2 → E3

所有格子来自同一批 144 次正式 A/B 测量，而不是从不同历史实验抽取最好值。
每个 context 在同一张 B300 上交错测量两个后端；使用同一份实际 Triton prefill 结果和同一个 KV 池。
BF16、LAST-EXITED、page=16、输出128、greedy/ignore_eos、token budget/chunk=2048，static/pad/graphs 全关。
每组完整预热，正式重复三次，以下为吞吐中位数，单位 **tok/s**。

| 上下文 | 并发 | 退出 | E0 Triton + S | E1 FA4 + S | E2 FA4 + AS | E3 FA4 + AM | E3 相对 E0 累计收益 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1K | 1 | fixed4 | 36.07 | 29.61 | 29.64 | 29.54 | -18.1% |
| 1K | 32 | fixed4 | 869.77 | 873.40 | 872.33 | 863.80 | -0.7% |
| 1K | 128 | fixed4 | 1869.51 | 2648.58 | 2611.79 | 2662.00 | +42.4% |
| 1K | 128 | mixed234 | 2073.71 | 2744.49 | 2716.45 | 2684.51 | +29.5% |
| 4K | 1 | fixed4 | 19.07 | 27.08 | 27.03 | 26.96 | +41.4% |
| 4K | 32 | fixed4 | 399.30 | 782.83 | 795.29 | 786.94 | +97.1% |
| 4K | 64 | fixed4 | 623.30 | 1082.58 | 1191.55 | 1195.11 | +91.7% |
| 4K | 64 | mixed234 | 691.89 | 1216.16 | 1326.15 | 1223.14 | +76.8% |

这张表的“累计收益”分母始终是 **E0 的 Triton 同步**。之前 A/B 表的分母是**同一调度模式的 Triton**，因此不能混用两种百分比。
例如 4K/C64 的 E3/E0 为 +91.7%；之前 FA4-AM/Triton-AM 为 +81.5%。两者都正确，但回答的问题不同。

### 每一步单独贡献多少

单步收益 = 本阶段吞吐 / 上阶段吞吐 − 1。累计倍率是各步倍率相乘，百分比不能直接相加。

| 上下文 | 并发 | 退出 | E1/E0：换 FA4 | E2/E1：加 async | E3/E2：加多流 |
| --- | ---: | --- | ---: | ---: | ---: |
| 1K | 1 | fixed4 | -17.9% | +0.1% | -0.4% |
| 1K | 32 | fixed4 | +0.4% | -0.1% | -1.0% |
| 1K | 128 | fixed4 | +41.7% | -1.4% | +1.9% |
| 1K | 128 | mixed234 | +32.3% | -1.0% | -1.2% |
| 4K | 1 | fixed4 | +42.0% | -0.2% | -0.3% |
| 4K | 32 | fixed4 | +96.0% | +1.6% | -1.1% |
| 4K | 64 | fixed4 | +73.7% | +10.1% | +0.3% |
| 4K | 64 | mixed234 | +75.8% | +9.0% | -7.8% |

小于约 1% 的差异在这里只有三次重复，且未锁时钟，不能当作可靠优化收益；原始每轮值与完整 p95 ITL 见 [A/B 报告](flash_attention_ab.md)。

## 4. Static / padding 作为另一条已测分支

不能把下面数字接到 E3 后面：这一批使用 **Triton**，不是 FA4。
为了每一步只改变一个变量，固定后端和调度模式，仅比较 dynamic → static → static+pad；都来自同一批 81 次实验。
1K 上下文，其余计时条件见 [static 结果](testing_summary_20260915.md)。

| 并发 | 退出 | 固定调度模式 | dynamic tok/s | + static tok/s | 再 + padding tok/s | static 单步收益 | padding 单步收益 |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 32 | fixed4 | S | 875.05 | 842.80 | 848.58 | -3.7% | +0.7% |
| 32 | fixed4 | AS | 931.54 | 918.44 | 917.96 | -1.4% | -0.1% |
| 32 | fixed4 | AM | 928.34 | 921.75 | 923.53 | -0.7% | +0.2% |
| 128 | fixed4 | S | 1886.45 | 1748.39 | 1747.29 | -7.3% | -0.1% |
| 128 | fixed4 | AS | 2039.67 | 2010.97 | 2010.38 | -1.4% | -0.0% |
| 128 | fixed4 | AM | 2042.27 | 2011.21 | 2009.23 | -1.5% | -0.1% |
| 128 | mixed234 | S | 2079.60 | 1943.87 | 1927.92 | -6.5% | -0.8% |
| 128 | mixed234 | AS | 2281.51 | 2248.16 | 2228.86 | -1.5% | -0.9% |
| 128 | mixed234 | AM | 2142.22 | 2126.09 | 2102.04 | -0.8% | -1.1% |

这里 static 单独开启后始终低于 dynamic。Padding 相对 static 某些配置略高、某些略低，但没有让最终 static+pad 超过 dynamic。
固定四轮的 32/128 本来就是 2 的幂，padding 通常没有新增有效的形状归一化效果；混合退出时 stage batch 会改变，padding 才可能补入虚拟行。具体开销原因未做专项 profiling，不能仅凭表格归因。

## 5. 分析：哪些收益已经成立，哪些不能下结论

**第一，主要收益来自更换 attention，但有明确的负载边界。**
4K/C32 的 E0→E1 接近翻倍；4K/C64 增加73.7%，1K/C128增加41.7%。
反例是1K/C1下降17.9%，1K/C32基本持平。测到的是整个引擎，包含 kernel 的 Python 调用、metadata 和 GPU 执行成本，不能概括为 FA4 kernel 在所有形状上更快。

**第二，async 的收益取决于 GPU 工作是否足够长，不能和 attention 加速混在一起。**
FA4 下，4K/C64 的 E1→E2 固定四轮为 +10.1%，混合退出为 +9.0%；其他已测点约 −1.4% 到 +1.6%。
这证明当前实现存在有意义的 async 收益，但不是普遍收益。已记录的 4K/C1 prepare 事件显示，Triton 通常还有前序 GPU 工作未结束，FA4 则没有；这是重叠机会变化的证据，不是对开销根因或空泡比例的完整证明。

**第三，多流不是 async 的同义词，额外多流收益很有限，且会负向作用。**
4K/C64 fixed4 的 E2→E3 只有 +0.3%，不能称为显著提升；同一上下文并发的 mixed234 则下降7.8%。
1K/C128 fixed4 的 +1.9% 也只是本批次观察，不能预设所有负载获益。要定位 mixed234 的多流退化，需要看实际 stage batch、跨流等待和资源争用；当前没有新 profiling 证明其中哪一项是根因。

**第四，提前退出要另看工作量和精度。**
例如 4K/C64、FA4+AS：fixed4 为1191.55 tok/s，mixed234为1326.15，提升约11.3%。
这里同时减少部分请求的轮数并改变队列分布，不能当作“async 又优化11.3%”。控制的2/3/4退出不是经过任务精度验证的自适应策略，也不是蒸馏 lookahead。
在每种退出策略内部比较 E0→E3，才能看到该负载下各执行特性的贡献。

**第五，static 未显示收益，graphs 未测，不能对齐论文的空泡数字。**
本文只有 Triton 分支的 static/pad 性能，没有 FA4+static 的正式性能数据。
论文报告包含 CUDA graphs 等优化，当前既未实现 graphs，也未统计跨流 GPU idle 占比；不能把 E3/E0 的吞吐提升解释成消除了同等比例的空泡。

## 6. 当前结果支持的使用判断

- 对已经测试的4K负载，FA4有明确吞吐优势；4K/C64可看到额外的AS收益。
- 1K小并发应保留Triton选择；FA4不能仅因为更新就默认视为更快。
- 多流和static/padding均应保留独立开关，不能把“全部打开”当作最佳配置。
- 要补齐完整叠加链，应在同一实验中补测 FA4+AS/AM 的 static/padding；这次只是重组已有结果，没有偷偷把缺失组合补成预测值，也没有新增测试或实现 CUDA graphs。

两个主要数据源均已结束、进程退出0、每条被测配置输出token/退出深度一致：
[FA4 A/B 原始汇总](/home/zjy/code/david/tmp/ouro-flash-attn-20260915/ab/results-summary.md)、
[static 原始汇总](/home/zjy/code/david/tmp/ouro-context-sweep-20260915/static-ablation/results-summary.md)。
功能正确性的范围和未覆盖项仍见 [全部测试汇总](testing_summary_20260915.md)，不能用性能结果代替 GSM8K 等质量评估。

补测设计（真实 early exit、1K–64K、TTFT/TPOT/ITL/E2E）见
[完整实验方案](feature_stacking_experiment_plan.md)，尚未启动。
