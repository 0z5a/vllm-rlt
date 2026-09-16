# 全部测试汇总（2026-09-15）

最新真实early exit与1K–64K叠加结果见[2026-09-16结果与分析](feature_stacking_results_20260916.md)。该批次已按用户要求停止，保留537次完整测量。

按特性逐步叠加的性能大表、术语解释及单步收益分析，见 [叠加对照报告](feature_stacking_results_20260915.md)。

## 当前结论与阅读口径

- 已完成的主要功能回归、真实模型重放和正式性能扫描，见下面逐批记录。最新完整回归为 **272 passed、11 skipped**；这是一个测试集快照，不能与之前的 244/249 项相加。
- 真正的 CPU prepare / GPU forward pipeline 已验证；早期短上下文严重退化得到修复，但 **AM 不是所有负载都更快**。Triton 混合退出在 4K/16K/64K 仍有退化。
- 仅开启 static buffers / padding 没有获得性能提升。这里没有 CUDA graphs，不能当作论文完整 static-shapes 优化的复现。
- 官方 FlashAttention 在本机 B300 使用 **FA4 4.0.0b30**。与 Triton 同卡配对，4K 约提升 38%–96%，1K 小并发退化；它不是论文 H100 上的 FA3。
- 已测工作负载中的 token/退出深度一致不等于完整模型任务精度通过。当前分支没有完成新的 GSM8K 评估，也没有蒸馏后的 lookahead 权重。

S = 同步单流；AS = 异步单流；AM = 异步多流（当前实现有 core、boundary、copy 三条流）。
除特别说明，性能使用真实 Ouro-1.4B、BF16、LAST-EXITED、greedy、ignore_eos。
论文的设备 idle 百分比尚未复现；吞吐、nvidia-smi 利用率、prepare 事件计数不能代替它。

## 1. 测试批次总表

| 批次 | 测量/用例数 | 最终状态 | 主要发现 | 报告 |
| --- | ---: | --- | --- | --- |
| 早期 A/B/C/D 基线 | 48 正式测量 | 历史结果，输出核验通过；未留存 OS 退出码 | C8 多流较同步慢 17.1%；旧 async 实现 | [早期基线](performance_results_20260915.md) |
| 早期 S/AS/AM、dynamic | 36 正式测量 | 历史结果，进程正常退出 | C8 AS/AM 慢约 19%；关闭 static 没有解决旧实现退化 | [早期 dynamic](performance_dynamic_results_20260915.md) |
| Torch ops/shapes + 因果诊断 | S/AS/AM traces + 独立计数/消融 | 已完成，诊断数据 | 旧 coda 回填时机导致 batch 碎片化；等待消融只用于定位 | [profiling](profiling_ops_shapes_20260915.md) |
| 完整 async pipeline 回归 | 当时 244 passed / 11 skipped | 历史回归快照 | 延迟交付、GPU token 反馈、EOS、取消、复用、DMA、重叠依赖 | [async 实现](async_scheduling.md) |
| 修复后短上下文 S/AS/AM | 27 正式测量 | 已完成，输出一致 | C1/4/8 接近同步，未证明稳定提速 | [async 结果](async_scheduling.md) |
| 真实 ragged 重放 | 12 配置 | 已完成，同 dtype/layout 内一致 | FP32/BF16 × 两种 KV × S/AS/AM，深度 1–4 | 同上，`ragged-real.json` |
| 1K→64K Triton 扫描 | 207 正式测量 | 207/207，四进程退出 0，输出一致 | AS 在部分并发有收益；长上下文混合退出 AM 退化 | 本文第 4 节 |
| Static/padding 消融 | 81 正式测量 | 81/81，退出 0，跨配置输出一致 | 三种负载均未发现收益 | 本文第 5 节 |
| FlashAttention 接入回归 | 完整套件 272 passed / 11 skipped | GPU 用例实际运行 | 新增分页数值与 S/AS/AM 验证 | [后端接入](flash_attention.md) |
| FA4 真实模型 smoke | 12 配置 + 单请求 CLI | 已完成 | 固定/混合退出、dynamic/static-pad 一致，自动 KV 启动成功 | 同上 |
| Triton / FA4 配对 A/B | 144 正式测量 | 144/144，两进程退出 0，跨后端输出一致 | 更长上下文/大并发获益，短上下文小并发退化 | [完整 A/B](flash_attention_ab.md) |

旧报告描述当时的代码；后续的通过记录不会把旧失败改写为通过。不同批次的 KV 池大小、输入、计时方式和代码快照不同，不能跨批次相除来声称加速。

## 2. 功能验证覆盖与边界

详细逐项操作表仍以 [functional_validation.md](functional_validation.md) 的 F01–F27 为入口。
下面的“通过”仅针对实际执行的用例，不代表所有参数和硬件组合均已验证。

| 功能 | 实际检查 | 结果与边界 |
| --- | --- | --- |
| 权重、模型和 prefill | 权重 roundtrip/错误拒绝；逐深度 dense 对照；packed/chunked 因果性 | 自动用例通过；真实 Ouro 能加载并生成 |
| Depth-aware KV | LAST-EXITED 深度补齐、SHARED 覆盖、ragged 深度、跨 block、旧页复用隔离 | 两种布局的自动用例和真实重放通过；不要求两种布局彼此输出相同 |
| 自动 KV 容量 | 字节预算、CUDA peak profile、容量上限、取消后回收 | 原后端与 FA4 自动容量真实启动验证通过 |
| Admission / chunked prefill | 短请求绕过、长请求保护、防饥饿限制、chunk 轮转及 decode 公平性 | 自动用例通过；未完成真实 HTTP 到达率负载的服务 SLO 评估 |
| 退出调度 | 原 gate、延迟 gate、可重复随机 lookahead、trace，最小/最大深度、队列路由 | 控制流测试通过；随机 gate/复用 early gate 的质量不等价于蒸馏 lookahead |
| Async pipeline | CPU prepare 在前一步 GPU 未结束时完成；GPU token 直接送 prelude；有界在途、DMA 银行复用 | 依赖/安全性用例通过；人为延长 GPU 的测试只证明可重叠，不证明性能 |
| EOS、取消、ID 复用 | 延迟 EOS、下一 token 在途、取消后复用、旧结果隔离、KV/state 回收 | 回归用例通过 |
| Static/padding | 虚拟行不写 KV、不输出；异步 workspace 生命周期；prefix restore | 自动用例通过；性能无收益，见第 5 节 |
| Attention 数值 | Triton/reference；FA4 FP16/BF16 MHA/GQA、非连续 layer views、非顺序页表、零长度行 | B300 实测通过；FA2/FA3 的架构选择有测试，其他 GPU 未实测 |
| 大 KV 地址 | int32 页号乘 stride 超过 2^31 元素的回归 | 旧 Triton 越界已复现并修复为乘法前 int64；修复后整套 249 passed / 11 skipped |
| 服务协议 | HTTP/streaming/错误处理等已有自动服务用例；真实模型 CLI 单请求 | 完整回归中的服务用例通过；新 FA4 尚无真实 HTTP 压测结果 |
| 模型任务精度 | 历史 GSM8K-87 固定回归集 | 历史 HF/native 均 59/87；不是当前 async/FA4 新成绩，当前 11 个 lm_eval 相关用例跳过 |

真实 ragged 重放：12 配置，每配置 8 请求、prompt 长度 1–8、16 输出，逐 token 深度 1–4，同 dtype/layout 内 S/AS/AM 一致。此前 BF16 多请求多流的历史分歧没有单独原样重现并关闭；新重放和后续扫描未发现差异，不能推广为任意 prompt/阈值都没有问题。

### 审计纠正：static 前置 GPU 检查

原 `static-ablation/gpu-tests.log` 实际为 **6 skipped**：驱动漏传 `--run-gpu`。此前将其描述为“前置 GPU 检查通过”不准确，现明确更正。
81 次实际 GPU 性能测量及 token/depth 对比仍然有效，不能与该 pytest 跳过记录混为一谈。后续完整回归已开启 GPU；本次汇总又单独使用 `--run-gpu -k cuda -vv` 补验，结果为 **6 passed、6 deselected**，见 `gpu-tests-audit.log`。保留原日志，不覆盖历史证据。

## 3. 修复前后短上下文与 profiling

128 输入、512 输出、并发 8 的独立等工作量诊断：

| 实现 | recurrent 总行数 | core 调用数 | 平均 core batch | tok/s |
| --- | ---: | ---: | ---: | ---: |
| 同步诊断 | 32704 | 4095 | 7.986 | 251.73 |
| 旧 AS | 32704 | 5075 | 6.444 | 205.90 |
| 旧 AM | 32704 | 5115 | 6.394 | 204.95 |
| 临时 wait-coda 消融 | 32704 | 4095 | 7.986 | 250.59 |

这支持旧实现的退化来自 batch 碎片化。正式修复采用 GPU token 反馈与有界流水执行，不是永久等待所有 coda。修复后 C8 的 core 调用恢复为约 4095–4097、平均 batch 约 7.98。

修复后的三次中位数：

| 并发 | S tok/s | AS tok/s | AM tok/s |
| --- | ---: | ---: | ---: |
| 1 | 33.01 | 32.83 | 32.69 |
| 4 | 128.03 | 126.11 | 126.59 |
| 8 | 255.12 | 255.47 | 254.17 |

27 次正式测量、234 请求、119808 输出 token，一致。这里证明短负载 batching 恢复、性能接近同步；并非稳定提速。

已保存修复前和修复后的 Torch profiler，均为 CPU/CUDA ops + shapes，无 stack、memory 或额外自定义 marker。早期 24 调度步的 trace 没有捕获长期碎片化，独立计数才复现；不能把不同工作量 trace 的总时间直接相除。新 FA4 和长上下文混合退出尚未进行专门的根因 profiling。

## 4. Triton 上下文与并发扫描：全部完成

同上下文真实 prefill 复用、独立 KV 页，decode-only，128 输出，预算/chunk=2048，static/padding/graphs 关闭。每组合三次重复。64K 名义上下文实际 prompt=65409，预留输出后最终 KV=65536，未越过模型位置上限。

| Context | Concurrency | Policy | S tok/s | AS tok/s | AM tok/s | AS / S | AM / S |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1024 | 1 | fixed4 | 36.45 | 36.03 | 36.29 | 0.989x | 0.996x |
| 1024 | 8 | fixed4 | 276.50 | 279.55 | 274.07 | 1.011x | 0.991x |
| 1024 | 32 | fixed4 | 869.65 | 931.34 | 927.56 | 1.071x | 1.067x |
| 1024 | 64 | fixed4 | 1462.65 | 1577.92 | 1579.51 | 1.079x | 1.080x |
| 1024 | 128 | fixed4 | 1804.28 | 2040.15 | 2038.02 | 1.131x | 1.130x |
| 1024 | 256 | fixed4 | 2173.27 | 2396.25 | 2397.42 | 1.103x | 1.103x |
| 1024 | 256 | mixed234 | 2517.21 | 2813.52 | 2780.65 | 1.118x | 1.105x |
| 4096 | 1 | fixed4 | 19.08 | 19.41 | 19.60 | 1.018x | 1.028x |
| 4096 | 4 | fixed4 | 73.04 | 74.93 | 74.85 | 1.026x | 1.025x |
| 4096 | 16 | fixed4 | 244.41 | 255.65 | 255.65 | 1.046x | 1.046x |
| 4096 | 32 | fixed4 | 397.31 | 417.53 | 417.04 | 1.051x | 1.050x |
| 4096 | 64 | fixed4 | 605.82 | 658.27 | 657.90 | 1.087x | 1.086x |
| 4096 | 64 | mixed234 | 691.19 | 732.75 | 669.04 | 1.060x | 0.968x |
| 16384 | 1 | fixed4 | 6.25 | 6.31 | 6.32 | 1.010x | 1.011x |
| 16384 | 2 | fixed4 | 12.31 | 12.45 | 12.47 | 1.011x | 1.013x |
| 16384 | 4 | fixed4 | 24.50 | 24.80 | 24.80 | 1.012x | 1.012x |
| 16384 | 8 | fixed4 | 48.57 | 49.38 | 49.39 | 1.017x | 1.017x |
| 16384 | 16 | fixed4 | 81.71 | 83.51 | 83.56 | 1.022x | 1.023x |
| 16384 | 16 | mixed234 | 85.52 | 87.10 | 75.77 | 1.018x | 0.886x |
| 65536 | 1 | fixed4 | 1.70 | 1.71 | 1.71 | 1.004x | 1.004x |
| 65536 | 2 | fixed4 | 3.38 | 3.39 | 3.39 | 1.004x | 1.004x |
| 65536 | 4 | fixed4 | 6.72 | 6.77 | 6.77 | 1.007x | 1.007x |
| 65536 | 4 | mixed234 | 6.75 | 6.78 | 5.78 | 1.005x | 0.856x |

207 次测量、8856 请求、1133568 输出 token；全部进程退出 0，输出无分歧。
固定四轮的 AS/AM 收益在 1K 中高并发最明显（约 7%–13%）；16K/64K 较小。混合退出时 AM 相对 S：4K −3.2%、16K −11.4%、64K −14.4%。这些新负载仍需定位，不能归为已经修复的旧 bug。
最初的大 KV 越界失败记录保留在扫描根目录，不计入这 207 次；接受结果仅来自 `round2/`。

## 5. Static/padding：全部完成

1K 上下文、Triton、每组合三次重复。static 只复用缓冲区；static-pad 再按 2 的幂补齐。没有 CUDA graphs。

| Concurrency | Policy | Buffers | S tok/s | AS tok/s | AM tok/s |
| ---: | --- | --- | ---: | ---: | ---: |
| 32 | fixed4 | dynamic | 875.05 | 931.54 | 928.34 |
| 32 | fixed4 | static | 842.80 | 918.44 | 921.75 |
| 32 | fixed4 | static-pad | 848.58 | 917.96 | 923.53 |
| 128 | fixed4 | dynamic | 1886.45 | 2039.67 | 2042.27 |
| 128 | fixed4 | static | 1748.39 | 2010.97 | 2011.21 |
| 128 | fixed4 | static-pad | 1747.29 | 2010.38 | 2009.23 |
| 128 | mixed234 | dynamic | 2079.60 | 2281.51 | 2142.22 |
| 128 | mixed234 | static | 1943.87 | 2248.16 | 2126.09 |
| 128 | mixed234 | static-pad | 1927.92 | 2228.86 | 2102.04 |

81 次测量、7776 请求、995328 输出 token，跨配置输出一致。相对各自 dynamic 基线，已测 static/static-pad 吞吐均下降，约 0.5%–7.4%；因此目前没有测试证据支持为提速默认打开它们。这不是 FA4 下的 static 性能消融。

## 6. Triton / FA4 A/B：全部完成

同卡交错、同一真实 Triton prefill 起点，只切换 decode attention；FA4 4.0.0b30，BF16，page=16，输出128，预算/chunk=2048，关闭 static/padding/graphs。以下百分比按同一调度模式的 FA4/Triton 中位数计算。

| 上下文 | 并发 | 退出 | S | AS | AM |
| --- | ---: | --- | ---: | ---: | ---: |
| 1K | 1 | fixed4 | -17.9% | -17.5% | -17.2% |
| 1K | 32 | fixed4 | +0.4% | -6.4% | -7.0% |
| 1K | 128 | fixed4 | +41.7% | +28.2% | +30.5% |
| 1K | 128 | mixed234 | +32.3% | +19.1% | +25.4% |
| 4K | 1 | fixed4 | +42.0% | +38.3% | +37.7% |
| 4K | 32 | fixed4 | +96.0% | +90.4% | +88.8% |
| 4K | 64 | fixed4 | +73.7% | +81.0% | +81.5% |
| 4K | 64 | mixed234 | +75.8% | +81.0% | +82.8% |

144 次测量、8100 请求、1036800 输出 token，同后端内与跨后端均一致。
4K/C64 固定四轮时，FA4 的 S/AS/AM 为 1082.58/1191.55/1195.11 tok/s，异步比 FA4 同步约快 10%。混合退出时则为 1216.16/1326.15/1223.14：AM 没有超过 AS。
4K/C32 的 AS p95 token 间隔从 Triton 的 77.90 ms 降到 FA4 的 40.78 ms。完整吞吐和 p95 表见 [A/B 报告](flash_attention_ab.md)。没有使用这轮短测 smoke 的数字代替正式三次中位数。

## 7. 尚未验证或未实现

| 项目 | 当前状态 |
| --- | --- |
| CUDA graphs | 未实现；按用户要求暂不做 |
| 论文 Fig.3 的 idle 40%→11.1%→0.67% | 未复现、未按跨流时间线统计；不能用吞吐替代 |
| 蒸馏 lookahead 权重与其精度 | 没有训练权重；随机权重和复用 early gate 均非蒸馏模型 |
| 当前代码/FA4/async 的 GSM8K 精度 | 尚未重测；历史 59/87 仅为固定回归集记录，且不是独立随机留出评估 |
| FA4 16K/64K 性能 | 尚未测；现有 16K/64K 数据属于 Triton |
| FA4 static/padding 性能 | 只有功能 smoke；81 次 static 消融是 Triton |
| 优化的 packed prefill / 端到端 TTFT | 本轮 attention A/B 不覆盖；当前适配器对每个可见前缀按单 query 表示 |
| 真实 HTTP 到达率、长短请求混合、SLO、长期压力 | 尚未完成系统性性能测试 |
| SHARED 性能 | 有功能一致性验证，没有系统性能基线 |
| 跨 GPU | 当前实际硬件为 B300；H100/A100/RTX 等不能标为已验证 |
| AM 在长上下文混合退出的退化 | 已观测，尚未进行新的因果定位 |

## 8. 日志和报告索引

所有路径位于 `/home/zjy/code/david/tmp/`，未把大日志或 trace 写入 Git。

| 目录 | 内容 |
| --- | --- |
| `ouro-baseline-20260915T043856Z/` | 历史 48 次 A/B/C/D，原始输出、环境/源码快照 |
| `ouro-dynamic-baseline-20260915T065600Z/` | 历史 36 次动态缓冲扫描、遥测、回归日志 |
| `ouro-ops-shapes-20260915T072138Z/` | 修复前 ops_shapes_traces.zip、batch counter、wait-coda 消融 |
| `ouro-async-pipeline-20260915/` | 正式 pipeline 回归；仅 `verified-c1/c4/c8` 为接受的 27 次性能结果；ragged-real.json；修复后的 ops-shapes-traces.zip |
| `ouro-context-sweep-20260915/round2/` | 接受的 207 次扫描、实时汇总、源码哈希、四进程退出码 |
| `ouro-context-sweep-20260915/static-ablation/` | 81 次 static 消融；原 gpu-tests.log 是 skipped；补验 gpu-tests-audit.log |
| `ouro-flash-attn-20260915/` | full-tests.log、real-smoke、cli-auto-kv.log、FA4 接入验证 |
| `ouro-flash-attn-20260915/ab/` | 144 次配对 A/B；results-summary.md、analysis.json、源快照、逐 trial 输出、telemetry.csv、exitcodes.json |

未将 warmup、短 smoke、失败轮次、被否决调度策略、profiler 窗口混入正式吞吐统计。未把重复的完整回归快照相加作为独立测试覆盖数量。
