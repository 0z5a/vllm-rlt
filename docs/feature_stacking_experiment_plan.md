# 特性叠加补测方案：1K–64K 与完整延迟指标

状态：2026-09-16 按用户要求停止剩余测试，537/576次完整测量已保留。见[已完成叠加结果与分析](feature_stacking_results_20260916.md)，不再继续运行本矩阵。已有数据见
[叠加结果](feature_stacking_results_20260915.md) 和
[全部测试汇总](testing_summary_20260915.md)。

## 叠加链

| 阶段 | Attention | 退出机制 | 调度 | Static | Padding |
| --- | --- | --- | --- | --- | --- |
| P0 | Triton | 固定4轮 | S | 关 | 关 |
| P1 | FA4 | 固定4轮 | S | 关 | 关 |
| P2 | FA4 | 原始 Ouro gate | S | 关 | 关 |
| P3 | FA4 | ouro_delayed，复用原 gate | S | 关 | 关 |
| P4 | FA4 | 同P3 | AS | 关 | 关 |
| P5 | FA4 | 同P3 | AM | 关 | 关 |
| P6 | FA4 | 同P3 | AM | 开 | 关 |
| P7 | FA4 | 同P3 | AM | 开 | 开 |

P2→P3 是退出语义变化，必须单列其代价。min_loops=2 时原 gate 最早2轮退出，
当前 delayed 机制最早3轮退出；不能把 P2→P4 全部记为 async 收益。
CUDA graphs 均关闭且暂不实现。混合234不替代真实 gate 阈值实验。

## 基本参数

| 参数 | 设置 |
| --- | --- |
| 模型与硬件 | 固定版本 Ouro-1.4B，B300，同上下文各阶段在同卡交错运行 |
| dtype / KV布局 / block_size | BF16 / LAST-EXITED / 16 |
| max_num_batched_tokens / prefill_chunk_size | 2048 / 2048 |
| max_num_seqs | 等于该测试点的并发 |
| 输出 | 固定128 tokens，greedy，ignore_eos=True |
| max_loops / min_loops | 4 / 2 |
| 阈值 | fixed4=1.0；真实早退主测0.2，补测0.5、0.7 |
| 预热与重复 | 每配置完整预热；三次正式重复 |
| KV池 | 每种上下文按最大并发预留同一容量，阶段间保持一致 |
| Prompt | 冻结内容不同的真实请求及token IDs；不能用同一句话重复代替真实退出分布 |

| 名义上下文 | 输入token数 | 并发点 | 最大并发KV估计 |
| --- | ---: | --- | ---: |
| 1K | 1024 | 1、32、128 | 108 GiB |
| 4K | 4096 | 1、16、64 | 198 GiB |
| 16K | 16384 | 1、4、16 | 193.5 GiB |
| 64K | 65409 | 1、2、4 | 192 GiB |

64K为输出预留位置，最终KV不超过模型65536上限。KV估计不包括模型及临时显存。
2048限制新调度token数，不限制每个query读取的历史KV长度。
真实prompt的选择/长度处理规则与哈希必须在正式运行前冻结。

## 两种计时实验，不能混成一份数字

1. **Decode-only配对实验**：恢复同一份真实prefill状态，比较退出和执行方式。
   记录decode输出吞吐、ITL、窗口耗时、轮数和batch统计。
   没有请求排队和prefill计时，TTFT与完整请求E2E必须标为N/A。
   原12个负载点×8阶段×3重复=288次正式测量仅指这一部分。
2. **端到端引擎实验**：每个请求从提交前开始计时，实际经过排队、KV分配、
   prefill、decode和输出交付，不使用prefill恢复或跨请求复制来跳过输入计算。
   所有阶段消费相同的请求序列，以固定并发closed-loop方式补入请求，完成固定请求数
   后排空；比较配置内固定样本数、顺序和并发，不采用“跑60秒各自完成多少算多少”。
   测试前明确冻结每组请求数/波次数；每请求记录实际排队时间及完整token时间戳。
   若完整镜像上述矩阵，将另增加288次正式测量，合计576次，预热和阈值补测另计；
   64K实际prefill成本很高，先做可行性计时后再公布总耗时，不把新增工作隐藏在原288次内。

引擎计时不含HTTP网络、客户端、JSON/SSE和文本tokenization。tokenization在提交前
完成并冻结输入。若另做HTTP测试，结果必须标为客户端口径，不能和引擎口径混在一列。
端到端补入请求会让不同配置有不同的到达时刻，这是closed-loop负载的定义；
要测试固定到达率与SLO，需要另一个open-loop实验。

## TTFT、TPOT、ITL、E2E定义

每个请求保存：提交时间t0、首token交付t1、逐token交付t2…tN、完成通知tf。
全部使用同一主机单调时钟；不为采集时间戳逐token添加CUDA同步。

| 指标 | 计算 | 含义 / 单位 |
| --- | --- | --- |
| TTFT | t1−t0 | 首token等待，包含引擎排队、prefill、首token计算/交付；ms |
| TPOT | (tN−t1)/(N−1) | 每请求首token之后的平均每token时间；ms/token，N<2为N/A |
| ITL_i | ti−t(i−1)，i=2…N | 相邻token的交付间隔；ms |
| E2E | tf−t0 | 从提交到请求完成通知的完整延迟；s |
| 输出吞吐 | 全部输出token数/整个测量窗口 | tok/s，包括端到端试验的prefill与排空开销 |
| 请求吞吐 | 完成请求数/整个测量窗口 | requests/s |

TPOT反映请求内平均速度，ITL分布反映生成过程的停顿。总体tok/s不等于1/TPOT，
因为多个请求并发执行。E2E在完成通知与末token同次交付时等于TTFT+(N−1)×TPOT；
若完成通知更晚，需再加tf−tN，原始日志保留该差值。

TTFT、TPOT、E2E先逐请求计算再求p50/p95/p99；ITL在单次trial内汇总所有token间隔
计算分位数，并保留逐请求间隔。展示三次trial各自分位值的中位数与范围，不能称为
合并所有请求后的分位数。同时报告请求样本数、间隔样本数、失败/取消数和完成率。
样本不足时尾部分位数明确标注样本少，不把p99当成稳定SLO结论；每请求TPOT的p95
不能代替所有token间隔的ITL p95。

当前引擎以step返回输出，若同一批输出共用主机观测时间，记录该真实交付粒度，
不人工平均分摊为虚构的逐token时间。加载和预热排除计时，额外失败/超时要留存，
不能通过删除慢请求美化结果。

## 性能大表的目标列

| Context | 并发 | 阶段/特性 | Gate/阈值 | 平均decode轮数 | tok/s | 单步/累计收益 | TTFT p50/p95 ms | TPOT p50/p95 ms/token | ITL p50/p95 ms | E2E p50/p95 s | 峰值显存 | 请求数/完成率 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 待测 | 待测 | P0–P7逐行 | 固定明确 | 不含prefill首输出 | 分计时口径 | 分母明确 | 端到端实测 | 端到端实测 | 两口径分开 | 端到端实测 | allocated/reserved | 不省略失败 |

主表展示p50/p95，p99及三次原始值放附表/JSON。另保存2/3/4轮占比、recurrent总行数、
调用次数、平均batch、CPU prepare期间GPU在途计数，以及逐请求token/深度差异。
P2相对P1回答early exit增益，P3相对P2回答延迟策略代价，P4相对P3回答async增益。
不同退出策略允许输出变化，不据此判错或宣称质量一致；同策略的S/AS/AM做一致性检查。

## 实现与解释边界

现有context_sweep仅支持固定4轮和受控mixed234，并假定同一prompt复制prefill。
补测需要支持真实gate、不同prompt的独立prefill checkpoint以及新的逐请求记录，
不能仅改一行阈值就沿用旧断言。已有runtime_baseline包含TTFT/TPOT/ITL/E2E公式，
但它的固定4轮断言和现有负载设置也需要适配。

论文§6.1/附录B.2性能实验采用预先记录的真实退出轨迹再重放。
可在在线gate主实验后增加同轨迹重放，分离gate等待和调度成本，并测no-refill/refill；
重放结果不冒充在线gate性能。蒸馏权重、GSM8K质量、跨卡与HTTP SLO仍属独立验证项。

## 本轮执行与回溯

入口为 `benchmarks/feature_stack.py`，测量器专项验证为 CPU/GPU 共 **3 passed**。
1K、并发3、输出8的预检：decode与端到端各完成8阶段。
端到端P7有1个请求的1个token退出深度由4变3，生成token未变化；原因待定位，
同策略一致性不能直接记为通过。正式结果保留token IDs与退出深度，差异不隐藏。

正式矩阵为288次decode + 288次端到端，共576次测量；每次另有完整预热。
端到端请求数冻结为 **2×并发**，完成后补入下一请求，decode为并发个请求。
每个阶段每次重复消费相同的输入序列。阈值0.5/0.7不含在本轮576次中。

输入来自THUDM/LongBench-v2，固定revision
`2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9`，seed=20260915，
保留问题与选项，对上下文做首尾截断并冻结token IDs和SHA256。
这是长文本问答性能负载，不是LongBench准确率评测。

运行目录：`/home/zjy/code/david/tmp/ouro-feature-stack-20260915/`。

| 文件 | 用途 |
| --- | --- |
| `run_matrix.py`、`driver.log` | 完整矩阵驱动及启动/退出记录 |
| `formal/status.json` | 当前完成数、进程PID、各任务退出码 |
| `formal/results.md` | 持续更新的大表；n=3才完成该行全部重复 |
| `formal/all_results.json` | 全部trial，含p99、显存、轮数、batch、差异 |
| `formal/*/manifest.json` | 每个worker的实际参数、KV池大小、环境及负载哈希 |
| `formal/*.command.json`、`formal/*.log` | 实际执行命令与重定向日志 |
| `formal/*/*.requests.json` | 逐请求提交、token交付、完成时间及输出/退出深度 |
| `formal/*/*.failure.json` | 异常与已采集请求记录（发生失败时） |
| `formal/source/`、`formal/source_hashes.json` | 本轮实际运行的代码快照及哈希 |
| `formal/gpu-telemetry.csv` | 每20秒GPU利用率、显存和功率 |

GPU0–3分别跑1K/4K/16K/64K decode；GPU4–7分别跑相同上下文的端到端。
GPU6待64K可行性预检释放后启动。各worker限制4个CPU亲和核，Torch/BLAS线程数1。
单上下文的全部阶段在同GPU运行，阶段顺序按固定种子交错，避免固定顺序偏置。

单项命令示例（输出目录必须不存在）：

```bash
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zjy/code/david/b_workspace/.b_rdma/bin/python -m benchmarks.feature_stack \
  --model /home/zjy/code/david/b_workspace/models/Ouro-1.4B \
  --workload /home/zjy/code/david/tmp/ouro-feature-stack-20260915/workloads/ctx1024.json \
  --scope e2e --concurrencies 1 32 128 --output-length 128 \
  --batch-tokens 2048 --threshold 0.2 --waves 2 --repeats 3 \
  --output /home/zjy/code/david/tmp/ouro-feature-stack-20260915/example-e2e \
  > /home/zjy/code/david/tmp/ouro-feature-stack-20260915/example-e2e.log 2>&1
```

从仓库根目录执行；`--scope decode`切换到前缀恢复计时。
64K端到端需要反复计算真实prefill，预计显著慢于短上下文，不能以恢复KV替代TTFT测量。
