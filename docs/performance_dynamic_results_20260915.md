# S / AS / AM 动态缓冲性能结果（2026-09-15）

> 历史测试：本文的严重退化对应旧 async 实现。修复后短负载接近同步，但长上下文混合退出仍有新的退化，统一见 [全部测试汇总](testing_summary_20260915.md)。

## 结论

统一关闭static buffers和padding后，async退化仍存在：并发4的AM比S慢约11.3%，
并发8慢约18.9%。并发1/2基本接近。关闭static没有消除退化，但本批未profile，
不能据此把原因确定为某个同步点或内核。

## 生效配置

| 组别 | 调度 | CUDA流 | static_buffers | padding |
|---|---|---|---|---|
| S | 同步 | 单流 | false | false |
| AS | 异步 | 单流 | false | false |
| AM | 异步 | 多流 | false | false |

本批先删除了引擎在async初始化时隐式强制static=true的行为。
测量器断言实际execution_config等于请求配置、Workspace为空、持久states池为None，
并将实际配置写入summary。async安全回传所需pinned readback缓冲仍预分配，
它独立于static_buffers开关；“关闭static”不表示取消所有异步回传缓冲。

其余条件与首批相同：真实Ouro-1.4B、BF16、Triton、LAST_EXITED、固定4轮，
128输入/512输出、greedy、ignore_eos、8条循环prompt、KV1280blocks，
max_num_seqs=8、max_num_batched_tokens=128。每组预热一次满并发完整生成，
每次补入至少60秒后排空，3次重复，共36次。GPU和CPU亲和性与首批相同。
基于commit e525f64，含本批未提交的引擎和测量器改动，源码快照随日志保存。

## 吞吐

单位输出tokens/s，三次中位数；变化按同批中位数之比计算。

| 并发 | S | AS | AM | AS相对S | AM相对S |
|---|---|---|---|---|---|
| 1 | 32.84 | 32.79 | 33.00 | -0.13% | +0.49% |
| 2 | 64.30 | 63.19 | 63.86 | -1.73% | -0.69% |
| 4 | 129.01 | 113.63 | 114.44 | -11.92% | -11.30% |
| 8 | 255.83 | 205.79 | 207.52 | -19.56% | -18.88% |

## 波动、延迟和样本量

范围为三次min–max；延迟是每次分位数的中位数。记录的是引擎主机观察时间，
不是HTTP延迟或GPU纯计算时间。显存峰值为PyTorch allocated，包含模型和固定KV池。

| 并发 | 组别 | tokens/s范围 | TTFT P95 ms | TPOT P50 ms/token | E2E P95 s | 峰值Allocated GiB | 三次请求总数 |
|---|---|---|---|---|---|---|---|
| 1 | S | 32.83–33.10 | 33.48 | 30.28 | 15.64 | 6.65 | 12 |
| 1 | AS | 32.59–33.08 | 33.58 | 30.48 | 15.63 | 6.65 | 12 |
| 1 | AM | 32.87–33.06 | 33.37 | 30.29 | 15.55 | 6.62 | 12 |
| 2 | S | 63.27–64.37 | 75.19 | 31.08 | 15.96 | 6.65 | 24 |
| 2 | AS | 62.62–63.86 | 74.93 | 31.46 | 16.37 | 6.65 | 24 |
| 2 | AM | 63.59–63.92 | 74.92 | 31.13 | 16.13 | 6.62 | 24 |
| 4 | S | 128.58–129.51 | 154.50 | 30.95 | 16.04 | 6.65 | 48 |
| 4 | AS | 110.18–124.39 | 153.00 | 32.34 | 19.63 | 6.65 | 48 |
| 4 | AM | 114.03–117.91 | 151.34 | 32.28 | 19.56 | 6.62 | 48 |
| 8 | S | 255.58–255.98 | 271.96 | 31.23 | 16.25 | 6.65 | 96 |
| 8 | AS | 186.78–208.59 | 277.64 | 38.81 | 20.05 | 6.65 | 93 |
| 8 | AM | 206.82–208.51 | 269.38 | 38.41 | 19.94 | 6.62 | 96 |

## 验证与限制

- 4个worker退出码均为0，36个trial完整；共537请求、274944输出token。
- 原始请求均512输出、每token退出深度4、finish_reason=length；时间戳与计数检查通过。
  相同并发、相同prompt对比S，全部token IDs一致。此前BF16可变退出分歧仍未关闭。
- 新增8组动态缓冲GPU用例，回归先通过63项runtime测试，随后全套223 passed、
  11 skipped（缺可选lm_eval），退出码0，原始日志见regression.log。
- c1每trial仅4请求，不能用其P95/P99推断稳定尾延迟。AS/c8第二次仅29请求，
  其他c8 trial为32请求；采样同一输入流但消费前缀不完全相同，吞吐包含排空。
  因此不同请求进度、不均衡和排空开销也可能影响结果，需要固定请求数及稳态窗口复测。
- c4 AS范围110.18–124.39，c8 AS范围186.78–208.59，波动明显；不能只报最好一次。
- 全程GPU遥测已保存，时钟没有锁定；四张独立GPU同时运行不同并发点，仍共享主机资源。
  c1完成后在GPU1及独立CPU40-43运行了回归，遥测后段GPU1活动包含该回归，不能混作性能trial。
- 这是固定深度引擎负载，不包含HTTP、可变深度trace、其他输入长度或SHARED语义。
  与首批static开启结果不是同一时间的随机顺序消融，跨批次差异只能作为线索。

## 下一步定位依据

并发8的S TPOT约31.23ms/token，AS约38.81、AM约38.41；相较首token延迟的变化，
主要损失仍在持续生成阶段。并发4则有明显请求总延迟和重复波动，需检查实际batch
组成、每阶段调用次数、提交间隙、动态metadata搬运及最后排空行为。尚未采集profile，
不把这些假设当成已定位根因。

## 原始文件

目录：`/home/zjy/code/david/tmp/ouro-dynamic-baseline-20260915T065600Z`。

- `c1/`、`c2/`、`c4/`、`c8/`：manifest、每trial的summary及完整requests.jsonl。
- `c*.json`、`c*.log`、`c*.exitcode`：命令/环境、日志与真实进程退出码。
- `gpu_telemetry.csv`：5秒采样的GPU利用率、显存、功率、时钟、温度；索引顺序见monitor.py。
- `summary.csv`、`verified_results.json`、`summarize.py`：汇总、逐请求核验和可重复统计脚本。
- `benchmark_source.py`、`engine_source.py`、`git_diff.log`、`packages.log`：源码与环境快照。
- `regression.log`、`regression.exitcode`：完整回归结果。

复现参数见 [performance_baseline.md](performance_baseline.md) 第8节。

后续[ops/shape采集与因果诊断](profiling_ops_shapes_20260915.md)已定位本负载的主要退化：
coda回填时机造成core batch碎片化；等待coda的诊断消融恢复吞吐，尚未改为正式策略。
