# Ouro 功能约束下的性能基线计划

目的：在相同计算量和正确性约束下，衡量静态缓冲、异步流水线与多流的收益。
功能检查及通用 `run_logged` 日志函数见 [functional_validation.md](functional_validation.md)。
本文件是后续逐项执行计划，不表示已经完成真实模型性能测量。

## 1. 前置条件与对照组

先完成CPU/GPU功能检查；真实BF16多请求多流token差异尚未解决，可记录探索性性能，
但不能作为“正确性已经验证的优化结果”。FP32也仅有有限工作负载的一致性证据。
分别报告BF16/FP32，禁止跨精度计算优化收益。

固定真实Ouro权重、tokenizer、LAST_EXITED、采样设置、KV block数量、调度模式、
max_num_seqs及max_num_batched_tokens。第一轮各组统一使用
`--exit-mode ouro_delayed`，请求中 `exit_threshold=1.0`、`max_loops=4`。

| 组别 | 执行参数 | 比较目的 |
|---|---|---|
| A | 同步，无static/padding | 原生引擎基础开销 |
| B | 同步，`--static-buffers --pad-to-power-of-two` | A→B衡量缓冲与padding组合影响 |
| C | B加 `--async-scheduling --single-stream` | B→C衡量流水线影响 |
| D | B加 `--async-scheduling` | C→D衡量多流影响 |

如果需区分static与padding，另增B0（只有static）作为消融。不能只比较A与D后把
所有收益归因于async。自动内存规划可用于准备容量，但正式对照固定相同num_blocks，
记录容量和是否发生排队；它影响请求接纳。不要默认512 blocks能支持长输入/高并发。

第二轮使用同一份可变退出深度trace，所有组固定每请求每token深度，覆盖早/晚退出。
trace要满足首token完整prefill深度及后续min/max限制，并固定请求ID。对比普通Ouro
和ouro_delayed会改变计算深度，不能仅归因于调度。随机Lookahead没有蒸馏训练权重，
其退出分布不能代表真实训练后负载。

## 2. 工作负载矩阵

| 类型 | 输入/输出token示例 | 观察点 |
|---|---|---|
| Decode为主 | 128/512 | 循环执行、异步调度 |
| Prefill为主 | 2048/128 | prefill与chunking |
| 中等长度 | 512/256 | 基础混合行为 |
| 长短混合 | 固定比例混合以上长度 | 接纳、公平性、尾延迟 |

先测decode负载×并发1/2/4/8×A/B/C/D，再扩展输入长度和混合负载。
输入按实际tokenizer计数；保存相同prompt token IDs，不能用字符数代替。
引擎测试可输入固定token序列；HTTP必须核对服务端实际token计数。

固定输出长度时使用 `ignore_eos=true`；这仅用于计算量可比的性能实验，质量测试
保留自然EOS。记录所有组实际输出数，防止提前结束造成虚假加速。
先用闭环固定并发：一条完成后补一条；记录预生成请求集合和固定顺序。
调度改变完成顺序可能影响补入时序，应保留每请求时间线。之后再做相同到达时间表的
开环QPS测试，检查排队与饱和；两种负载结果分开报告。

## 3. 指标定义

| 指标 | 定义/记录方式 |
|---|---|
| TTFT | 客户端发送至首个含输出内容事件，报告P50/P95/P99；若使用引擎首token定义须另标注 |
| TPOT | `(最后token时间-首token时间)/(输出token数-1)`；不足2token不参与该指标 |
| ITL | 相邻输出token/事件时间差；事件可能含多个token时只能称事件间隔，不冒充token间隔 |
| 请求总延迟 | 发送到完成，报告P50/P95/P99 |
| 输出tokens/s | 完成工作量的输出token总数/公共测量窗口耗时，不能平均各请求tokens/s |
| 请求/s | 同一窗口的完成请求数/耗时 |
| 退出深度 | 分布、平均值及每请求序列，确认计算量可比 |
| 资源/可靠性 | 峰值allocated/reserved显存、GPU状态、成功/失败/超时数量 |

报告窗口口径：第一批测量请求发出到最后一批完成，预热单独完成；包含排队和排空。
如果另测稳态窗口，必须说明跨边界请求和token如何计数。样本过少不宣称P99稳定。
TTFT的首个可见文本可能晚于首个token（特殊token、UTF-8拼接）；引擎和HTTP不要混用定义。
`stream=false`仅提供完整响应耗时，不能测TTFT/ITL。网络包不是token或SSE事件。

## 4. 引擎与服务分开测

**引擎层**：计时前完成加载、tokenization和预热；输入相同token IDs。
CUDA在计时区间前后同步，**不要每步同步**，否则破坏async重叠。覆盖完整生成流程，
保存token IDs、深度和配置；正式计时中避免逐步CPU拷贝logits/打印。

**HTTP层**：独立客户端记录请求开始、流式事件、结束、usage、HTTP错误和超时。
包含tokenization、排队、序列化和网络。客户端不能成为瓶颈；先同机loopback，
远程测量另报网络条件。并发由压测客户端产生，max_num_seqs只是服务端上限。

现有 `benchmarks.cdb_runtime` 只支持tiny模型：可做8组trace回放功能检查，不能
通过添加一个不存在的 `--model` 参数变成真实模型性能工具。真实权重引擎测量器现已新增为 `benchmarks.runtime_baseline`（见下文）；
带时间戳的HTTP并发客户端仍需后续实现/核验，不能用普通curl替代。
可参考现有 [serving.md](serving.md) 的压测入口，但必须先核对工具版本、实际支持
的额外请求参数和事件计时方式，再纳入统一基线。

## 5. 每组执行与日志管理

1. 记录commit、dirty diff、新增未跟踪文件、权重哈希、依赖、GPU型号/驱动、
   CUDA可见设备、CPU线程、精度、TF32/归约设置、退出策略、KV容量、请求集合哈希。
2. 空闲GPU运行；保留测试前后nvidia-smi，确认无其他任务争抢。客户端也固定资源。
3. 用代表性batch/长度充分预热；排除模型加载、自动KV规划、首次编译和预热时间。
4. 每组至少持续几十秒，例如60秒以上，同时确保样本量足够；固定完成请求集合时
   先选足够规模。记录实际持续时间与样本数，不强行截断未完成请求后忽略失败。
5. 每组至少3次，交替/打乱A/B/C/D运行顺序，报告中位数、范围，不能只选最快一次。
6. profiler另开一轮，检查GPU空隙、CPU等待及事件依赖；不把profile耗时当正式结果。

使用功能文档的RUN_DIR与run_logged；每个case/重复独立子目录，例如：

```text
<run>/perf/decode_128_512/c4/D/repeat1/
  command.txt
  config.json
  environment.txt
  server.log
  client.log
  requests.jsonl
  results.jsonl
  summary.json
  gpu_before.txt
  gpu_after.txt
  exitcode.txt
```

原始数据至少包含request_id、输入/实际输出长度、开始/首输出/结束时间、退出深度
（引擎层）、错误、精度、模式、重复编号；汇总表链接原始日志。不同服务器PID/启动
配置分开保存。服务日志可以重定向，正式测量避免开启逐token调试打印。

以下是**日志模板**，待真实测量器实现后替换命令，不是现在可运行的benchmark：

```text
run_logged perf_<workload>_<concurrency>_<variant>_<repeat> <verified benchmark command>
```

## 6. 首批实验与验收表

首批：真实Ouro、LAST_EXITED、固定4轮、128输入/512输出，
并发1/2/4/8 × A/B/C/D × 至少3次。先建立单请求执行开销，再找并发收益拐点。
完成后加入固定trace的可变深度负载；最后扩展prefill和混合请求。
SHARED属于不同KV语义，另列实验，不能未经质量验证直接作为LAST_EXITED的等价加速。

| workload | dtype | 并发 | 组别 | 实际请求/token数 | 输出tokens/s | TTFT P95 | TPOT P95 | E2E P95 | 峰值显存 | 错误数 | 正确性状态 | 日志 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 待执行 | | | | | | | | | | | | |

报告A→B、B→C、C→D的相对吞吐变化和尾延迟变化，保留绝对值和重复间波动。
无吞吐提升、延迟恶化或差异小于波动时如实记录；功能失败的配置不得标记为可接受优化。
历史tiny测量没有证明吞吐提升，真实模型性能和质量结论仍待执行。


## 7. 已实现的真实权重引擎测量入口

`benchmarks.runtime_baseline` 直接加载本地真实模型，以循环的8条确定性prompt
构成闭环请求流。默认每条输入128token、输出512token、固定4轮、LAST_EXITED、
BF16、greedy、ignore_eos；所有组固定1280 blocks、max_num_seqs=8、batch预算128。
每种配置先执行一轮满并发的完整512token生成进行预热；计时开始/结束同步CUDA，
计时区间不逐步同步。每次先持续补入至少60秒，然后停止补入并排空；结果分母包含排空。
因此不同组完成请求数可能不同，输入流相同但消费前缀/混合比例可能不同，不能视为
完全相同有限请求集合。需要固定数量时使用 `--requests N` 覆盖持续时间。

Bash（先按功能文档激活环境并设置RUN_DIR和GPU_ID）：

```bash
run_logged engine_c1 env CUDA_VISIBLE_DEVICES="$GPU_ID" OMP_NUM_THREADS=1 \
  python -u -m benchmarks.runtime_baseline \
  --model "$MODEL_DIR" --concurrency 1 --seconds 60 --repeats 3 \
  --input-length 128 --output-length 512 --dtype bfloat16 --num-blocks 1280 \
  --output "$RUN_DIR/engine_c1"
```

将concurrency改为2/4/8，使用不同output目录。output目录必须不存在，避免覆盖结果。
默认每次重复用固定随机种子改变A/B/C/D执行顺序。可用 `--variants A B` 选择子集，
用 `--dtype float32` 做独立精度配置，用 `--device cpu --toy` 校验测量器。
不支持把新参数当作HF/vLLM外部引擎对照；此入口只对比当前vllm-lt内部四组。

每目录包含manifest.json（环境/输入token/配置/设备/代码状态）、results.json、
每组每次的summary.json和requests.jsonl。原始请求记录token IDs、exit_depths、
主机观察的逐token时间；其采集开销属于本次引擎测量，所有组一致保留。
记录每prompt相对本进程“首次观察输出”的分歧数，只用于提示，首次观察不一定是A，
不作为正确性判定。固定4轮、输出长度、EOS忽略及回收计数有显式断言。

TTFT/ITL取engine.step返回结果时刻，同一step中的输出用相同时刻，不能据此推断
GPU内核完成时间或HTTP时间。mean-depth=4只说明计算深度一致，不保证BF16文本一致。
低并发60秒样本可能较少，P95/P99仅描述这些样本，不代表稳定尾延迟估计。
显存包括模型及固定KV池，reported peak是PyTorch allocated/reserved，不等同于nvidia-smi。

本轮使用4张同型号空闲B300，各并发点固定在同一张卡上依次对比A/B/C/D，并绑定
互不重叠的CPU亲和性。A/B/C/D组内比较有效性高于跨卡并发缩放比较；跨卡硬件差异、
同机共享资源及GPU时钟未锁定均属于限制，实际设备状态留存在日志中。

首批48组实测已完成，见 [2026-09-15结果报告](performance_results_20260915.md)。


## 8. S / AS / AM：统一关闭static和padding

按用户要求新增第二批对照。引擎不再在async初始化时隐式把static_buffers设为True。
三组实际生效配置在测量器中断言，并写入summary；动态路径没有Workspace和持久
request-state池。注意：async的pinned readback存储仍预分配并受事件保护，它不受
static_buffers选项控制。这批关闭的是输入/metadata/hidden-state静态池和padding，
并非移除所有异步结果回传缓冲。

| 组别 | async_scheduling | multi_stream | static_buffers | pad_to_power_of_two |
|---|---|---|---|---|
| S | false | 不适用 | false | false |
| AS | true | false | false | false |
| AM | true | true | false | false |

```bash
run_logged dynamic_c8 env CUDA_VISIBLE_DEVICES="$GPU_ID" OMP_NUM_THREADS=1 \
  python -u -m benchmarks.runtime_baseline \
  --model "$MODEL_DIR" --concurrency 8 --seconds 60 --repeats 3 \
  --variants S AS AM --input-length 128 --output-length 512 \
  --dtype bfloat16 --num-blocks 1280 --output "$RUN_DIR/dynamic_c8"
```

并发1/2/4/8分别测3组×3次，共36次；其余负载、KV容量与首批一致。
GPU/CPU亲和性映射与首批相同，但不同批次时间不同；优先比较本批S/AS/AM，
跨批次对比static效果仅作为线索，不能替代同一次随机顺序消融。
动态metadata的host-to-device搬运可能阻塞提交；若因此无法重叠，也属于动态路径
实际表现，不能把“打开async开关”等同于已经实现运行时重叠。

第二批36组S/AS/AM实测已完成，见[动态缓冲结果报告](performance_dynamic_results_20260915.md)。

## Completed async pipeline baseline

The updated implementation, accepted S/AS/AM measurements, batching counters,
and reproduction commands are in [async scheduling](async_scheduling.md).
Static device buffers, padding and CUDA graphs remain disabled in these runs.
