# Torch profiler：ops / shapes 与 async 退化定位

> 历史诊断：本文保留采集当时的结果及 wait-coda 消融；正式 async pipeline 后来已实现，不采用永久等待全部 coda。最新验证见 [全部测试汇总](testing_summary_20260915.md)。

## 本次采集设置

仅CPU/CUDA活动和record_shapes=True；with_stack=False、profile_memory=False、
with_flops=False，不增加自定义阶段标记。ProfilerStep是PyTorch自动标记。
三组分别S、AS、AM，static和padding均关闭；同一GPU0、CPU8-11、BF16真实Ouro，
128输入、512输出上限、8并发、LAST_EXITED、固定4轮、greedy、ignore_eos。

先做8请求完整512token预热，再开始新的8请求；所有请求至少生成96token后启动profiler，
再经过4个profiler warmup调度步，采集24个active调度步。捕获窗口结束后取消剩余请求，
因此profile metadata不是完整512token测量结果。trace timing只用于诊断，不代替性能基线。

## 文件怎么读

每组目录包含：

- `trace.json`：原始Chrome trace，使用Perfetto的Open trace file打开；观察CPU算子、CUDA运行时、GPU流。
- `cpu_ops_shapes.txt`：按Self CPU排序的前100个算子/输入shape组合。
- `cuda_ops_shapes.txt`：按Self CUDA排序的前100个组合。
- `ops_shapes.json`：全部组合、调用数、CPU/CUDA self和total时间（微秒），不截断。
- `cuda_api_summary.json`：从trace派生的CUDA API次数及CPU耗时，值为[count, duration_us]。
- `metadata.json`：实际配置、窗口前后输出进度、trace大小。

`aten::linear`和内部`aten::mm`可能嵌套，CPU/CUDA total不能跨所有行直接相加。
同一kernel及其父算子的device时间也可能重复归属；比较同一算子同一shape的每次开销。
S/AS/AM相同24个调度步不代表相同core次数，不能拿总CUDA耗时直接比较速度。

## 纯ops / shape窗口看到了什么

| 模式 | active窗口core次数（由gate调用数核对） | 主要mm shape | mm次数 | 每次mm CUDA时间 |
|---|---|---|---|---|
| S | 8 | [8,2048] × [2048,2048] | 768 | 6.754 us |
| AS | 12 | [8,2048] × [2048,2048] | 1152 | 6.738 us |
| AM | 12 | [8,2048] × [2048,2048] | 1152 | 6.727 us |

该窗口未显示主要矩阵乘法单次变慢，core batch都为8。AM确有两个kernel流：
主要core流16668个kernel，boundary流48个kernel；这只证明使用了两个流，
不能由数量直接推断有有效重叠或加速。

## 为什么增加不带profiler的独立核对

本地torch.profiler文档明确指出：record_shapes=True临时持有Tensor引用，
可能影响依赖引用计数的优化，并带来额外开销。观察时间本身也影响异步ready轮询。
因此在原始轻量trace之外，独立跑了只对stage/batch做Counter累加的诊断。
它不向trace增加标记，不增加CUDA同步，保留用户要求的原始ops/shape采集。

三组使用完全相同的16请求、每请求512输出，总8192输出token；均先预热8请求。
这是一次等工作量诊断，不是替代此前3次正式基线。统计所有recurrent调用，
不包含prefill内部循环；总recurrent计算量均为16×511×4=32704行。

| 配置 | recurrent行数 | core调用数 | 平均core batch | 输出tokens/s |
|---|---|---|---|---|
| S | 32704 | 4095 | 7.986 | 251.73 |
| AS | 32704 | 5075 | 6.444 | 205.90 |
| AM | 32704 | 5115 | 6.394 | 204.95 |
| AS wait-coda diagnostic | 32704 | 4095 | 7.986 | 250.59 |

AS/AM比S多约24%/25%的core调用，平均batch从接近8降至约6.4。
AS的batch分布包含1063次4行、1477次6行；不是GPU矩阵乘法突然变慢，
而是相同总行数被拆成了更多次小批计算。该诊断复现了约18%的吞吐退化。

## 代码路径与因果消融

`engine/llm_engine.py` 的 `_step_async()` 先调用 `_collect_coda(wait=False)`
（refill模式），只收取ready的ticket，然后立即调用 `scheduler.schedule()`。
尚未ready的coda留在_pending_coda，对应请求不能进入下一token的prelude。
`core/scheduler.py` 的 `_take()` 只打包当前stage队列里已有的请求，没有考虑
即将回来的pending coda请求，于是可能先提交不满的core batch。

同步路径在execute拿到coda结果后立即_update，再开始下一次调度，回填时机不同。
例如8个活跃请求中2个仍在pending coda、6个在recurrent队列，异步可能直接提交6行，
而不是等那2个经过prelude后合批。这个例子解释代码机制，实际分布见batch计数文件。

为了检验该路径是否能解释退化，诊断脚本只在收取coda时等待所有已提交coda的event，
其余保留AS、动态缓冲、原始gate、相同16请求。结果：core调用恢复到4095、
平均batch恢复到7.986，吞吐250.59，接近同步251.73（低约0.45%）。
AS、AM及消融版的16个请求token IDs和退出深度全部与S一致。

**结论：这组固定深度负载的主要退化已定位到coda结果回填时机造成的batch碎片化。**
纯ops/shape窗口没有复现小batch，说明其观测扰动/窗口选择不能忽略；不能单靠
这份trace宣称完整程序没有碎片化。Counter诊断复现退化，针对coda回收的消融恢复
batch和吞吐，共同支持上述判断。单次诊断不排除其他工作负载还存在额外瓶颈。

## 不是正式修复

等待全部coda的实验仅存在于日志目录诊断脚本中，没有修改正式引擎策略。
直接永久加入全量等待会压缩CPU/GPU重叠机会，在可变深度或不同长度负载下可能不合适。
后续正式方案需要在等待收益、core批次填充和尾延迟之间设计有界合批策略，并重新验证
S/AS/AM、变深度trace以及此前BF16多流一致性问题。当前不宣称已经完成性能优化。

## 原始目录与复现

原始目录：`/home/zjy/code/david/tmp/ouro-ops-shapes-20260915T072138Z`。

- `ops_shapes_traces.zip`：约10MB，包含三组原始trace与表格（解压后约140MB）。
- `capture/S/`、`capture/AS/`、`capture/AM/`：三组采集。
- `count_batches.py`、`*_measured_batch_counts.json`、`unprofiled_counts/`：无profiler统计。
- `ablate_wait_coda.py`、`ablation_wait_coda/`：消融脚本、原始输出和统计。
- `capture.log`、`exitcode`、`count_batches.log`、`count_batches.exitcode`、
  `ablation_wait_coda.log`、`ablation_wait_coda.exitcode`：采集和诊断均退出码0。
- `diagnosis_summary.json`、`profile_comparison.json`：汇总。

激活.b_rdma后，采集入口为：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 taskset -c 8-11 \
  python -u -m benchmarks.profile_ops \
  --model /home/zjy/code/david/b_workspace/models/Ouro-1.4B \
  --manifest /home/zjy/code/david/tmp/ouro-dynamic-baseline-20260915T065600Z/c8/manifest.json \
  --output /新的空目录/capture --variants S AS AM --active-steps 24 \
  > /已经存在的日志目录/capture.log 2>&1
```

选择可用GPU，output目录必须不存在；stdout/stderr日志目录需提前创建。
