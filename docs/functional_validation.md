# 功能验证与日志留存

> 状态更新：下文主要是逐项验收方案及历史记录，不能把用例存在当作通过。最新完整回归、真实模型验证、审计更正和未覆盖项见 [全部测试汇总](testing_summary_20260915.md)。

本方案验证实现行为，不把“请求能返回”视为模型精度对齐。性能计划见
[performance_baseline.md](performance_baseline.md)。所有 GPU 操作先用
`nvidia-smi` 选择可用卡；不需要 `gpu` 预约命令。以下命令由使用者手动执行，
本次文档更新不代表重新运行了这些实验。

## 0. 逐项功能验收表：测什么、怎么看、怎样算通过

下面每行是一项独立验收，不能用“服务能返回文本”代替。**本轮状态均为待执行**；
“已有自动测试”只表示有检查入口，不表示你这次运行已经通过。
测试路径简写：`C` = `tests/test_cdb_runtime.py`，`E` = `tests/test_engine.py`，
`K` = `tests/test_kv_cache.py`，`O` = `tests/test_ouro.py`，
`A` = `tests/test_attention.py`，`S` = `tests/test_serving.py`。
表中 `C::test_x` 要展开为 `tests/test_cdb_runtime.py::test_x` 后执行。

### 模型、prefill 与 KV

| 编号 / 功能 | 怎样构造验证 | 要看什么 / 通过标准 | 自动测试入口与覆盖边界 |
|---|---|---|---|
| F01 权重加载 | 保存tiny模型再加载；另外加载本地真实Ouro | 保存/加载参数对应一致；缺失或形状错误的权重要拒绝；真实服务health=200 | `O::test_native_safetensors_roundtrip`、`O::test_checkpoint_mismatch_is_rejected`；真实权重另走第4节，不代表模型质量合格 |
| F02 全深度prefill | packed prompt与dense参考逐深度比较 | 每轮结果满足测试数值容差；不因chunking遗漏prompt；首输出记录完整prefill深度 | `O::test_packed_prefill_matches_dense_at_every_depth`、`E::test_chunked_packed_generation_matches_serial`；tiny数值对照 |
| F03 因果attention | 同一请求多个位置packed执行，并混合不同请求/深度 | 当前token不能看到未来token，也不能看到其他请求KV；attention与dense参考一致 | `A::test_packed_prefill_is_causal_for_repeated_request_ids`、`A::test_mixed_depth_and_request_batch_matches_dense` |
| F04 LAST_EXITED补齐 | 两层KV写入已执行深度，提前退出后finalize | 已执行深度保留自身KV；未执行深度复制最后执行深度；修改源不会连带修改目标 | `K::test_last_exited_copies_every_layer_and_preserves_executed_depths`；检查值而非只看block数量 |
| F05 Ragged KV及跨block | 连续token分别执行不同深度，并跨越block边界 | 后一个深层token读到前面token对应的最后退出KV；attention与构造的期望值一致 | `K::test_early_exit_then_later_token_loops_deeper_across_block_boundary` |
| F06 SHARED物理平面 | 多个逻辑深度写同一位置，随后回收 | 同一物理平面按轮覆盖；各逻辑深度读到共享值；free计数恢复且不重复释放 | `C::test_shared_overwrites_one_plane_and_reclaims_it_once`；不能要求SHARED与LAST_EXITED输出相同 |
| F07 SHARED chunk一致性 | 同一KV布局下比较串行与chunked/padded批处理 | token IDs、退出深度保持一致；不同chunk边界不改变定义的prefill语义 | `C::test_padded_chunked_execution_matches_serial`；检查参数ID中的shared分支 |
| F08 分配原子性与回收隔离 | 故意容量不足，再释放并复用block | 失败分配不占用部分KV；重复free不破坏计数；新请求不读到旧历史 | `K::test_admission_is_atomic_and_counts_all_depths`、`K::test_block_reuse_never_exposes_stale_request_history` |
| F09 KV容量规划 | CPU字节预算/默认容量；GPU实际自动规划 | CPU容量与预算计算相符；GPU memory_plan来源为cuda_profile、profile_peak_bytes>0，block数不超max_useful_blocks，分配可执行 | `C::test_explicit_bytes_and_cpu_auto_sizing`；**GPU** `C::test_cuda_auto_memory_plan_and_abort_pending_work`；不是对外部进程抢显存的保证 |

### 请求调度与退出决策

| 编号 / 功能 | 怎样构造验证 | 要看什么 / 通过标准 | 自动测试入口与覆盖边界 |
|---|---|---|---|
| F10 Admission绕过与公平性 | 池24块，运行请求占8块；队首长请求要20块，后面短请求要4块；绕过上限1 | 短请求先进入prefill；long仍等待；达到上限后later不能继续绕过；资源释放后long进入prefill并最终完成 | `C::test_short_requests_bypass_then_protected_long_request_gets_memory`；看stage、WAITING顺序和最后完成集合 |
| F11 Chunked prefill轮转 | 长prompt8token、短prompt1token，chunk=1 | 首批两个请求各推进1token；短请求先完成，长请求继续推进；不是一次吞掉整条长prompt | `C::test_chunk_limit_allows_short_prompt_before_long_prefill_finishes` |
| F12 Decode防饥饿 | 已有decode期间持续加入短prompt | 已有decode仍持续获得执行机会并完成，不因新增prefill一直等待 | `E::test_repeated_short_arrivals_cannot_starve_existing_decode`；覆盖refill/no_refill |
| F13 Refill/no_refill区别 | fast提前退出、slow继续循环 | refill中fast下一token在slow当前token完成前回到core；no_refill按阶段处理；最终请求都完成 | `E::test_refill_recycles_exited_token_before_slow_token_coda`；看recurrent队列trace，不只看最终文本 |
| F14 原始Ouro累计gate | 每轮hazard固定0.4，阈值0.6、min_loops=2 | 第一轮累计0.4，第二轮0.64，decode第2轮退出；首输出来自完整prefill，测试深度序列[4,2,2] | `E::test_cumulative_gate_and_minimum_depth` |
| F15 禁止提前退出与硬边界 | threshold=1，即使sigmoid接近1；分别设置min/max | threshold=1时不自适应提前退出；不超过max；各模式的min规则符合定义 | `E::test_threshold_one_never_exits_early_even_if_sigmoid_rounds_to_one`、`C::test_delayed_bounds_and_threshold_one`、F17边界参数 |
| F16 随机Lookahead | 相同seed建head，改变seed再建；脚本控制第2轮发出退出信号 | 相同seed权重相同、不同seed不同，不改原模型权重/RNG；第2轮信号在第3轮退出，用第3轮最终状态生成 | `C::test_random_head_is_independent_reproducible_and_does_not_change_base_weights`、`C::test_signal_applies_after_one_more_loop_and_final_hidden_is_used`；只有随机权重，无蒸馏权重，不验质量 |
| F17 复用Ouro gate延迟退出 | 原gate每轮hazard=0.3，阈值0.5、min=2；生成多个token | 第2轮累计0.51、第3轮退出；无随机辅助head；每token重新累计；最终token IDs与固定3轮参考一致，首输出全深度 | `C::test_ouro_delayed_reuses_gate_and_accumulates_hazards`；含sync/async、min/max、threshold=1；不是原始Ouro第2轮输出等价性 |
| F18 Trace replay | 给每请求指定逐token深度，并使gate调用报错 | 实际深度严格等于trace，gate不执行；非法trace在admission前拒绝；8组tiny回放token/深度一致 | `C::test_trace_replay_skips_gate_and_obeys_variable_depths`、`C::test_trace_validation_before_admission`、第3节回放命令 |

### 异步执行、服务与真实模型

| 编号 / 功能 | 怎样构造验证 | 要看什么 / 通过标准 | 自动测试入口与覆盖边界 |
|---|---|---|---|
| F19 Submit/collect顺序 | 记录core提交和上一轮信号收集顺序 | 先提交r+1，再收集r；不能提前阻塞GPU下一轮；延迟策略用r信号判断r+1后退出 | `C::test_async_submits_next_core_before_collecting_previous_signal`；CPU状态机检查，不证明GPU重叠 |
| F20 CUDA同步/异步一致性 | tiny GPU模型，两个KV布局，两种lookahead模式，padding及采样 | 同一配置下token IDs和exit_depths一致，结束后KV归零 | **GPU** `C::test_cuda_async_matches_synchronous_with_padding_and_sampling`；不覆盖所有真实模型数值边界 |
| F21 CUDA多流确实重叠 | 预热后人为延长core，CUDA Event记录core/coda区间 | 两区间存在交集：max(start)<min(end)；只完成而没有交集不能算通过 | **GPU** `C::test_cuda_boundary_and_core_can_overlap`；这是受控依赖验证，不证明实际吞吐收益 |
| F22 Padding不污染KV | 真实3行pad到4行，空闲KV预填17/19哨兵 | submitted_rows=4，但空闲key/value仍是17/19；dummy不参与写KV/输出 | `C::test_padding_does_not_touch_free_blocks`，结合F20验证GPU数值路径 |
| F23 Pending coda与回收 | 保持一个coda未就绪；取消后复用相同request_id；注入提交失败 | 别的请求core能前进；不把旧结果交给新请求；失败后KV和state slot回收 | `C::test_pending_coda_does_not_block_other_recurrent_work`、`C::test_cancel_pending_coda_and_reuse_request_id`、`C::test_async_failed_submission_reclaims_all_affected_state`；GPU回收另看F09 |
| F24 HTTP普通/流式输出 | 相同prompt走直接引擎、HTTP非流式及流式 | 测试中输出对应；流式token事件和终止标志正确；usage、EOS符合契约 | `S::test_http_output_matches_direct_and_stream_has_exact_token_events`、`S::test_eos_usage_and_non_usage_stream`；真实服务另走第4节 |
| F25 HTTP错误与隔离 | 无效请求、队列过载、客户端断开或慢读 | 拒绝不合法请求且不污染引擎；正常请求仍可运行；断开后资源清理 | `S::test_http_invalid_admission_and_browser_requests_preserve_readiness`、`S::test_owner_thread_dynamic_batching_cancellation_and_overload`、`S::test_request_failure_and_disconnect_cleanup`、`S::test_slow_channel_does_not_block_other_requests` |
| F26 真实模型sync/async对照 | 相同token IDs、dtype、KV、退出策略与采样；分别单请求/多请求、单流/多流 | 同时保存并比对输出token IDs、逐token退出深度和完成状态；分歧定位到首个不同token，不能仅比较自然语言大意 | **人工/API待补批量工具**。FP32有限用例历史一致；BF16多请求多流已有未解分歧，不得标为完全通过 |
| F27 模型任务精度 | 固定GSM8K协议、相同依赖与HF参考 | 完整样本评分及每题结果按accuracy.md门槛验收；依赖缺失/样本未完成属于未验证 | [accuracy.md](accuracy.md)；历史分数不是本次结果，随机Lookahead无质量基线，新async覆盖需另确认 |

### 每行怎么运行、怎么看日志

先完成第1节环境与 `run_logged` 定义。用 `-vv` 保留参数化用例名称，
`--tb=short` 保留失败断言，`-ra` 显示跳过原因。下面示范F10与F20：

```bash
run_logged F10_admission python -m pytest -vv -ra --tb=short \
  tests/test_cdb_runtime.py::test_short_requests_bypass_then_protected_long_request_gets_memory \
  --junitxml="$RUN_DIR/F10_admission.xml"

run_logged F20_cuda_equivalence env CUDA_VISIBLE_DEVICES="$GPU_ID" \
  python -m pytest -vv -ra --tb=short --run-gpu \
  tests/test_cdb_runtime.py::test_cuda_async_matches_synchronous_with_padding_and_sampling \
  --junitxml="$RUN_DIR/F20_cuda_equivalence.xml"
```

其他行替换node ID并更换日志标签/XML文件名，一行多个入口可以放在同一条pytest
命令中。按表逐项运行后，全套再补齐未单独列出的参数校验等用例。

1. 打开对应 `.command` 核对配置与GPU；打开 `.exitcode`，必须为0。
2. 打开 `.log`，确认该node确实被collected并显示PASSED；GPU项若是SKIPPED，
   记为“未验证”，不能当通过。退出码5/没有收集到测试也不能算通过。
3. 参数化测试逐个看：如F20应有两种layout×两种exit mode，不只检查第一条。
4. 表中“要看什么”由测试断言检查；默认日志不打印所有成功中间值。要人工追踪
   时打开对应测试查看fixture/assert；失败日志保存实际/期望值。**不要把不存在的
   KV、深度字段想象成服务端已经输出的日志字段。**
5. 若需新增真实请求trace，单独保存position、stage、loops_done、pending_exit_depth、
   remaining_probability、KV计数；这是建议的诊断数据，当前没有统一CLI开关。
   GPU async的loops_done是主机已提交进度，完成要以Event为准。诊断拷贝可能改变
   调度时序，所以不能把带诊断打印的运行直接作为性能基线。

验收记录模板（每个编号分别填，不能把F01~F27合成一个“正常”）：

| 编号 | commit / 配置 | 测试用例实际执行数 | 结果：通过/失败/跳过/待执行 | 观察到的事实 | 日志/XML路径 | 未覆盖边界或下一步 |
|---|---|---|---|---|---|---|
| F10 | 待填 | 待填 | 待执行 | 短请求绕过、long保护与最终admission | 待填 | 不等于生产负载公平性证明 |
| F20 | 待填 | 待填 | 待执行 | 两布局×两退出模式tiny GPU一致性 | 待填 | 真实BF16另记F26 |
| F26 | 待填 | 待填 | 待执行；已有已知问题 | 需保存首个token分歧及对应深度 | 待填 | BF16多流根因未解 |

## 1. 环境与日志目录

为避免 fish/bash 重定向和变量语法不同，**如果当前使用 fish，先执行 `bash`**。
后续所有命令均为 Bash。在其他终端运行时也先进入 Bash 并激活同一环境。

```bash
bash
```

```bash
cd /home/zjy/code/david/b_workspace/vllm-lt
source /home/zjy/code/david/b_workspace/.b_rdma/bin/activate
export OMP_NUM_THREADS=1
export MODEL_DIR=/home/zjy/code/david/b_workspace/models/Ouro-1.4B
export RUN_DIR=/home/zjy/code/david/tmp/vllm-lt-validation/$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$RUN_DIR"
printf 'Logs: %s\n' "$RUN_DIR"
nvidia-smi
# 根据上一步结果修改；不要占用仍运行服务或其他任务的卡。
export GPU_ID=5

run_logged() {
    local label="$1"
    shift
    local stamp
    stamp=$(date -u +%Y%m%dT%H%M%S%N)
    local prefix="$RUN_DIR/${label}_${stamp}"
    printf '%q ' "$@" > "${prefix}.command"
    printf '\n' >> "${prefix}.command"
    printf 'Running %s; log: %s.log\n' "$label" "$prefix"
    "$@" > "${prefix}.log" 2>&1
    local rc=$?
    printf '%s\n' "$rc" > "${prefix}.exitcode"
    tail -n 25 "${prefix}.log"
    printf 'Exit code: %s\n' "$rc"
    return "$rc"
}

run_logged git_status git status --short
run_logged git_commit git log -1 --format=fuller
run_logged tracked_changes git diff HEAD
run_logged packages python -m pip freeze
run_logged gpu nvidia-smi
run_logged model_hash sha256sum "$MODEL_DIR/model.safetensors"
```

每项命令保存 `.command`、`.log`、`.exitcode`；非零退出码表示失败，不能因
日志中部分用例通过就继续认定整项成功。pytest 另外保存 JUnit XML。重跑测试
时 XML 用同一名称会覆盖，所以需保留旧报告或新建 RUN_DIR；文本日志带时间戳不会覆盖。
不要把权重、原始测试日志或大型结果加入 Git。

另一个终端可以用 `tail -f /实际路径/某项.log` 跟踪测试。下面服务测试使用独立日志。

## 2. 分项回归命令

按表格顺序执行；发现失败先保存日志，不要直接跳过。CPU 标记筛选不会验证 CUDA
异步执行，GPU 项必须实际执行 `--run-gpu`。测试框架仍保留旧的“reservation”提示，
当前只需手动设置可用卡的 `CUDA_VISIBLE_DEVICES`。

| 项目 | 检查内容 | 通过标准 |
|---|---|---|
| 静态检查 | Ruff、格式、CPU pre-commit | 所有 hook 成功 |
| 模型与 KV | packed prefill、dense 对照、不同深度、保存加载 | 数值对照和状态断言通过 |
| 调度 | admission bypass、公平性、chunked prefill | 短请求可推进，受保护请求不继续被绕过 |
| 退出 | 原始/随机/延迟 gate、累计概率、边界、trace | 输出深度和最终隐状态符合断言 |
| 生命周期 | cancellation、ID reuse、异常清理、pending coda | 无陈旧结果，KV 和 state slot 正确回收 |
| 服务 | HTTP 协议、流式输出和错误处理 | 自动化服务用例通过 |
| GPU | Triton attention、两种 KV、padding、多流事件 | GPU 用例实际运行并通过 |

```bash
run_logged precommit python -m pre_commit run --all-files

run_logged model_kv python -m pytest -q -ra -m 'not gpu' \
  tests/test_ouro.py tests/test_kv_cache.py tests/test_prepared_kv.py tests/test_attention.py \
  --junitxml="$RUN_DIR/model_kv.xml"

run_logged scheduler python -m pytest -q -ra -m 'not gpu' \
  tests/test_engine.py tests/test_cdb_runtime.py \
  -k 'bypass or chunk or prefill or scheduler' \
  --junitxml="$RUN_DIR/scheduler.xml"

run_logged exits python -m pytest -q -ra -m 'not gpu' \
  tests/test_cdb_runtime.py -k 'signal or delayed or random_head or trace' \
  --junitxml="$RUN_DIR/exits.xml"

run_logged lifecycle python -m pytest -q -ra -m 'not gpu' \
  tests/test_cdb_runtime.py -k 'cancel or pending or failed_submission' \
  --junitxml="$RUN_DIR/lifecycle.xml"

run_logged serving python -m pytest -q -ra tests/test_serving.py \
  --junitxml="$RUN_DIR/serving.xml"

run_logged gpu_checks env CUDA_VISIBLE_DEVICES="$GPU_ID" \
  python -m pytest -q -ra --run-gpu -m gpu \
  tests/test_attention.py tests/test_cdb_runtime.py \
  --junitxml="$RUN_DIR/gpu.xml"
```

分项适合定位问题；最终验收用全套覆盖未被分项选择的用例：

```bash
run_logged full_suite env CUDA_VISIBLE_DEVICES="$GPU_ID" \
  python -m pytest -q -ra --run-gpu --junitxml="$RUN_DIR/full_suite.xml"
```

已有记录为 **215 passed、11 skipped**，11 项依赖未安装的可选 `lm_eval`。
这是历史结果，不应硬编码为未来用例数；每次核对 `-ra` 的实际 skip 原因。
`lm_eval` 被跳过意味着没有完成对应评估，不是精度测试通过。

## 3. tiny 模型 trace 功能一致性

```bash
run_logged replay_last_exited env CUDA_VISIBLE_DEVICES="$GPU_ID" \
  python -m benchmarks.cdb_runtime --device cuda --layout last_exited \
  --trace-out "$RUN_DIR/last_exited_trace.json" \
  --output "$RUN_DIR/last_exited_results.json"

run_logged replay_shared env CUDA_VISIBLE_DEVICES="$GPU_ID" \
  python -m benchmarks.cdb_runtime --device cuda --layout shared \
  --trace-out "$RUN_DIR/shared_trace.json" \
  --output "$RUN_DIR/shared_results.json"

run_logged replay_saved env CUDA_VISIBLE_DEVICES="$GPU_ID" \
  python -m benchmarks.cdb_runtime --device cuda --layout last_exited \
  --trace-in "$RUN_DIR/last_exited_trace.json" \
  --output "$RUN_DIR/last_exited_replay.json"
```

每次比较 refill/no_refill × sync_eager/sync_padded/async_single/async_multi 共8组，
必须保持 token IDs 和退出深度一致；不一致会抛错。trace 包含模型、精度、KV布局、
输入和采样指纹，不能跨布局/精度混用。这里是 tiny 模型功能回归，其 tokens/s
不是 Ouro-1.4B 性能基线，也不代表论文加速比。

## 4. 真实权重 HTTP 冒烟测试

在上面已初始化的终端启动服务，日志重定向到文件。默认示例用端口8001，
避免与已有8000服务冲突；选择可用 GPU。前台运行，结束时 Ctrl+C，勿批量杀进程。

```bash
run_logged server env CUDA_VISIBLE_DEVICES="$GPU_ID" \
  python -u -m vllm_lt.entrypoints.serve \
  --model "$MODEL_DIR" --served-model-name ouro \
  --device cuda --dtype float32 --attention-backend triton \
  --exit-mode ouro_delayed --kv-layout last_exited \
  --max-num-seqs 8 --num-blocks 512 \
  --host 127.0.0.1 --port 8001
```

`512` 个block是这个短请求示例的固定预算，不是所有工作负载的容量建议。
服务初始化失败时查看server日志；不要把连接失败当作模型输出失败。

在第二个 Bash 终端激活环境，将 RUN_DIR 指向第一终端打印的同一个目录：

```bash
source /home/zjy/code/david/b_workspace/.b_rdma/bin/activate
export RUN_DIR=/第一终端打印的实际日志目录
export CASE=fixed4
mkdir -p "$RUN_DIR/$CASE"
curl --fail-with-body -sS --max-time 10 -D "$RUN_DIR/$CASE/health.headers" \
  http://127.0.0.1:8001/health > "$RUN_DIR/$CASE/health.json" \
  2> "$RUN_DIR/$CASE/health.stderr"
printf '%s\n' "$?" > "$RUN_DIR/$CASE/health.exitcode"
```

HTTP200后请求。将请求原文、响应、HTTP头和客户端错误分开保存，不只保留中文正文：

```bash
cat > "$RUN_DIR/$CASE/request.json" <<'JSON'
{
  "model": "ouro",
  "prompt": "请解释一下什么是矩阵乘法。",
  "max_tokens": 128,
  "temperature": 0,
  "min_loops": 2,
  "max_loops": 4,
  "exit_threshold": 1.0,
  "stream": false
}
JSON

curl --fail-with-body -sS --max-time 600 \
  -D "$RUN_DIR/$CASE/response.headers" \
  http://127.0.0.1:8001/v1/completions \
  -H 'Content-Type: application/json' \
  --data-binary @"$RUN_DIR/$CASE/request.json" \
  > "$RUN_DIR/$CASE/response.json" 2> "$RUN_DIR/$CASE/curl.stderr"
printf '%s\n' "$?" > "$RUN_DIR/$CASE/curl.exitcode"

python - "$RUN_DIR/$CASE/response.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    result = json.load(f)
assert 'error' not in result, result
assert result['choices'], result
print(result['choices'][0]['text'])
print('finish_reason:', result['choices'][0]['finish_reason'])
print('usage:', result['usage'])
PY
```

逐组换 CASE 名称，保存独立文件，并修改相应请求/服务参数：

| CASE | 服务改动 | 请求改动 | 目的 |
|---|---|---|---|
| fixed4 | 上述同步配置 | threshold=1 | 固定4轮运行 |
| ouro_sync | `--exit-mode ouro` | threshold=0.9 | 原始累计 gate |
| delayed_sync | `--exit-mode ouro_delayed --static-buffers --pad-to-power-of-two` | threshold=0.9 | 延迟 gate 同步参考 |
| delayed_async_single | 与delayed_sync相同，加 `--async-scheduling --single-stream` | 与delayed_sync相同 | 单流异步 |
| delayed_async_multi | 与delayed_sync相同，加 `--async-scheduling` | 与delayed_sync相同 | 多流异步 |
| random_async | `--exit-mode random_lookahead --lookahead-seed 7 --async-scheduling` | threshold=0.5 | 仅测试未训练Lookahead链路 |

不同退出模式可能合法地产生不同文本；不能把它们当同一个输出对照。
HTTP输出不包含每个token的退出深度，不能仅用此响应验证深度。精确对照应走
engine/LLM API，保存 `token_ids`、`exit_depths`，固定输入与配置，再比较同步/异步。
已有tiny测试自动执行这些断言；面向真实权重的批量对照工具仍需补齐。

再用 `stream=true` 单独保存流式响应，检查正常终止和错误。它是协议冒烟检查，
不能根据curl输出估算准确的TTFT/ITL；性能客户端需记录每个事件的到达时间。

## 5. 正确性结论的边界

- 原始权重、tokenizer和完整精度评估：参考 [accuracy.md](accuracy.md) 的固定
  GSM8K协议。其历史59/87不是本分支新测结果；当前环境未装lm_eval，且包版本
  与历史基线不同，不能直接宣称沿用该分数。准备依赖后核对fingerprint，必要时
  在同一环境重测HF参考。现有GSM8K runner也不能默认视为覆盖新async模式。
- 随机Lookahead只有可重复随机权重，没有蒸馏训练权重；不用于模型质量结论。
  ouro_delayed复用训练好的原始gate，但延迟决策会改变退出深度，不是训练后的预测器。
- 已有真实模型检查：FP32三请求各16token，sync/async结果及深度一致；BF16中文
  单请求也一致。BF16多请求多流曾有token分歧，深度一致，根因尚未确认。
  此项必须保持“未解决”，不能由tiny测试通过推导为没有问题。
- 多流重叠测试用人为延长core验证事件依赖，只能证明该受控用例存在重叠。

验收记录至少填写：commit/dirty状态、配置、GPU、测试命令、退出码、原始日志、
通过/失败/跳过原因、复现步骤；不要只记录“正常”。

补充：首批性能运行同时核对了固定4轮真实模型的token/深度一致性，见
[结果报告](performance_results_20260915.md)。测量器保留原始记录，但不覆盖
任意可变退出策略的完整正确性矩阵，F26已知问题仍未关闭。


### 动态缓冲async补充验收

新增 `tests/test_cdb_runtime.py::test_cuda_dynamic_async_respects_buffers_and_matches_sync`：
两种KV布局×单/多流×两种退出模式，共8个GPU参数组合，断言实际execution_config
保持static=false、Workspace为空、states=None，并核对sync/async输出及退出深度。
受控CUDA重叠测试明确使用static=true，不能用其通过推断动态缓冲路径也有同样重叠。

## 完整 async pipeline 的新增验证

实现细节、论文 §4.3 / Fig.3 对应及日志位置见
[async scheduling](async_scheduling.md)。以下测试位于 `tests/test_async_pipeline.py`。

| 功能 | 怎么验 | 要看什么 |
| --- | --- | --- |
| CPU prepare 与 GPU forward 重叠 | `test_cpu_prepare_finishes_while_previous_gpu_forward_is_inflight` 人为延长前一次 GPU forward | CPU KV 准备完成时，前一次 GPU event 仍未完成；AM 的 H2D 也能先完成。这是依赖拓扑测试，不是提速测试。 |
| GPU token 直接喂给 prelude | `test_prelude_consumes_device_sample_before_cpu_delivery` 暂停 coda 的 CPU 交付 | CPU token 列表仍为空，prelude/core 已执行；embedding 收到的 tensor 与 GPU sample 相同。 |
| 延迟结果不读错 depth/position | `test_delayed_delivery_uses_snapshot_depths_and_bounds_placeholders` 回放 1–4 层不同深度并强制延迟交付 | 每次输出和同步完全对应；深度来自该次 coda 的快照，不能取下一 token 已改变的 `loops_done`。 |
| EOS 与下一 token 在途工作 | `test_delayed_eos_discards_next_work_and_reclaims_after_gpu_completion` 强制返回 EOS，并先提交下一 token 的 prelude/core | 只输出一个 EOS；不再次采样；KV 在相关 GPU 工作完成后归零，ID 可以立即复用。 |
| 取消后复用 | `test_abort_speculative_core_then_reuse_id` 在下一 token 的 core 已提交后取消 | 旧结果不能污染同 ID 的新请求；KV/state 释放正确，新请求正常完成。 |
| 在途上限、输入 DMA 复用 | `test_bounded_pipeline_survives_slow_gpu_and_dma_bank_reuse` 延长 GPU 执行并连续生成 | 不耗尽 readback 槽、不覆盖尚在读取的输入，不超过三个在途模型提交，输出与同步一致。 |
| coda 回收不能再次拆小 batch | `test_completed_core_refills_even_when_coda_cpu_delivery_is_pending` 令 GPU core 完成但 coda CPU 交付仍未进行 | 先进入 prelude 补回请求，再与其他请求组成 core batch；不能仅因 CPU 交付滞后就拆批。 |

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zjy/code/david/b_workspace/.b_rdma/bin/python -m pytest \
  tests/test_async_pipeline.py --run-gpu -vv \
  > "$RUN_DIR/async-pipeline.log" 2>&1
```

使用前先按本文前面的步骤设置 `RUN_DIR`，并选择空闲 GPU。
