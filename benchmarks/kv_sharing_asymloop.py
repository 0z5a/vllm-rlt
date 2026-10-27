"""High-concurrency long-context benchmark on one in-process vllm-rlt engine (single GPU).

Research driver for the BeNice! (arXiv 2610.31633) KV-sharing and AsymLoop
(arXiv 2610.28489) serving proxies on Ouro-1.4B; see
docs/kv-sharing-asymloop-exploration.md in the upstream repository. Run from the
repository root, e.g. ``python -m benchmarks.kv_sharing_asymloop --help``.

Workloads
  burst     C requests of fresh random prompts submitted at once (no KV reuse).
  sessions  Closed-loop multi-turn agent sessions from the AgentX trace
            (cc-traces-weka-062126): every 64-token block hash maps to
            --tokens-per-block tokens seeded by (session, hash), so shared
            trace blocks are shared token-for-token -> prefix caching matters.
  open      Open-loop Poisson arrivals at --rate req/s of random prompts.
  mooncake  Open-loop replay of a Mooncake FAST'25 trace (--traces): the first
            --requests records, inter-arrival gaps rescaled to the offered
            rate, every 512-token hash block mapped to --tokens-per-block
            deterministic tokens (shared hash blocks are shared tokens, so
            prefix caching sees the trace's reuse), lengths capped by
            --max-prompt / --max-decode.

Per request: submit, first token, done, prompt/out tokens. Per step (optional):
wall time, phase, rows, loop count. Plus KV pool / prefix-cache counters and an
nvidia-smi SM-utilization sampler. One JSON record per invocation.
"""

import argparse
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
import threading
import time


def pct(values, q):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(round(q / 100 * len(ordered) + 0.5)) - 1))]


def summary(values):
    values = [v for v in values if v is not None]
    return dict(
        n=len(values),
        mean=statistics.fmean(values) if values else None,
        p50=pct(values, 50),
        p90=pct(values, 90),
        p95=pct(values, 95),
        p99=pct(values, 99),
        max=max(values) if values else None,
    )


def load_sessions(path, count, turns, tokens_per_block, max_prompt, vocab, seed, skip=0):
    sessions = []
    with open(path) as handle:
        for i, line in enumerate(handle):
            if i < skip:
                continue
            record = json.loads(line)
            main = [r for r in record["requests"] if r.get("type") != "subagent"]
            built = []
            for request in main[:turns]:
                prompt = []
                for block in request["hash_ids"]:
                    digest = hashlib.sha256(f"{record['id']}:{block}".encode()).digest()
                    rng = random.Random(int.from_bytes(digest[:8], "little") ^ seed)
                    prompt.extend(rng.randrange(100, vocab - 100) for _ in range(tokens_per_block))
                if len(prompt) > max_prompt or len(prompt) < 2:
                    break
                out = max(4, round(request["out"] * tokens_per_block / 64))
                built.append((prompt, out, float(request.get("think_time") or 0.0)))
            if len(built) >= 2:
                sessions.append((record["id"], built))
            if len(sessions) == count:
                break
    return sessions


def load_mooncake(path, count, tokens_per_block, max_prompt, max_decode, vocab, seed):
    """Mooncake rows -> [(arrival_seconds, prompt_tokens, output_tokens)], arrivals from 0."""
    rows = []
    scale = tokens_per_block / 512
    blocks = {}
    with open(path) as handle:
        for line in handle:
            record = json.loads(line)
            length = min(max_prompt, max(2, round(record["input_length"] * scale)))
            prompt = []
            for block in record["hash_ids"]:
                if block not in blocks:
                    digest = hashlib.sha256(f"mooncake:{block}".encode()).digest()
                    rng = random.Random(int.from_bytes(digest[:8], "little") ^ seed)
                    blocks[block] = [
                        rng.randrange(100, vocab - 100) for _ in range(tokens_per_block)
                    ]
                prompt.extend(blocks[block])
                if len(prompt) >= length:
                    break
            prompt = prompt[:length]
            if len(prompt) < length:  # trace rows whose hash blocks do not cover input_length
                digest = hashlib.sha256(f"mooncake-tail:{len(rows)}".encode()).digest()
                rng = random.Random(int.from_bytes(digest[:8], "little") ^ seed)
                prompt.extend(rng.randrange(100, vocab - 100) for _ in range(length - len(prompt)))
            out = min(max_decode, max(1, record["output_length"]))
            rows.append((record["timestamp"] / 1000.0, prompt, out))
            if len(rows) == count:
                break
    t0 = rows[0][0]
    return [(t - t0, p, o) for t, p, o in rows]


def busy_window(records, q=0.9):
    """Output rate until q of requests finish (wall is set by the longest chain)."""
    ends = sorted((r["done"], r["out"], r["prompt"]) for r in records)
    if not ends:
        return None
    t_q = ends[max(0, int(q * len(ends)) - 1)][0]
    done = [e for e in ends if e[0] <= t_q]
    return dict(
        t_seconds=t_q,
        output_tokens_per_second=sum(e[1] for e in done) / t_q if t_q > 0 else None,
        prompt_tokens_per_second=sum(e[2] for e in done) / t_q if t_q > 0 else None,
        requests=len(done),
    )


class SmiSampler(threading.Thread):
    def __init__(self, interval=0.2):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples = []
        self.stop_event = threading.Event()

    def run(self):
        try:
            proc = subprocess.Popen(
                [
                    "nvidia-smi",
                    "-i",
                    os.environ.get("BENCH_SMI_GPU", "0"),
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                    "-lms",
                    str(int(self.interval * 1000)),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except OSError:
            return
        while not self.stop_event.is_set():
            line = proc.stdout.readline()
            if not line:
                break
            try:
                util, mem = (float(x) for x in line.strip().split(","))
            except ValueError:
                continue
            self.samples.append((time.monotonic(), util, mem))
        proc.terminate()

    def stop(self):
        self.stop_event.set()

    def window(self, start, end):
        rows = [s for s in self.samples if start <= s[0] <= end]
        if not rows:
            return None
        utils = [r[1] for r in rows]
        return dict(
            samples=len(rows),
            sm_util_mean=statistics.fmean(utils),
            sm_util_p50=pct(utils, 50),
            mem_used_max_mb=max(r[2] for r in rows),
        )


def build_engine(args, model):
    import torch

    from vllm_rlt import CacheConfig, ExecutionConfig, SamplingParams, SchedulerConfig
    from vllm_rlt.config import ExitConfig, SpeculativeConfig
    from vllm_rlt.engine.llm_engine import LLMEngine

    del SamplingParams
    spec = None
    if args.spec:
        spec = SpeculativeConfig(
            num_speculative_tokens=args.spec,
            draft_loops=args.draft_loops,
            target_loops=model.config.total_ut_steps,
        )
    cache_kw = {}
    if args.kv_bytes:
        cache_kw["kv_cache_memory_bytes"] = int(args.kv_bytes * 1024**3)
    else:
        cache_kw["gpu_memory_utilization"] = args.kv_util
    cache = CacheConfig(
        block_size=args.block_size,
        shared_prefill=args.shared_prefill,
        enable_prefix_caching=args.prefix_cache,
        incremental_allocation=args.incremental_kv,
        watermark_ratio=args.kv_watermark,
        layout=args.kv_layout,
        **cache_kw,
    )
    sched = SchedulerConfig(
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_batched_tokens,
        mode=args.mode,
        prefill_chunk_size=args.chunk,
        max_prefill_batches_before_decode=args.prefill_batches_before_decode,
        enable_preemption=args.preemption,
        wavefront_chunked_prefill=args.wavefront,
        policy=args.policy,
    )
    exe = ExecutionConfig(
        async_scheduling=args.async_sched,
        multi_stream=not args.single_stream,
        static_buffers=args.static_buffers,
        pad_to_power_of_two=args.pad_pow2,
        cuda_graphs=args.cuda_graphs,
        cuda_graph_max_batch_size=args.graph_max,
        cuda_graph_max_graphs=args.graph_max_graphs,
        use_uva=not args.no_uva,
        prefill_uva=args.prefill_uva,
    )
    exit_cfg = ExitConfig(mode=args.exit_mode)
    engine = LLMEngine(
        model,
        cache_config=cache,
        scheduler_config=sched,
        execution_config=exe,
        exit_config=exit_cfg,
        speculative_config=spec,
        attention_backend=args.attention,
    )
    torch.cuda.synchronize()
    return engine


def sampling(args, max_tokens):
    from vllm_rlt import SamplingParams

    kw = dict(max_tokens=max_tokens, ignore_eos=True, exit_threshold=args.exit_threshold)
    if args.loops:
        kw["min_loops"] = kw["max_loops"] = args.loops
    if args.min_loops:
        kw["min_loops"] = args.min_loops
    if args.temperature > 0:
        kw.update(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
    return SamplingParams(**kw)


class Driver:
    """Steps the engine, records per-request and per-step timings."""

    def __init__(self, engine, step_trace, dump_tokens=False):
        self.engine = engine
        self.step_trace = step_trace
        self.dump_tokens = dump_tokens
        self.records = {}
        self.first = {}
        self.steps = []
        self.exit_depths = {}
        self.counter = 0

    def submit(self, prompt, params, meta):
        rid = f"r{self.counter}"
        self.counter += 1
        self.records[rid] = dict(rid=rid, prompt=len(prompt), submit=time.monotonic(), **meta)
        self.engine.add_request(rid, prompt, params)
        return rid

    def step(self):
        t0 = time.monotonic()
        outputs = self.engine.step()
        t1 = time.monotonic()
        batch = self.engine.last_schedule
        if self.step_trace:
            self.steps.append(
                (
                    round(t0, 6),
                    round(t1 - t0, 6),
                    batch.phase.value
                    if batch is not None and batch.phase is not None
                    else ("mixed" if batch is not None else None),
                    batch.num_tokens if batch is not None else 0,
                    len(batch.requests) if batch is not None else 0,
                    batch.loop_count if batch is not None else None,
                )
            )
        finished = []
        for out in outputs:
            rec = self.records.get(out.request_id)
            if rec is None:
                continue
            if out.token_ids and out.request_id not in self.first:
                self.first[out.request_id] = t1
                rec["first"] = t1
            if out.finished:
                rec["done"] = t1
                rec["out"] = len(out.token_ids)
                rec["finish_reason"] = out.finish_reason
                if self.dump_tokens:
                    rec["token_ids"] = list(out.token_ids)
                    rec["depths"] = list(out.exit_depths or [])
                depths = list(out.exit_depths or [])
                for d in depths:
                    self.exit_depths[d] = self.exit_depths.get(d, 0) + 1
                finished.append(rec)
        return finished

    def drain(self, timeout):
        deadline = time.monotonic() + timeout
        while self.engine.has_unfinished_requests():
            self.step()
            if time.monotonic() > deadline:
                raise TimeoutError("drain timed out")


REAL_PROMPTS = []


def run_burst(args, driver, concurrency, rng, vocab, warm=False):
    # Warm-up: the same concurrency with short prompts captures every decode graph
    # size the measured cell will use without paying for its long prefills.
    params = sampling(args, 32 if warm else args.decode)
    length = min(args.prompt_len, 512) if warm else args.prompt_len
    t_start = time.monotonic()
    for i in range(concurrency):
        if REAL_PROMPTS:
            prompt = REAL_PROMPTS[(driver.counter) % len(REAL_PROMPTS)]
        else:
            prompt = [rng.randrange(100, vocab - 100) for _ in range(length)]
        driver.submit(prompt, params, dict(kind="burst", c=concurrency))
    driver.drain(args.timeout)
    return t_start, time.monotonic()


def run_open(args, driver, rate, rng, vocab):
    params = sampling(args, args.decode)
    t_start = time.monotonic()
    arrivals = []
    t = 0.0
    for _ in range(args.requests):
        t += rng.expovariate(rate)
        arrivals.append(t)
    i = 0
    while i < len(arrivals) or driver.engine.has_unfinished_requests():
        now = time.monotonic() - t_start
        while i < len(arrivals) and arrivals[i] <= now:
            prompt = [rng.randrange(100, vocab - 100) for _ in range(args.prompt_len)]
            driver.submit(prompt, params, dict(kind="open", rate=rate))
            i += 1
        if driver.engine.has_unfinished_requests():
            driver.step()
        else:
            time.sleep(min(0.001, max(0.0, arrivals[i] - now)) if i < len(arrivals) else 0)
        if time.monotonic() - t_start > args.timeout:
            raise TimeoutError("open-loop run timed out")
    return t_start, time.monotonic()


def run_mooncake(args, driver, rows, rate):
    """Replay trace arrivals compressed or stretched so that the offered rate is ``rate``."""
    natural = (len(rows) - 1) / rows[-1][0] if rows[-1][0] > 0 else rate
    stretch = natural / rate
    t_start = time.monotonic()
    i = 0
    while i < len(rows) or driver.engine.has_unfinished_requests():
        now = time.monotonic() - t_start
        while i < len(rows) and rows[i][0] * stretch <= now:
            _, prompt, out = rows[i]
            driver.submit(
                prompt, sampling(args, out), dict(kind="mooncake", rate=rate, trace_row=i)
            )
            i += 1
        if driver.engine.has_unfinished_requests():
            driver.step()
        elif i < len(rows):
            time.sleep(min(0.001, max(0.0, rows[i][0] * stretch - now)))
        if time.monotonic() - t_start > args.timeout:
            raise TimeoutError("mooncake replay timed out")
    return t_start, time.monotonic()


def run_sessions(args, driver, sessions, rng):
    """Closed loop: a session's next turn arrives after its previous turn finishes + think."""
    t_start = time.monotonic()
    state = {}  # sid -> (turn index, ready_at)
    for k, (sid, turns) in enumerate(sessions):
        state[sid] = [0, t_start + args.ramp * k / max(1, len(sessions))]
    active = {}  # rid -> sid
    remaining = sum(len(t) for _, t in sessions)
    by_sid = {sid: turns for sid, turns in sessions}
    while remaining > 0:
        now = time.monotonic()
        for sid, st in state.items():
            idx, ready = st
            if idx is None or ready is None or ready > now:
                continue
            prompt, out, think = by_sid[sid][idx]
            params = sampling(args, out)
            rid = driver.submit(
                prompt, params, dict(kind="session", sid=sid, turn=idx, think=think)
            )
            active[rid] = sid
            st[1] = None  # in flight
        if driver.engine.has_unfinished_requests():
            for rec in driver.step():
                remaining -= 1
                sid = active.pop(rec["rid"])
                st = state[sid]
                st[0] += 1
                if st[0] >= len(by_sid[sid]):
                    st[0] = None
                else:
                    think = min(args.think_cap, by_sid[sid][st[0] - 1][2] * args.think_scale)
                    st[1] = rec["done"] + think
        else:
            nxt = min((st[1] for st in state.values() if st[1] is not None), default=None)
            if nxt is None:
                break
            time.sleep(max(0.0, min(0.05, nxt - time.monotonic())))
        if time.monotonic() - t_start > args.timeout:
            raise TimeoutError("sessions run timed out")
    return t_start, time.monotonic()


def finish_cell(args, driver, engine, t_start, t_end, smi, cell_meta, cache_before):
    recs = [r for r in driver.records.values() if "done" in r]
    for r in recs:
        r["ttft"] = r["first"] - r["submit"] if "first" in r else None
        r["tpot"] = (
            (r["done"] - r["first"]) / (r["out"] - 1) if "first" in r and r["out"] > 1 else None
        )
        r["e2e"] = r["done"] - r["submit"]
    base = t_start
    for r in recs:
        for key in ("submit", "first", "done"):
            if key in r:
                r[key] = r[key] - base
    wall = t_end - t_start
    out_tokens = sum(r["out"] for r in recs)
    prompt_tokens = sum(r["prompt"] for r in recs)
    cache = engine.cache_manager
    hits = cache.prefix_hits - cache_before[0]
    queries = cache.prefix_queries - cache_before[1]
    steps = driver.steps
    step_summary = None
    if steps:
        durations = [s[1] for s in steps]
        phases = {}
        for s in steps:
            p = phases.setdefault(s[2], dict(n=0, rows=0, reqs=0, time=0.0))
            p["n"] += 1
            p["rows"] += s[3]
            p["reqs"] += s[4]
            p["time"] += s[1]
        step_summary = dict(
            steps=len(steps),
            step_time=summary(durations),
            step_time_total=sum(durations),
            phases=phases,
        )
    cell = dict(
        **cell_meta,
        wall=wall,
        requests=len(recs),
        output_tokens=out_tokens,
        prompt_tokens=prompt_tokens,
        output_tokens_per_second=out_tokens / wall if wall > 0 else None,
        prompt_tokens_per_second=prompt_tokens / wall if wall > 0 else None,
        busy_t90=busy_window(recs, 0.9),
        busy_t50=busy_window(recs, 0.5),
        ttft=summary([r["ttft"] for r in recs]),
        tpot=summary([r["tpot"] for r in recs]),
        e2e=summary([r["e2e"] for r in recs]),
        finish_reasons={
            k: sum(1 for r in recs if r.get("finish_reason") == k)
            for k in {r.get("finish_reason") for r in recs}
        },
        all_complete=all(r["out"] >= 1 for r in recs),
        prefix_cache=dict(
            hit_tokens=hits,
            queries=queries,
            hit_tokens_per_prompt_token=(hits / prompt_tokens) if prompt_tokens else None,
        ),
        kv=dict(
            num_blocks=cache.num_blocks,
            block_size=cache.block_size,
            used_blocks_end=cache.num_used_blocks,
        ),
        exit_depths=dict(sorted(driver.exit_depths.items())),
        smi=smi.window(t_start, t_end) if smi else None,
        step_summary=step_summary,
    )
    import torch

    cell["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    cell["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    return cell, recs, steps


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--workload", choices=["burst", "sessions", "open", "mooncake"], default="burst"
    )
    parser.add_argument(
        "--concurrency", default="1,8,32,64,128", help="burst: C; sessions: #sessions; open: rates"
    )
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--decode", type=int, default=128)
    parser.add_argument("--requests", type=int, default=256, help="open-loop request count")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1, help="warm-up runs per cell (0/1)")
    parser.add_argument("--traces")
    parser.add_argument("--turns", type=int, default=16)
    parser.add_argument("--tokens-per-block", type=int, default=2)
    parser.add_argument("--max-prompt", type=int, default=16000)
    parser.add_argument("--think-scale", type=float, default=0.1)
    parser.add_argument("--think-cap", type=float, default=2.0)
    parser.add_argument("--ramp", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--timeout", type=float, default=1800)
    # engine knobs
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attention", default="triton")
    parser.add_argument("--kv-util", type=float, default=0.85)
    parser.add_argument("--kv-layout", default="last_exited")
    parser.add_argument("--shared-prefill", default="position", choices=["position", "loop"])
    parser.add_argument(
        "--kv-bytes",
        type=float,
        default=0.0,
        help="pin the KV pool to this many GiB (same for every arm)",
    )
    parser.add_argument("--max-decode", type=int, default=512)
    parser.add_argument(
        "--asymloop-encoder",
        type=int,
        default=0,
        help="AsymLoop: one-pass encoder layers (0 = Full Loop)",
    )
    parser.add_argument(
        "--asymloop-loops", type=int, default=4, help="AsymLoop: decoder loop count K"
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--prefix-cache", action="store_true")
    parser.add_argument("--incremental-kv", action="store_true")
    parser.add_argument("--kv-watermark", type=float, default=0.0)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--max-batched-tokens", type=int, default=2048)
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--prefill-batches-before-decode", type=int, default=1)
    parser.add_argument("--mode", default="refill", choices=["refill", "no_refill", "mixed"])
    parser.add_argument("--wavefront", action="store_true")
    parser.add_argument("--preemption", action="store_true")
    parser.add_argument("--policy", default="fcfs")
    parser.add_argument("--async", dest="async_sched", action="store_true")
    parser.add_argument("--single-stream", action="store_true")
    parser.add_argument("--static-buffers", action="store_true")
    parser.add_argument("--pad-pow2", action="store_true")
    parser.add_argument("--cuda-graphs", action="store_true")
    parser.add_argument("--graph-max", type=int, default=128)
    parser.add_argument("--graph-max-graphs", type=int, default=16)
    parser.add_argument("--no-uva", action="store_true")
    parser.add_argument("--prefill-uva", action="store_true")
    parser.add_argument("--exit-mode", default="ouro_delayed")
    parser.add_argument("--exit-threshold", type=float, default=1.0, help="1.0 = fixed depth")
    parser.add_argument("--loops", type=int, default=0, help="fix min=max loops (Huginn depth)")
    parser.add_argument("--min-loops", type=int, default=0)
    parser.add_argument("--spec", type=int, default=0, help="speculative tokens K")
    parser.add_argument("--draft-loops", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--step-trace", action="store_true")
    parser.add_argument("--no-smi", action="store_true")
    parser.add_argument("--save-rows", action="store_true")
    parser.add_argument(
        "--dump-tokens", action="store_true", help="keep token ids per request (implies rows)"
    )
    parser.add_argument(
        "--cprofile",
        help="profile the measured (non-warm-up) runs with cProfile into this .prof path",
    )
    parser.add_argument(
        "--prompts-json",
        help="burst: JSON list of real text prompts, tokenized and cycled per request",
    )
    return parser


def main():
    args = make_parser().parse_args()

    import torch

    from vllm_rlt.models import AutoModelForCausalLM

    dtype = getattr(torch, args.dtype)
    t_load = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(args.model, device="cuda:0", dtype=dtype)
    if args.asymloop_encoder:
        from vllm_rlt.models.asymloop import AsymLoopOuroForCausalLM

        model = AsymLoopOuroForCausalLM(
            model, encoder_layers=args.asymloop_encoder, loops=args.asymloop_loops
        )
    if args.prompts_json:
        from vllm_rlt.models import load_tokenizer

        tok = load_tokenizer(args.model)
        texts = json.load(open(args.prompts_json))
        REAL_PROMPTS.extend([list(tok.encode(t)) for t in texts])
    load_s = time.monotonic() - t_load
    vocab = model.config.vocab_size
    record = dict(
        label=args.label,
        args=vars(args),
        source_sha=os.environ.get("SOURCE_SHA"),
        model_type=getattr(model.config, "model_type", type(model).__name__),
        total_ut_steps=model.config.total_ut_steps,
        architecture="asymloop" if args.asymloop_encoder else "full_loop",
        encoder_layers=args.asymloop_encoder,
        load_seconds=load_s,
        device=torch.cuda.get_device_name(0),
        torch=torch.__version__,
        cells=[],
        errors=[],
    )
    t_eng = time.monotonic()
    try:
        engine = build_engine(args, model)
    except Exception as exc:  # configuration rejected: record and exit
        record["errors"].append(dict(stage="build_engine", error=repr(exc)))
        json.dump(record, open(args.out, "w"), indent=1)
        print("BUILD_FAILED", repr(exc), flush=True)
        return 2
    record["engine_build_seconds"] = time.monotonic() - t_eng
    cm = engine.cache_manager
    record["kv_num_blocks"] = cm.num_blocks
    record["kv_bytes_per_block"] = cm.bytes_per_block
    record["kv_pool_bytes"] = cm.num_blocks * cm.bytes_per_block
    record["kv_storage_planes"] = cm.storage_depths
    record["kv_token_capacity"] = cm.num_blocks // cm.storage_depths * cm.block_size
    record["kv_layout"] = cm.layout
    record["shared_prefill"] = getattr(cm, "shared_prefill", None)
    print(
        "CAPACITY",
        json.dumps(
            {
                k: record[k]
                for k in (
                    "kv_num_blocks",
                    "kv_pool_bytes",
                    "kv_storage_planes",
                    "kv_token_capacity",
                )
            }
        ),
        flush=True,
    )
    smi = None if args.no_smi else SmiSampler()
    if smi:
        smi.start()
    rng = random.Random(args.seed)
    sessions_all = None
    if args.workload == "sessions":
        sessions_all = load_sessions(
            args.traces, 4096, args.turns, args.tokens_per_block, args.max_prompt, vocab, args.seed
        )
        record["sessions_available"] = len(sessions_all)
    mooncake_rows = None
    if args.workload == "mooncake":
        mooncake_rows = load_mooncake(
            args.traces,
            args.requests,
            args.tokens_per_block,
            args.max_prompt,
            args.max_decode,
            vocab,
            args.seed,
        )
        lengths = sorted(len(p) for _, p, _ in mooncake_rows)
        record["mooncake"] = dict(
            rows=len(mooncake_rows),
            prompt_mean=statistics.fmean(lengths),
            prompt_p50=lengths[len(lengths) // 2],
            prompt_p90=lengths[int(0.9 * len(lengths))],
            prompt_max=lengths[-1],
            output_mean=statistics.fmean(o for _, _, o in mooncake_rows),
            natural_rate=(len(mooncake_rows) - 1) / mooncake_rows[-1][0],
        )

    levels = [
        float(x) if args.workload in ("open", "mooncake") else int(x)
        for x in args.concurrency.split(",")
    ]
    profiler = None
    if args.cprofile:
        import cProfile

        profiler = cProfile.Profile()
    for level in levels:
        for rep in range(args.warmup + args.repeats):
            warm = rep < args.warmup
            driver = Driver(engine, args.step_trace and not warm, dump_tokens=args.dump_tokens)
            cache_before = (engine.cache_manager.prefix_hits, engine.cache_manager.prefix_queries)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            if profiler is not None and not warm:
                profiler.enable()
            try:
                if args.workload == "burst":
                    t0, t1 = run_burst(args, driver, int(level), rng, vocab, warm=warm)
                elif args.workload == "mooncake":
                    if warm:
                        t0, t1 = run_burst(args, driver, 16, rng, vocab, warm=True)
                    else:
                        t0, t1 = run_mooncake(args, driver, mooncake_rows, float(level))
                elif args.workload == "open":
                    if warm:
                        t0, t1 = run_burst(args, driver, 8, rng, vocab)
                    else:
                        t0, t1 = run_open(args, driver, float(level), rng, vocab)
                else:
                    n = int(level)
                    if warm:
                        subset = sessions_all[: min(n, 8)]
                        subset = [(sid, turns[:2]) for sid, turns in subset]
                    else:
                        # disjoint session pool per repeat so KV reuse is only intra-run
                        start = (rep - args.warmup) * n
                        subset = sessions_all[start : start + n]
                        if len(subset) < n:
                            record["errors"].append(
                                dict(
                                    stage="sessions", error=f"only {len(subset)} sessions for n={n}"
                                )
                            )
                    t0, t1 = run_sessions(args, driver, subset, rng)
            except Exception as exc:
                record["errors"].append(dict(stage=f"level={level} rep={rep}", error=repr(exc)))
                print("CELL_FAILED", level, rep, repr(exc), flush=True)
                json.dump(record, open(args.out, "w"), indent=1)
                try:
                    engine.model_runner.synchronize()
                    for rid in list(engine.scheduler.requests):
                        engine.abort_request(rid)
                except Exception as exc2:
                    record["errors"].append(dict(stage="recover", error=repr(exc2)))
                    break
                continue
            torch.cuda.synchronize()
            if profiler is not None and not warm:
                profiler.disable()
            if warm:
                continue
            cell, recs, steps = finish_cell(
                args,
                driver,
                engine,
                t0,
                t1,
                smi,
                dict(level=level, rep=rep - args.warmup),
                cache_before,
            )
            if args.save_rows or args.dump_tokens:
                cell["rows"] = recs
            if steps:
                cell["steps"] = steps
            for runner_name, field in (
                ("model_runner", "graphs"),
                ("speculative_runner", "spec_graphs"),
            ):
                graphs = getattr(getattr(engine, runner_name, None), "graphs", None)
                if graphs is not None:  # cumulative per engine
                    cell[field] = {
                        k: getattr(graphs, k, None)
                        for k in (
                            "captures",
                            "replays",
                            "fallbacks",
                            "evictions",
                            "capture_seconds",
                        )
                    }
                    cell[field]["entries"] = len(graphs.entries)
            record["cells"].append(cell)
            brief = {k: cell[k] for k in ("level", "wall", "requests", "output_tokens_per_second")}
            brief["ttft_p50"] = cell["ttft"]["p50"]
            brief["tpot_p50"] = cell["tpot"]["p50"]
            brief["busy90"] = (
                cell["busy_t90"]["output_tokens_per_second"] if cell["busy_t90"] else None
            )
            brief["hit"] = cell["prefix_cache"]["hit_tokens_per_prompt_token"]
            brief["sm"] = cell["smi"]["sm_util_mean"] if cell["smi"] else None
            print(json.dumps(brief), flush=True)
            json.dump(record, open(args.out, "w"), indent=1)
    if smi:
        smi.stop()
    if profiler is not None:
        import io
        import pstats

        profiler.dump_stats(args.cprofile)
        for sort in ("tottime", "cumulative"):
            buf = io.StringIO()
            pstats.Stats(profiler, stream=buf).sort_stats(sort).print_stats(45)
            open(args.cprofile + f".{sort}.txt", "w").write(buf.getvalue())
    try:
        engine.close()
    except Exception as exc:
        record["errors"].append(dict(stage="close", error=repr(exc)))
    json.dump(record, open(args.out, "w"), indent=1)
    print("DONE", args.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
