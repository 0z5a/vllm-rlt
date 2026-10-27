"""Batched GSM8K gate on a frozen harness protocol (benchmarks/gsm8k.py prepare output).

Research gate for the BeNice! KV-sharing proxy (docs/kv-sharing-asymloop-exploration.md).

Same records, prompt token IDs, stop strings and lm-eval scorer as the repository harness,
but all questions are submitted at once to one engine (CUDA graphs, sync scheduling for
Huginn) instead of one request at a time. Greedy decoding; a request is aborted as soon as
its decoded text contains a stop string, and the text is cut there, as the harness does.

  run:     --protocol P --out OUT.json --label L [--prefix-cache] [--max-num-seqs 32]
           [--kv-layout last_exited|shared] [--shared-prefill position|loop]
           (VLLM_RLT_HUGINN_POSITION_NOISE is read from the environment at import)
  compare: --compare REF.json CAND.json
"""

import argparse
import json
import math
import os
import sys
import time


def cut(text, stops):
    end = len(text)
    for s in stops:
        i = text.find(s)
        if i != -1:
            end = min(end, i)
    return text[:end], end < len(text)


def run(args):
    if args.prefix_cache:
        os.environ["VLLM_RLT_HUGINN_PREFIX_CACHE"] = "1"
    import torch
    from transformers import AutoTokenizer

    from benchmarks.gsm8k import make_task, score
    from vllm_rlt import CacheConfig, ExecutionConfig, SamplingParams, SchedulerConfig
    from vllm_rlt.config import ExitConfig
    from vllm_rlt.engine.llm_engine import LLMEngine
    from vllm_rlt.models import AutoModelForCausalLM

    protocol = json.load(open(args.protocol))
    loops = protocol["loops"]
    tokenizer = AutoTokenizer.from_pretrained(protocol["model"], local_files_only=True)
    task = make_task(protocol["task_config"])
    model = AutoModelForCausalLM.from_pretrained(
        protocol["model"], device="cuda:0", dtype=torch.bfloat16
    )
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(
            block_size=16,
            gpu_memory_utilization=0.85,
            enable_prefix_caching=args.prefix_cache,
            layout=args.kv_layout,
            shared_prefill=args.shared_prefill,
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=args.max_num_seqs, max_num_batched_tokens=4096, prefill_chunk_size=2048
        ),
        execution_config=ExecutionConfig(
            async_scheduling=False, cuda_graphs=True, cuda_graph_max_batch_size=args.max_num_seqs
        ),
        exit_config=ExitConfig(mode="ouro"),
        attention_backend="triton",
    )
    params = SamplingParams(
        max_tokens=protocol["max_new_tokens"], seed=args.seed, min_loops=loops, max_loops=loops
    )
    records = {f"q{r['id']}": r for r in protocol["records"]}
    stops = {
        rid: r["generation_kwargs"]["until"] + [tokenizer.eos_token] for rid, r in records.items()
    }
    latest, done, stopped = {}, {}, set()
    t0 = time.monotonic()
    for rid, r in records.items():
        engine.add_request(rid, r["prompt_ids"], params)
    checks = 0
    while engine.has_unfinished_requests():
        for out in engine.step():
            latest[out.request_id] = list(out.token_ids)
            if out.finished:
                done[out.request_id] = (list(out.token_ids), out.finish_reason)
        checks += 1
        if checks % 4 == 0:
            for rid, ids in list(latest.items()):
                if rid in done:
                    continue
                text = tokenizer.decode(ids, skip_special_tokens=False)
                if cut(text, stops[rid])[1]:
                    aborted = engine.abort_request(rid)
                    ids = list(aborted.token_ids) if aborted is not None else ids
                    done[rid] = (ids, "stop")
                    stopped.add(rid)
    seconds = time.monotonic() - t0
    hits = engine.cache_manager.prefix_hits if hasattr(engine, "cache_manager") else None
    engine.close()
    rows = []
    for rid, r in records.items():
        ids, reason = done.get(rid, (latest.get(rid, []), None))
        text, hit_stop = cut(tokenizer.decode(ids, skip_special_tokens=False), stops[rid])
        result = score(task, r, text)
        rows.append(
            dict(id=r["id"], token_ids=ids, text=text, finish=reason, stop=hit_stop, **result)
        )
    summary = dict(
        label=args.label,
        protocol=os.path.abspath(args.protocol),
        source_sha=os.environ.get("SOURCE_SHA"),
        position_noise=os.environ.get("VLLM_RLT_HUGINN_POSITION_NOISE", "0"),
        prefix_cache=args.prefix_cache,
        kv_layout=args.kv_layout,
        shared_prefill=args.shared_prefill,
        max_num_seqs=args.max_num_seqs,
        examples=len(rows),
        correct=sum(r["correct"] for r in rows),
        unparseable=sum(r["unparseable"] for r in rows),
        stopped_by_string=len(stopped),
        generated_tokens_mean=sum(len(r["token_ids"]) for r in rows) / len(rows),
        seconds=seconds,
        prefix_hit_tokens=hits,
    )
    json.dump(dict(summary=summary, rows=rows), open(args.out, "w"))
    print(json.dumps(summary), flush=True)


def compare(ref_path, cand_path):
    ref, cand = json.load(open(ref_path)), json.load(open(cand_path))
    a = {r["id"]: r for r in ref["rows"]}
    b = {r["id"]: r for r in cand["rows"]}
    ids = sorted(set(a) & set(b))
    diffs = [int(b[i]["correct"]) - int(a[i]["correct"]) for i in ids]
    n = len(ids)
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
    out = dict(
        reference=ref["summary"]["label"],
        candidate=cand["summary"]["label"],
        n=n,
        reference_correct=sum(a[i]["correct"] for i in ids),
        candidate_correct=sum(b[i]["correct"] for i in ids),
        delta_pp=100 * mean,
        paired_se_pp=100 * math.sqrt(var / n) if n > 1 else 0.0,
        ref_only=sum(d < 0 for d in diffs),
        cand_only=sum(d > 0 for d in diffs),
        identical_sequences=sum(a[i]["token_ids"] == b[i]["token_ids"] for i in ids),
        identical_answers=sum(a[i]["answer"] == b[i]["answer"] for i in ids),
    )
    print(json.dumps(out))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol")
    ap.add_argument("--out")
    ap.add_argument("--label")
    ap.add_argument("--prefix-cache", action="store_true")
    ap.add_argument("--kv-layout", default="last_exited", choices=["last_exited", "shared"])
    ap.add_argument("--shared-prefill", default="position", choices=["position", "loop"])
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compare", nargs=2)
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
