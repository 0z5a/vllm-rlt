"""Finite localhost HTTP token-ID load using the installed Python runtime.

The transport is an experiment adapter, separate from the aiohttp text server.
One engine owner thread serves real TCP clients on an ephemeral loopback port.
Client timestamps include transport, queueing and the complete generation path.
"""

import asyncio
import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

from experiments.loopkv.metrics import WorkCounters
from vllm_rlt import SamplingParams


async def run_load(engine, workload, params, *, concurrency=None):
    """Drain every request; concurrency=None replays open-loop arrival_s offsets."""
    assert workload and len({row["request_id"] for row in workload}) == len(workload)
    assert concurrency is None or concurrency > 0
    pending, channels, delivered = deque(), {}, {}
    connections: set[asyncio.StreamWriter] = set()
    wake = asyncio.Event()
    stopping = False
    counters = WorkCounters()
    loop = asyncio.get_running_loop()
    active = peak = 0

    def tick(admissions):
        for body in admissions:
            engine.add_request(
                body["request_id"], body["prompt_token_ids"], SamplingParams(**body["params"])
            )
        outputs = engine.step()
        counters.observe(engine.last_schedule, len(engine.cache_manager._allocations))
        return outputs, engine.has_unfinished_requests()

    async def owner(executor):
        running = False
        while not stopping or pending or running:
            if not pending and not running:
                wake.clear()
                await wake.wait()
                continue
            admissions = list(pending)
            pending.clear()
            outputs, running = await loop.run_in_executor(executor, tick, admissions)
            for output in outputs:
                rid = output.request_id
                count = delivered.get(rid, 0)
                channels[rid].put_nowait(
                    {
                        "token_ids": output.token_ids[count:],
                        "finished": output.finished,
                        "exit_depths": output.exit_depths if output.finished else None,
                        "finish_reason": output.finish_reason,
                    }
                )
                delivered[rid] = len(output.token_ids)
                if output.finished:
                    del delivered[rid]

    async def handle(reader, writer):
        connections.add(writer)
        headers = (await reader.readuntil(b"\r\n\r\n")).decode("ascii").split("\r\n")
        assert headers[0] == "POST /generate HTTP/1.1"
        fields = dict(line.split(": ", 1) for line in headers[1:] if line)
        body = json.loads(await reader.readexactly(int(fields["Content-Length"])))
        rid = body["request_id"]
        assert rid not in channels
        events = channels[rid] = asyncio.Queue()
        pending.append(body)
        wake.set()
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nConnection: close\r\n\r\n"
        )
        while True:
            event = await events.get()
            writer.write(json.dumps(event, separators=(",", ":")).encode() + b"\n")
            await writer.drain()
            if event["finished"]:
                break
        del channels[rid]
        writer.close()
        await writer.wait_closed()
        connections.discard(writer)

    bodies = []
    for row in workload:
        request_params = asdict(params) | {"max_tokens": row["max_tokens"], "seed": row["seed"]}
        bodies.append(
            json.dumps(
                {
                    "request_id": row["request_id"],
                    "prompt_token_ids": row["prompt_token_ids"],
                    "params": request_params,
                },
                separators=(",", ":"),
            ).encode()
        )
    semaphore = asyncio.Semaphore(concurrency or len(workload))
    results = []
    server = await asyncio.start_server(handle, "127.0.0.1", 0, backlog=4096)
    port = server.sockets[0].getsockname()[1]

    async def request(row, body, start):
        nonlocal active, peak
        planned = start + row["arrival_s"]
        await asyncio.sleep(max(0, planned - loop.time()))
        async with semaphore:
            sent = loop.time()
            active += 1
            peak = max(peak, active)
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(
                f"POST /generate HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
                f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
                + body
            )
            await writer.drain()
            headers = await reader.readuntil(b"\r\n\r\n")
            assert headers.startswith(b"HTTP/1.1 200 OK\r\n")
            tokens, times, chunks = [], [], []
            while True:
                event = json.loads(await reader.readline())
                observed = loop.time() - start
                tokens.extend(event["token_ids"])
                times.extend([observed] * len(event["token_ids"]))
                chunks.append({"time_s": observed, "tokens": len(event["token_ids"])})
                if event["finished"]:
                    break
            ended = loop.time()
            writer.close()
            await writer.wait_closed()
            active -= 1
            results.append(
                {
                    "request_id": row["request_id"],
                    "planned_s": row["arrival_s"],
                    "sent_s": sent - start,
                    "finished_s": ended - start,
                    "token_ids": tokens,
                    "token_times_s": times,
                    "chunks": chunks,
                    "exit_depths": event["exit_depths"],
                    "finish_reason": event["finish_reason"],
                }
            )

    async def clients(start):
        nonlocal stopping
        await asyncio.gather(*(request(row, body, start) for row, body in zip(workload, bodies)))
        stopping = True
        wake.set()

    async with server:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="loopkv-engine") as executor:
            start = loop.time()
            try:
                async with asyncio.TaskGroup() as group:
                    group.create_task(owner(executor))
                    group.create_task(clients(start))
            finally:
                # An engine error leaves HTTP handlers waiting for token events.
                # Close their streams before Server.__aexit__ waits for clients.
                for writer in connections:
                    writer.close()
                await asyncio.gather(*(writer.wait_closed() for writer in connections))
            await loop.run_in_executor(executor, engine.model_runner.synchronize)
            seconds = loop.time() - start
    assert not channels and not pending and not delivered
    assert not engine.has_unfinished_requests()
    assert engine.cache_manager.num_free_blocks == engine.cache_manager.num_blocks
    assert len(results) == len(workload)
    return {
        "seconds": seconds,
        "requests": sorted(results, key=lambda row: row["request_id"]),
        "peak_client_concurrency": peak,
        "arrival_mode": "closed_loop" if concurrency is not None else "open_loop",
        "work": counters.summary(),
        "scope": "localhost_http_token_ids_shared_client_server_event_loop_full_drain",
    }
