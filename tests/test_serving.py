"""CPU contract and real-socket lifecycle tests for the serving frontend."""

import asyncio
import json
import threading
from contextlib import asynccontextmanager

import pytest
import torch

pytest.importorskip("aiohttp")
from aiohttp import ClientPayloadError
from aiohttp.test_utils import TestClient, TestServer

from vllm_lt import CacheConfig, SamplingParams, SchedulerConfig
from vllm_lt.engine.llm_engine import LLMEngine
from vllm_lt.models import OuroConfig, OuroForCausalLM
from vllm_lt.request import Stage
from vllm_lt.serving.protocol import CompletionRequest, IncrementalText, ServingError
from vllm_lt.serving.server import WORKER, create_app
from vllm_lt.serving.worker import EngineWorker


class TinyTokenizer:
    def encode(self, text):
        return [1 + ord(c) % 63 for c in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(64 + token) for token in ids if token != 0)


def factory(*, seqs=8, blocks=256, hook=None):
    torch.manual_seed(123)
    engine = LLMEngine(
        OuroForCausalLM(OuroConfig.tiny()),
        cache_config=CacheConfig(num_blocks=blocks, block_size=2),
        scheduler_config=SchedulerConfig(max_num_seqs=seqs, max_num_batched_tokens=3),
    )
    if hook:
        hook(engine)
    return engine, TinyTokenizer()


async def until(predicate):
    for _ in range(1000):
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition did not become true")


@asynccontextmanager
async def client_for(load=factory, **kwargs):
    app = create_app(load, **kwargs)
    client = TestClient(TestServer(app, handler_cancellation=True))
    await client.start_server()
    try:
        yield client, app[WORKER]
    finally:
        await client.close()


def body(**kwargs):
    return {
        "model": "ByteDance/Ouro-1.4B",
        "prompt": "abc",
        "max_tokens": 4,
        "ignore_eos": True,
        **kwargs,
    }


@pytest.mark.parametrize(
    "extra",
    [
        {"prompt": [1, 2]},
        {"prompt": None},
        {"stream": 1},
        {"ignore_eos": "true"},
        {"n": 2},
        {"n": True},
        {"logprobs": 1},
        {"stop": "end"},
        {"max_tokens": 0},
        {"max_tokens": None},
        {"max_tokens": True},
        {"temperature": "0"},
        {"temperature": float("nan")},
        {"top_p": True},
        {"seed": -1},
        {"repetition_penalty": 1.1},
        {"unknown": None},
        {"stream_options": {}},
        {"stream": True, "stream_options": {"include_usage": 1}},
    ],
)
def test_validation(extra):
    with pytest.raises((ValueError, TypeError)):
        CompletionRequest.parse(body(**extra), "ByteDance/Ouro-1.4B")


def test_neutral_client_fields_and_extensions():
    spec = CompletionRequest.parse(
        body(
            stream=True,
            stream_options={"include_usage": True},
            repetition_penalty=1,
            logprobs=None,
            min_loops=2,
            max_loops=3,
            exit_threshold=0.7,
            top_k=9,
            top_p=0.8,
        ),
        "ByteDance/Ouro-1.4B",
    )
    assert spec.include_usage and spec.params.max_loops == 3 and spec.params.top_k == 9


def test_byte_decoder_unicode_special_tokens_and_final_flush():
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    alphabet = sorted(tokenizers.pre_tokenizers.ByteLevel.alphabet())
    vocab = {token: i for i, token in enumerate(alphabet)}
    vocab["<eos>"] = len(vocab)
    backend = tokenizers.Tokenizer(tokenizers.models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = tokenizers.decoders.ByteLevel()
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<eos>")
    for text in ["你好🙂 café", "👩🏽‍💻", "hello 🌍", "a � b", "  a  b\n"]:
        ids = tokenizer.encode(text) + [tokenizer.eos_token_id]
        decoder = IncrementalText(tokenizer)
        deltas = [decoder.decode(ids[:i], i == len(ids)) for i in range(1, len(ids) + 1)]
        assert "".join(deltas) == text
        assert deltas[-1] == ""
    ids = tokenizer.encode("🙂")[:1]
    decoder = IncrementalText(tokenizer)
    assert decoder.decode(ids, False) == ""
    assert decoder.decode(ids, True) == tokenizer.decode(ids)


def test_startup_readiness_failure_and_shutdown_during_load():
    async def run():
        loading, release = threading.Event(), threading.Event()

        def load():
            loading.set()
            assert release.wait(5)
            return factory()

        async with client_for(load) as (client, worker):
            await until(loading.is_set)
            assert (await client.get("/health")).status == 503
            assert (await client.post("/v1/completions", json=body())).status == 503
            release.set()
            await until(lambda: worker.ready)
            assert (await client.get("/health")).status == 200
            models = await (await client.get("/v1/models")).json()
            assert models["data"][0]["id"] == body()["model"]

        def fail():
            raise RuntimeError("load failed")

        async with client_for(fail) as (client, worker):
            await until(lambda: worker.failure is not None)
            assert (await client.get("/health")).status == 503
        loading.clear()
        release.clear()
        worker = EngineWorker(load)
        worker.start()
        await until(loading.is_set)
        closing = asyncio.create_task(worker.close())
        await asyncio.sleep(0)
        release.set()
        await closing
        assert not worker.ready and worker.engine is None

    asyncio.run(run())


def test_http_output_matches_direct_and_stream_has_exact_token_events():
    engine, tokenizer = factory()
    engine.add_request(
        "direct", tokenizer.encode("abc"), SamplingParams(max_tokens=4, ignore_eos=True)
    )
    outputs = []
    while engine.has_unfinished_requests():
        outputs.extend(engine.step())
    expected = tokenizer.decode(outputs[-1].token_ids)

    async def run():
        async with client_for() as (client, worker):
            await until(lambda: worker.ready)
            response = await client.post("/v1/completions", json=body())
            data = await response.json()
            assert response.status == 200
            assert data["choices"][0]["text"] == expected
            assert data["usage"] == {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
            response = await client.post(
                "/v1/completions",
                json=body(
                    stream=True,
                    stream_options={"include_usage": True},
                    repetition_penalty=1,
                    logprobs=None,
                ),
            )
            wire = await response.text()
            frames = wire.strip().split("\n\n")
            assert frames[-1] == "data: [DONE]"
            events = [json.loads(frame.removeprefix("data: ")) for frame in frames[:-1]]
            tokens = [event for event in events if event["choices"]]
            assert len(tokens) == 4
            assert "".join(event["choices"][0]["text"] for event in tokens) == expected
            assert [event["choices"][0]["finish_reason"] for event in tokens] == [None] * 3 + [
                "length"
            ]
            assert events[-1]["choices"] == [] and events[-1]["usage"] == data["usage"]
            response = await client.post("/v1/completions", json=body(prompt=""))
            assert response.status == 400
            response = await client.post("/v1/completions", json=body(model="missing"))
            assert response.status == 404
            response = await client.post("/v1/completions", data="{bad")
            assert response.status == 400
            await until(lambda: not worker.channels)

    asyncio.run(run())


def test_eos_usage_and_non_usage_stream():
    def load():
        engine, tokenizer = factory()
        with torch.no_grad():
            engine.model.lm_head.weight.zero_()
        return engine, tokenizer

    async def run():
        async with client_for(load) as (client, worker):
            await until(lambda: worker.ready)
            response = await client.post("/v1/completions", json=body(ignore_eos=False))
            result = await response.json()
            assert result["choices"][0]["text"] == ""
            assert result["choices"][0]["finish_reason"] == "stop"
            assert result["usage"]["completion_tokens"] == 1
            response = await client.post(
                "/v1/completions", json=body(stream=True, ignore_eos=False)
            )
            frames = (await response.text()).strip().split("\n\n")
            assert len(frames) == 2 and "usage" not in json.loads(frames[0][6:])

    asyncio.run(run())


def test_owner_thread_dynamic_batching_cancellation_and_overload():
    async def run():
        blocked, release = threading.Event(), threading.Event()
        trace, owner_threads, engines = [], [], []

        def hook(engine):
            engines.append(engine)
            execute = engine.model_runner.execute

            def record(batch):
                owner_threads.append(threading.get_ident())
                trace.append((batch.stage, [item.request.request_id for item in batch.items]))
                if batch.stage == Stage.RECURRENT and not blocked.is_set():
                    blocked.set()
                    assert release.wait(5)
                return execute(batch)

            engine.model_runner.execute = record
            for name in ["add_request", "abort_request"]:
                method = getattr(engine, name)

                def check(*args, method=method):
                    owner_threads.append(threading.get_ident())
                    return method(*args)

                setattr(engine, name, check)

        worker = EngineWorker(lambda: factory(hook=hook), max_requests=2, output_buffer=32)
        worker.start()
        try:
            await until(lambda: worker.ready)
            first = worker.submit(CompletionRequest.parse(body(max_tokens=8), body()["model"]))
            await first.receive()
            await until(blocked.is_set)
            second = worker.submit(CompletionRequest.parse(body(max_tokens=8), body()["model"]))
            with pytest.raises(ServingError, match="capacity"):
                worker.submit(second.spec)
            release.set()
            await second.receive()
            await until(lambda: any(len(ids) == 2 for _, ids in trace))
            worker.release(first)
            while (await second.receive()).finish_reason is None:
                pass
            worker.release(second)
            await until(lambda: not worker.channels)
            assert engines[0].cache_manager.num_used_blocks == 0
            assert len(set(owner_threads)) == 1 and owner_threads[0] != threading.get_ident()
        finally:
            release.set()
            await worker.close()

    asyncio.run(run())


def test_slow_channel_does_not_block_other_requests():
    async def run():
        engines = []

        def load():
            engine, tokenizer = factory()
            engines.append(engine)
            return engine, tokenizer

        worker = EngineWorker(load, output_buffer=1, max_requests=2)
        worker.start()
        try:
            await until(lambda: worker.ready)
            slow = worker.submit(CompletionRequest.parse(body(max_tokens=20), body()["model"]))
            await until(lambda: slow.error is not None)
            assert slow.error.code == "slow_client"
            assert slow.request_id in worker.channels  # occupied until consumer releases it
            good = worker.submit(CompletionRequest.parse(body(max_tokens=1), body()["model"]))
            assert (await good.receive()).finish_reason == "length"
            worker.release(slow)
            worker.release(good)
            await until(lambda: not worker.channels)
            assert engines[0].cache_manager.num_used_blocks == 0
        finally:
            await worker.close()

    asyncio.run(run())


def test_disconnect_and_execution_failure_after_first_token():
    async def run(fail):
        blocked, release = threading.Event(), threading.Event()
        engines = []

        def hook(engine):
            engines.append(engine)
            execute = engine.model_runner.execute

            def pause(batch):
                if batch.stage == Stage.RECURRENT and not blocked.is_set():
                    blocked.set()
                    assert release.wait(5)
                    if fail:
                        raise RuntimeError("injected engine failure")
                return execute(batch)

            engine.model_runner.execute = pause

        try:
            async with client_for(lambda: factory(hook=hook)) as (client, worker):
                await until(lambda: worker.ready)
                response = await client.post(
                    "/v1/completions", json=body(stream=True, max_tokens=20)
                )
                first = await response.content.readline()
                assert json.loads(first[6:])["choices"]
                await until(blocked.is_set)
                assert (await client.get("/health")).status == 200
                if fail:
                    release.set()
                    with pytest.raises(ClientPayloadError):
                        await response.read()
                    await until(lambda: not worker.ready)
                    assert (await client.get("/health")).status == 503
                else:
                    response.close()
                    await until(lambda: any(c.cancelled for c in worker.channels.values()))
                    release.set()
                    await until(lambda: not worker.channels)
                    assert (
                        await client.post("/v1/completions", json=body(max_tokens=1))
                    ).status == 200
                await until(lambda: engines[0].cache_manager.num_used_blocks == 0)
        finally:
            release.set()

    asyncio.run(run(False))
    asyncio.run(run(True))


def test_shutdown_fails_waiters_and_cleans_up():
    async def run():
        blocked, release = threading.Event(), threading.Event()
        engines = []

        def hook(engine):
            engines.append(engine)
            execute = engine.model_runner.execute

            def pause(batch):
                if batch.stage == Stage.PREFILL:
                    blocked.set()
                    assert release.wait(5)
                return execute(batch)

            engine.model_runner.execute = pause

        worker = EngineWorker(lambda: factory(hook=hook))
        worker.start()
        await until(lambda: worker.ready)
        channel = worker.submit(CompletionRequest.parse(body(), body()["model"]))
        await until(blocked.is_set)
        closing = asyncio.create_task(worker.close())
        await asyncio.sleep(0)
        release.set()
        await closing
        with pytest.raises(ServingError, match="shutting down"):
            await channel.receive()
        assert not worker.channels and engines[0].cache_manager.num_used_blocks == 0

    asyncio.run(run())


def test_http_capacity_body_limit_and_cancellation_during_tokenization():
    async def run():
        entered, release = threading.Event(), threading.Event()
        engines = []

        class BlockingTokenizer(TinyTokenizer):
            def encode(self, text):
                if text == "hold":
                    entered.set()
                    assert release.wait(5)
                return super().encode(text)

        def load():
            engine, _ = factory()
            engines.append(engine)
            return engine, BlockingTokenizer()

        async with client_for(load, max_requests=1, max_body_bytes=256) as (client, worker):
            await until(lambda: worker.ready)
            too_big = await client.post("/v1/completions", json=body(prompt="x" * 512))
            assert too_big.status == 413
            connection = asyncio.create_task(
                client.post("/v1/completions", json=body(prompt="hold"))
            )
            try:
                await until(entered.is_set)
                overloaded = await client.post("/v1/completions", json=body())
                assert overloaded.status == 429 and overloaded.headers["Retry-After"] == "1"
                connection.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await connection
                await until(lambda: any(c.cancelled for c in worker.channels.values()))
                # The owner has not yet observed cancellation, so its slot stays reserved.
                assert len(worker.channels) == 1
                release.set()
                await until(lambda: not worker.channels)
                assert engines[0].cache_manager.num_used_blocks == 0
                assert (await client.post("/v1/completions", json=body(max_tokens=1))).status == 200
            finally:
                release.set()

    asyncio.run(run())


def test_write_deadline_aborts_socket_without_blocking_other_clients(monkeypatch):
    from aiohttp import web

    async def run():
        original_write = web.StreamResponse.write
        held_id = None
        hold = asyncio.Event()

        async def stalled_write(response, data):
            nonlocal held_id
            if data.startswith(b"data: {"):
                request_id = json.loads(data[6:])["id"]
                if held_id is None:
                    held_id = request_id
                if request_id == held_id:
                    await hold.wait()
            return await original_write(response, data)

        monkeypatch.setattr(web.StreamResponse, "write", stalled_write)
        async with client_for(write_timeout=0.1) as (client, worker):
            await until(lambda: worker.ready)
            slow = await client.post("/v1/completions", json=body(stream=True))
            good = await client.post("/v1/completions", json=body(max_tokens=1))
            assert good.status == 200
            with pytest.raises(ClientPayloadError):
                await slow.read()
            await until(lambda: not worker.channels)
            assert worker.ready

    asyncio.run(run())
