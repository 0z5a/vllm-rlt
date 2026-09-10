"""Validation-only native observation and supplied-input execution.

All hooks are per-instance and restored on exit. They observe actual computation;
only supplied next inputs and explicitly forced exits change controller decisions.
"""

import hashlib
import json
from contextlib import contextmanager

import torch

from vllm_lt.engine.llm_engine import LLMEngine
from vllm_lt.models.serial_oracle import KVSnapshot
from vllm_lt.request import Stage


def history_hash(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def snapshot_native(cache, request_id, positions):
    """Use this manager's page tables; materialize only the requested position chunk."""
    positions = tuple(positions)
    if not positions or len(positions) != len(set(positions)):
        raise ValueError("snapshot requires unique positions")
    allocation = cache._get_allocation(request_id)
    for position in positions:
        cache._validate_position(allocation, position)
    blocks = [
        [cache.get_block_table(request_id, depth)[p // cache.block_size] for p in positions]
        for depth in range(cache.max_loops)
    ]
    initialized = torch.tensor(
        [
            [
                [position in allocation.written[depth][layer] for position in positions]
                for layer in range(cache.num_layers)
            ]
            for depth in range(cache.max_loops)
        ],
        dtype=torch.bool,
        device="cpu",
    )
    if not bool(initialized.all()):
        raise RuntimeError("native snapshot contains an uninitialized KV position")
    block_ids = torch.tensor(blocks, dtype=torch.long, device=cache.device)
    layers = torch.arange(cache.num_layers, device=cache.device)
    offsets = torch.tensor([p % cache.block_size for p in positions], device=cache.device)
    keys = cache.key_cache[block_ids[:, None, :], layers[None, :, None], offsets[None, None, :]]
    values = cache.value_cache[block_ids[:, None, :], layers[None, :, None], offsets[None, None, :]]
    return KVSnapshot(positions, keys, values, initialized)


class ValidationEngine(LLMEngine):
    def __init__(self, model, *, fixtures, history_mode, **kwargs):
        self.fixtures = {row["fixture_id"]: row for row in fixtures}
        if len(self.fixtures) != len(fixtures):
            raise ValueError("duplicate fixture IDs")
        if history_mode not in ("teacher_forced", "live_gate"):
            raise ValueError("unsupported validation history mode")
        self.history_mode = history_mode
        self.traces = {key: [] for key in self.fixtures}
        self.gates = {}
        self.current_logits = {}
        super().__init__(model, **kwargs)

    def _should_exit(self, request):
        fixture = self.fixtures[request.request_id]
        if self.history_mode == "teacher_forced" and fixture["history_policy"] == "forced":
            target = fixture["forced_exit_depths"][len(request.generated_token_ids)]
            if request.loops_done > target:
                raise RuntimeError("native traversal exceeded its forced exit")
            return request.loops_done == target
        return super()._should_exit(request)

    def _update(self, batch, result):
        if batch.stage == Stage.RECURRENT:
            rows = [
                (item.request, len(item.request.generated_token_ids), probability)
                for item, probability in zip(batch.items, result)
            ]
            outputs = super()._update(batch, result)
            for request, index, probability in rows:
                gate = self.gates[request.request_id, index, request.loops_done]
                gate["probability"] = probability
                gate["cdf"] = 1.0 - request.remaining_probability
            return outputs
        if batch.stage != Stage.CODA:
            return super()._update(batch, result)
        returned = []
        for item, actual_token in zip(batch.items, result):
            request = item.request
            fixture = self.fixtures[request.request_id]
            index = len(request.generated_token_ids)
            inputs = fixture["continuation_input_ids"]
            if index > len(inputs):
                raise RuntimeError("native prediction count exceeds fixture")
            forced_token = (
                inputs[index]
                if self.history_mode == "teacher_forced" and index < len(inputs)
                else actual_token
            )
            gates = [
                dict(self.gates[request.request_id, index, depth])
                for depth in range(1, request.loops_done + 1)
            ]
            if index == 0:
                remaining = 1.0
                for gate in gates:
                    remaining *= 1.0 - gate["probability"]
                    gate["cdf"] = 1.0 - remaining
            self.traces[request.request_id].append(
                {
                    "position": request.position,
                    "output_index": index,
                    "exit_depth": request.loops_done,
                    "actual_token_id": actual_token,
                    "emitted_token_id": forced_token,
                    "history_sha256": history_hash(
                        request.prompt_token_ids + request.generated_token_ids
                    ),
                    "gate_logits": [gate["logit"] for gate in gates],
                    "gate_probabilities": [gate["probability"] for gate in gates],
                    "cumulative_probabilities": [gate["cdf"] for gate in gates],
                    "gate_usage": "full_depth_prefill_diagnostic"
                    if index == 0
                    else "actual_decode",
                    **self.current_logits.pop(request.request_id),
                }
            )
            returned.append(forced_token)
        return super()._update(batch, returned)


@contextmanager
def observe_native(engine, sink, *, on_completed_kv=None):
    """sink(metadata, tensor) receives selected query positions with borrowed values."""
    saved, handles = [], []
    context = []

    def replace(obj, name, value):
        saved.append((obj, name, name in vars(obj), vars(obj).get(name)))
        setattr(obj, name, value)

    def emit(operation, values, *, layer=None):
        if not context:
            raise RuntimeError("native diagnostic boundary has no row context")
        rows = context[-1]
        if len(rows) != values.shape[0]:
            raise RuntimeError("native diagnostic tensor rows differ from scheduling metadata")
        for index, (request_id, position, depth) in enumerate(rows):
            fixture = engine.fixtures[request_id]
            output_index = position - len(fixture["prompt_token_ids"]) + 1
            if not 0 <= output_index <= len(fixture["continuation_input_ids"]):
                continue
            request = engine.scheduler.requests[request_id]
            prefix = request.prompt_token_ids + request.generated_token_ids
            metadata = {
                "fixture_id": request_id,
                "operation": operation,
                "positions": [position],
                "depth": depth,
                "layer": layer,
                "output_index": output_index,
                "history_sha256": history_hash(prefix),
            }
            row = values[index]
            if operation == "gate_logits":
                logit = float(row.float().item())
                engine.gates[request_id, output_index, depth] = {
                    "logit": logit,
                    "probability": float(row.float().sigmoid().item()),
                    "cdf": None,
                }
            elif operation == "logits":
                selected = row.float().topk(2).values.cpu().tolist()
                engine.current_logits[request_id] = {"top_two_margin": selected[0] - selected[1]}
            sink(metadata, row)

    recurrent = engine.model.recurrent
    execute = engine.model_runner.execute
    write = engine.cache_manager.write
    attend = engine.cache_manager.attend
    finish = engine.scheduler.finish

    def observed_recurrent(hidden, request_ids, depths, positions, cache):
        context.append(
            [
                (key, int(position), int(depth) + 1)
                for key, position, depth in zip(request_ids, positions, depths)
            ]
        )
        try:
            return recurrent(hidden, request_ids, depths, positions, cache)
        finally:
            context.pop()

    def observed_execute(batch):
        if batch.stage != Stage.CODA:
            return execute(batch)
        context.append(
            [
                (item.request.request_id, item.request.position, item.request.loops_done)
                for item in batch.items
            ]
        )
        try:
            return execute(batch)
        finally:
            context.pop()

    def observed_write(layer, request_ids, depths, positions, key, value):
        result = write(layer, request_ids, depths, positions, key, value)
        emit("key", key, layer=layer)
        emit("value", value, layer=layer)
        return result

    def observed_attend(layer, request_ids, depths, positions, query):
        emit("query", query, layer=layer)
        return attend(layer, request_ids, depths, positions, query)

    def observed_finish(request, reason):
        if reason == "length" and on_completed_kv is not None:
            on_completed_kv(request.request_id, engine.cache_manager)
        return finish(request, reason)

    def hook(operation, layer=None, *, squeeze=False):
        def callback(module, args, output):
            emit(operation, output.squeeze(-1) if squeeze else output, layer=layer)

        return callback

    try:
        replace(engine.model, "recurrent", observed_recurrent)
        replace(engine.model_runner, "execute", observed_execute)
        replace(engine.cache_manager, "write", observed_write)
        replace(engine.cache_manager, "attend", observed_attend)
        replace(engine.scheduler, "finish", observed_finish)
        for layer_id, layer in enumerate(engine.model.model.layers):
            for module, operation in (
                (layer.input_layernorm, "attention_input"),
                (layer.self_attn, "attention_output"),
                (layer, "layer_output"),
            ):
                handles.append(module.register_forward_hook(hook(operation, layer_id)))
        for module, operation, squeeze in (
            (engine.model.model.norm, "loop_hidden", False),
            (engine.model.model.early_exit_gate, "gate_logits", True),
            (engine.model.lm_head, "logits", False),
        ):
            handles.append(module.register_forward_hook(hook(operation, squeeze=squeeze)))
        yield
    finally:
        for handle in reversed(handles):
            handle.remove()
        for obj, name, existed, previous in reversed(saved):
            if existed:
                setattr(obj, name, previous)
            else:
                delattr(obj, name)
