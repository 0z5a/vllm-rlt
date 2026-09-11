"""Benchmark-only routing replay; model execution and request lifecycle stay real."""

from vllm_lt.engine.llm_engine import LLMEngine
from vllm_lt.request import Request, Stage
from vllm_lt.sampling_params import SamplingParams


class ReplayEngine(LLMEngine):
    def __init__(self, model, *, replay: dict[str, dict], **engine_kwargs):
        # Validate before the base constructor allocates the physical KV pool.
        if not isinstance(replay, dict) or not replay:
            raise ValueError("replay must be a nonempty request mapping")
        self._replay = {}
        full_depth = model.config.total_ut_steps
        for request_id, trace in replay.items():
            if not isinstance(request_id, str) or not request_id:
                raise ValueError("replay request IDs must be nonempty strings")
            if not isinstance(trace, dict) or set(trace) != {"output_token_ids", "exit_depths"}:
                raise ValueError("replay requires output_token_ids and exit_depths only")
            tokens, depths = trace["output_token_ids"], trace["exit_depths"]
            if not isinstance(tokens, (list, tuple)) or not isinstance(depths, (list, tuple)):
                raise ValueError("replay token/depth histories must be arrays")
            if not tokens or len(tokens) != len(depths):
                raise ValueError("replay histories must have equal nonzero lengths")
            if any(
                type(token) is not int or not 0 <= token < model.config.vocab_size
                for token in tokens
            ):
                raise ValueError("replay token IDs must be integers within the vocabulary")
            if any(type(depth) is not int or not 2 <= depth <= full_depth for depth in depths):
                raise ValueError("replay depths must be integers between two and full depth")
            if depths[0] != full_depth:
                raise ValueError("the first replay output must use full-depth prefill")
            # Freeze caller-owned arrays before requests begin executing.
            self._replay[request_id] = (tuple(tokens), tuple(depths))
        super().__init__(model, **engine_kwargs)

    def add_request(self, request_id, prompt_token_ids, sampling_params=None):
        if request_id not in self._replay:
            raise ValueError(f"request {request_id!r} has no replay trace")
        params = sampling_params or SamplingParams()
        tokens, depths = self._replay[request_id]
        if params.max_tokens != len(tokens):
            raise ValueError("max_tokens must equal the replay history length")
        if not params.ignore_eos:
            raise ValueError("replay requires ignore_eos=True to consume its complete history")
        if params.temperature != 0:
            raise ValueError("M1 replay requires greedy sampling")
        maximum = params.max_loops or self.model.config.total_ut_steps
        if any(not params.min_loops <= depth <= maximum for depth in depths[1:]):
            raise ValueError("replay decode depths exceed the request's loop bounds")
        return super().add_request(request_id, prompt_token_ids, params)

    def _target(self, request: Request) -> tuple[int, int]:
        tokens, depths = self._replay[request.request_id]
        output_index = len(request.generated_token_ids)
        if output_index >= len(tokens):
            raise RuntimeError("replay history exhausted before request completion")
        return tokens[output_index], depths[output_index]

    def _should_exit(self, request: Request) -> bool:
        _, target_depth = self._target(request)
        if request.loops_done > target_depth:
            raise RuntimeError("recurrent execution exceeded the replay target depth")
        return request.loops_done == target_depth

    def _update(self, batch, result):
        if batch.stage == Stage.CODA:
            if len(result) != len(batch.items):
                raise RuntimeError("coda result length does not match the scheduled replay batch")
            forced = []
            for item in batch.items:
                token, depth = self._target(item.request)
                if item.request.loops_done != depth:
                    raise RuntimeError("coda reached a different depth from the replay trace")
                forced.append(token)
            # The unchanged runner has already executed coda, sampling and readback.
            result = forced
        outputs = super()._update(batch, result)
        for output in outputs:
            if output.finished:
                tokens, depths = self._replay[output.request_id]
                if tuple(output.token_ids) != tokens or tuple(output.exit_depths) != depths:
                    raise RuntimeError("completed request did not consume its exact replay trace")
        return outputs
