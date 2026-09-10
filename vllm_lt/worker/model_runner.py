"""Stage execution; the scheduler never invokes a whole-token model forward."""

import torch

from vllm_lt.core.scheduler import SchedulerOutput
from vllm_lt.request import Request, Stage


class ModelRunner:
    def __init__(self, model, cache_manager):
        self.model = model.eval()
        self.cache_manager = cache_manager
        self.device = next(model.parameters()).device

    @torch.inference_mode()
    def execute(self, batch: SchedulerOutput):
        requests = [item.request for item in batch.items]
        if batch.stage == Stage.PREFILL:
            ids, positions, tokens = [], [], []
            for item in batch.items:
                start, count, request = item.token_start, item.token_count, item.request
                ids.extend([request.request_id] * count)
                positions.extend(range(start, start + count))
                tokens.extend(request.prompt_token_ids[start : start + count])
            hidden = self.model.prelude(torch.tensor(tokens, device=self.device, dtype=torch.long))
            # Every prompt token reaches the model's full depth, independently of decode policy.
            for depth in range(self.model.config.total_ut_steps):
                hidden, _ = self.model.recurrent(
                    hidden, ids, [depth] * len(ids), positions, self.cache_manager
                )
            offset = 0
            for item in batch.items:
                offset += item.token_count
                item.request.hidden_state = hidden[offset - 1].clone()
            return None
        if batch.stage == Stage.PRELUDE:
            token_ids = torch.tensor(
                [r.input_token_id for r in requests], device=self.device, dtype=torch.long
            )
            hidden = self.model.prelude(token_ids)
            for request, state in zip(requests, hidden):
                request.hidden_state = state
            return None
        hidden = torch.stack([r.hidden_state for r in requests])
        if batch.stage == Stage.RECURRENT:
            hidden, gate_logits = self.model.recurrent(
                hidden,
                [r.request_id for r in requests],
                [r.loops_done for r in requests],
                [r.position for r in requests],
                self.cache_manager,
            )
            for request, state in zip(requests, hidden):
                request.hidden_state = state
            # This is explicitly synchronous. A stock gate cannot act as the paper's lookahead gate.
            return gate_logits.float().sigmoid().cpu().tolist()
        if batch.stage == Stage.CODA:
            logits = self.model.coda(hidden)
            return [self._sample(row, request) for row, request in zip(logits, requests)]
        raise ValueError(f"unsupported execution stage {batch.stage}")

    def _sample(self, logits: torch.Tensor, request: Request) -> int:
        params = request.sampling_params
        if params.temperature == 0:
            return int(logits.argmax().item())
        logits = logits.float() / params.temperature
        if params.top_k > 0:
            threshold = logits.topk(min(params.top_k, logits.numel())).values[-1]
            logits = logits.masked_fill(logits < threshold, -torch.inf)
        if params.top_p < 1:
            sorted_logits, indices = logits.sort(descending=True)
            cumulative = sorted_logits.softmax(-1).cumsum(-1)
            remove = cumulative > params.top_p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            logits = logits.scatter(0, indices, sorted_logits.masked_fill(remove, -torch.inf))
        if request.generator is None:
            request.generator = torch.Generator(device=self.device).manual_seed(params.seed)
        return int(torch.multinomial(logits.softmax(-1), 1, generator=request.generator).item())
