"""Document-paired quality intervals; tokens are not independent observations."""

import math
import random
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.models.hrm_text import HrmTextForCausalLM
from vllm_rlt.models.loopformer import LoopFormerForCausalLM
from vllm_rlt.models.nanbeige import NanbeigeForCausalLM
from vllm_rlt.models.ouro import OuroForCausalLM


@torch.inference_mode()
def native_window_nll(
    model: OuroForCausalLM | NanbeigeForCausalLM | HrmTextForCausalLM | LoopFormerForCausalLM,
    tokens: torch.Tensor,
    loops: int,
    cache: KVCacheManager,
) -> tuple[float, int]:
    """Teacher-forced NLL through the actual native core, including packed weights.

    The caller reuses an empty cache with the registered deployment backend.
    Fixed loops bypass halting. This is quality evaluation, not a timed engine cohort.
    """
    if isinstance(model, HrmTextForCausalLM):
        raise ValueError("HRM requires an explicit prefix and continuation scoring boundary")
    if isinstance(model, LoopFormerForCausalLM) and loops != 8:
        raise ValueError("LoopFormer requires its complete fixed recurrence")
    if tokens.ndim != 1 or len(tokens) < 2 or not 1 <= loops <= model.config.total_ut_steps:
        raise ValueError("expected a token window and a supported fixed recurrence")
    if cache.num_used_blocks or not cache.allocate("quality", len(tokens)):
        raise ValueError("quality evaluation requires an empty cache with enough capacity")
    try:
        hidden = model.prelude(tokens)
        requests, positions = ["quality"] * len(tokens), list(range(len(tokens)))
        for loop in range(loops):
            hidden, _ = model.recurrent(
                hidden, requests, [loop] * len(tokens), positions, cache, compute_gate=False
            )
        logits = model.coda(hidden[:-1]).float()
        loss = F.cross_entropy(logits, tokens[1:], reduction="sum")
        if not torch.isfinite(loss):
            raise ValueError("nonfinite native NLL")
        return float(loss), len(tokens) - 1
    finally:
        cache.free("quality")


@torch.inference_mode()
def native_continuation_nll(
    model: OuroForCausalLM | NanbeigeForCausalLM | HrmTextForCausalLM | LoopFormerForCausalLM,
    tokens: torch.Tensor,
    prefix_length: int,
    loops: int,
    cache: KVCacheManager,
) -> tuple[float, int]:
    """Score only causal continuation after one fixed, atomically admitted prefix.

    HRM's prefix queries can attend to the complete prompt. No continuation
    target enters that bidirectional prefix or the cache before it is scored.
    Other families retain causal prefill. This is native quality, not throughput.
    """
    if tokens.ndim != 1 or not 1 <= prefix_length < len(tokens):
        raise ValueError("expected a nonempty prefix followed by continuation targets")
    if not 1 <= loops <= model.config.total_ut_steps:
        raise ValueError("unsupported fixed recurrence")
    fixed = isinstance(model, (HrmTextForCausalLM, LoopFormerForCausalLM))
    if fixed and loops != model.config.total_ut_steps:
        raise ValueError("this family requires its complete fixed recurrence")
    if cache.num_used_blocks or not cache.allocate("quality", len(tokens)):
        raise ValueError("quality evaluation requires an empty cache with enough capacity")
    try:
        loss = torch.zeros((), device=tokens.device)
        for position in range(prefix_length - 1, len(tokens) - 1):
            positions = list(range(prefix_length)) if position == prefix_length - 1 else [position]
            hidden = model.prelude(tokens[positions])
            read_lengths = (
                [prefix_length] * len(positions)
                if position == prefix_length - 1
                and isinstance(model, HrmTextForCausalLM)
                and model.config.prefix_lm
                else None
            )
            for loop in range(loops):
                batch = cache._prepare_batch(
                    ["quality"] * len(positions),
                    [loop] * len(positions),
                    positions,
                    read_lengths=read_lengths,
                )
                hidden, _ = model.recurrent_prepared(hidden, batch, cache, compute_gate=False)
            logits = model.coda(hidden[-1:]).float()
            loss += F.cross_entropy(logits, tokens[position + 1 : position + 2], reduction="sum")
            for row in positions:
                cache.finalize_token("quality", row, loops - 1)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite continuation NLL")
        return float(loss), len(tokens) - prefix_length
    finally:
        cache.free("quality")


def next_token_nll(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    target_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    valid = valid_mask[:, :-1] & valid_mask[:, 1:]
    if target_mask is not None:
        if target_mask.shape != valid.shape or target_mask.dtype != torch.bool:
            raise ValueError("target mask must align with next-token positions")
        valid = valid & target_mask
    count = int(valid.sum())
    if count == 0:
        raise ValueError("no valid next-token targets")
    losses = F.cross_entropy(
        logits[:, :-1].float().flatten(0, 1), token_ids[:, 1:].flatten(), reduction="none"
    ).reshape_as(valid)
    return losses[valid].sum(), count


@dataclass(frozen=True)
class DocumentNLL:
    document_id: str
    tokens: int
    baseline_nll: float
    candidate_nll: float


@dataclass(frozen=True)
class QuestionCorrectness:
    question_id: str
    baseline: bool
    candidate: bool


def accuracy_interval(
    questions: list[QuestionCorrectness], *, seed: int = 20261005, resamples: int = 10000
) -> tuple[float, float]:
    """Return BF16-minus-candidate accuracy drop and its paired one-sided95% upper bound."""
    if len(questions) < 2 or len({row.question_id for row in questions}) != len(questions):
        raise ValueError("aggregate each original question once; need two distinct questions")
    if resamples < 1000:
        raise ValueError("at least1000 paired bootstrap resamples are required")
    differences = [int(row.baseline) - int(row.candidate) for row in questions]
    rng = random.Random(seed)
    samples = sorted(
        sum(rng.choices(differences, k=len(differences))) / len(differences)
        for _ in range(resamples)
    )
    return sum(differences) / len(differences), samples[math.ceil(0.95 * resamples) - 1]


def ppl_interval(
    documents: list[DocumentNLL], *, seed: int = 20261005, resamples: int = 10000
) -> tuple[float, float]:
    if len(documents) < 2 or len({row.document_id for row in documents}) != len(documents):
        raise ValueError("aggregate all windows per document; need at least two documents")
    if resamples < 1000 or any(
        row.tokens <= 0
        or not all(math.isfinite(x) and x >= 0 for x in (row.baseline_nll, row.candidate_nll))
        for row in documents
    ):
        raise ValueError("invalid document NLL, token count, or resampling budget")

    def ratio(rows: list[DocumentNLL]) -> float:
        return math.exp(
            sum(row.candidate_nll - row.baseline_nll for row in rows)
            / sum(row.tokens for row in rows)
        )

    rng = random.Random(seed)
    samples = sorted(ratio(rng.choices(documents, k=len(documents))) for _ in range(resamples))
    return ratio(documents), samples[math.ceil(0.95 * resamples) - 1]
