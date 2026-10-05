"""Document-paired quality intervals; tokens are not independent observations."""

import math
import random
from dataclasses import dataclass

import torch
from torch.nn import functional as F


def next_token_nll(
    logits: torch.Tensor, token_ids: torch.Tensor, valid_mask: torch.Tensor
) -> tuple[torch.Tensor, int]:
    valid = valid_mask[:, :-1] & valid_mask[:, 1:]
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
