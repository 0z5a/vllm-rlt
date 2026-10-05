"""Token-normalized Q0 updates with scale audits and exact optimizer resumption."""

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from .adapters.loopformer import LoopFormerAdapter
from .adapters.nanbeige import NanbeigeAdapter
from .adapters.ouro import OuroAdapter
from .quality import next_token_nll


@dataclass(frozen=True)
class Q0Config:
    loops: int = 4
    tokens_per_update: int = 8192
    learning_rate: float = 1e-4
    max_grad_norm: float = 1.0
    kl_weight: float = 0.0
    trajectory_weight: float = 0.0
    temperature: float = 1.0


@dataclass(frozen=True)
class TrainingBatch:
    token_ids: torch.Tensor
    valid: torch.Tensor


class Q0Trainer:
    """Only log-scales are optimized; all recurrent weights remain frozen.

    The straight-through rule is defined in fp8_fake_quant. Shared calls receive
    their ordinary chain-rule sum; only effective targets normalize CE/KL, while
    trajectory loss averages coordinates and loops. No second division by the
    number of shared calls is applied to the complete recurrent derivative.
    """

    def __init__(
        self, student: OuroAdapter | NanbeigeAdapter | LoopFormerAdapter, config: Q0Config
    ) -> None:
        if not student.quantized or any(p.requires_grad for p in student.model.parameters()):
            raise ValueError("Q0 requires attached scale modules and a frozen model")
        if config.tokens_per_update < 2 or config.learning_rate <= 0 or config.max_grad_norm <= 0:
            raise ValueError("token budget, learning rate, and clipping norm must be positive")
        if min(config.kl_weight, config.trajectory_weight) < 0 or config.temperature <= 0:
            raise ValueError("invalid loss coefficients or temperature")
        self.student = student
        self.teacher = student.reference()
        self.config = config
        self.parameters = dict(student.quantized.named_parameters())
        self.optimizer = torch.optim.AdamW(
            self.parameters.values(), lr=config.learning_rate, weight_decay=0
        )
        self.updates = 0
        self.cumulative_tokens = 0
        self.data_position = 0

    def step(
        self, batches: list[TrainingBatch]
    ) -> dict[str, int | float | bool | dict[str, float]]:
        config = self.config
        tokens = sum(int(batch.valid.sum()) for batch in batches)
        targets = sum(int((batch.valid[:, :-1] & batch.valid[:, 1:]).sum()) for batch in batches)
        if tokens != config.tokens_per_update or targets == 0:
            raise ValueError("an update must match the registered effective token budget")
        before = {name: value.detach().clone() for name, value in self.parameters.items()}
        self.optimizer.zero_grad(set_to_none=True)
        totals = dict(ce=0.0, kl=0.0, trajectory=0.0)
        for batch in batches:
            result = self.student(batch.token_ids, batch.valid, config.loops)
            nll, count = next_token_nll(result.logits, batch.token_ids, batch.valid)
            loss = nll / targets
            totals["ce"] += float(loss.detach())
            if config.kl_weight or config.trajectory_weight:
                with torch.no_grad():
                    teacher = self.teacher(batch.token_ids, batch.valid, config.loops)
                mask = batch.valid[:, :-1] & batch.valid[:, 1:]
                if config.kl_weight:
                    temperature = config.temperature
                    student_logp = (result.logits[:, :-1].float() / temperature).log_softmax(-1)
                    teacher_logp = (teacher.logits[:, :-1].float() / temperature).log_softmax(-1)
                    kl = F.kl_div(student_logp, teacher_logp, reduction="none", log_target=True)
                    kl = kl.sum(-1)[mask].sum() * temperature**2 / targets
                    totals["kl"] += float(kl.detach())
                    loss = loss + config.kl_weight * kl
                if config.trajectory_weight:
                    trajectory = (
                        sum(
                            (actual.float() - expected.float()).square()[batch.valid].mean()
                            for actual, expected in zip(result.states, teacher.states, strict=True)
                        )
                        / config.loops
                    )
                    trajectory = trajectory * count / targets
                    totals["trajectory"] += float(trajectory.detach())
                    loss = loss + config.trajectory_weight * trajectory
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            loss.backward()
        gradients = {}
        for name, parameter in self.parameters.items():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise ValueError(f"missing or nonfinite scale gradient: {name}")
            gradients[name] = float(parameter.grad.norm())
        if not any(gradients.values()):
            raise ValueError("all scale gradients are zero")
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            list(self.parameters.values()), config.max_grad_norm, error_if_nonfinite=True
        )
        self.optimizer.step()
        updates = {
            name: float((parameter.detach() - before[name]).norm())
            for name, parameter in self.parameters.items()
        }
        if not any(updates.values()):
            raise ValueError("optimizer did not update any scale")
        self.updates += 1
        self.cumulative_tokens += tokens
        self.data_position += len(batches)
        return dict(
            update=self.updates,
            input_tokens=tokens,
            supervised_targets=targets,
            cumulative_tokens=self.cumulative_tokens,
            losses=totals,
            gradient_norm=float(gradient_norm),
            clipped=bool(gradient_norm > config.max_grad_norm),
            scale_gradient_norms=gradients,
            scale_update_norms=updates,
        )

    def save(self, destination: Path) -> None:
        device = next(iter(self.parameters.values())).device
        torch.save(
            dict(
                config=asdict(self.config),
                quantized=self.student.quantized.state_dict(),
                optimizer=self.optimizer.state_dict(),
                updates=self.updates,
                cumulative_tokens=self.cumulative_tokens,
                data_position=self.data_position,
                cpu_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            ),
            destination,
        )

    def resume(self, source: Path) -> None:
        device = next(iter(self.parameters.values())).device
        checkpoint = torch.load(source, map_location=device, weights_only=True)
        if checkpoint["config"] != asdict(self.config):
            raise ValueError("resumption requires the original registered training configuration")
        self.student.quantized.load_state_dict(checkpoint["quantized"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.updates = checkpoint["updates"]
        self.cumulative_tokens = checkpoint["cumulative_tokens"]
        self.data_position = checkpoint["data_position"]
        torch.set_rng_state(checkpoint["cpu_rng"].cpu())
        if checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state(checkpoint["cuda_rng"].cpu(), device)
