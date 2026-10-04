"""Measure output collection only; this is not a model or full RL benchmark."""

import argparse
import json
import platform
import time

import torch

from vllm_rlt.request import FinishReason, Request, RequestOutput, Stage
from vllm_rlt.sampling_params import SamplingParams


def measure(length: int, final_only: bool, device: str) -> dict:
    request = Request("bench", [2, 3], SamplingParams(max_tokens=length, logprobs=0))
    scores = torch.linspace(-8, -0.5, length, device=device).unbind()
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    exports, elements = 0, 0
    output = None
    for index, score in enumerate(scores):
        request.generated_token_ids.append(7)
        request.exit_depths.append(4)
        request.log_probs.append(score)
        request.stage = Stage.FINISHED if index + 1 == length else Stage.CODA
        if request.stage == Stage.FINISHED:
            request.finish_reason = FinishReason.LENGTH
        if not final_only or request.stage == Stage.FINISHED:
            output = RequestOutput.from_request(request)
            exports += 1
            elements += len(output.log_probs)
    if device == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    assert output is not None and output.finished
    assert len(output.log_probs) == len(output.token_ids) == len(output.exit_depths) == length
    return {
        "seconds": seconds,
        "exports": exports,
        "score_elements": elements,
        "last_score": output.log_probs[-1],
        "tokens": output.token_ids,
        "scores": output.log_probs,
        "depths": output.exit_depths,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 512, 2048, 8192])
    args = parser.parse_args()
    torch.set_num_threads(1)
    for length in args.lengths:
        for arm in (False, True):
            measure(length, arm, args.device)
        reference = None
        for position, arm in enumerate((False, True, True, False)):
            result = measure(length, arm, args.device)
            trace = [result.pop(key) for key in ("tokens", "scores", "depths")]
            if reference is None:
                reference = trace
            assert trace == reference
            print(
                json.dumps(
                    {
                        "run_id": args.run_id,
                        "position": position,
                        "arm": "final" if arm else "cumulative",
                        "length": length,
                        "scope": "output_materialization_only",
                        "device": args.device,
                        "torch": torch.__version__,
                        "host": platform.platform(),
                        **result,
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
