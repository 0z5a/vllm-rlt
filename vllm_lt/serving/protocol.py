"""The deliberately small completions contract, independent of HTTP and device code."""

import json
import math
from dataclasses import dataclass, fields

from vllm_lt.sampling_params import SamplingParams


class ServingError(Exception):
    def __init__(self, message, status=500, code="server_error"):
        super().__init__(message)
        self.status = status
        self.code = code

    def as_dict(self):
        return {"error": {"message": str(self), "type": self.code, "code": self.code}}


@dataclass(frozen=True)
class CompletionRequest:
    prompt: str
    params: SamplingParams
    stream: bool = False
    include_usage: bool = False

    @classmethod
    def parse(cls, body, model):
        if not isinstance(body, dict):
            raise ValueError("request body must be a JSON object")
        params_keys = {field.name for field in fields(SamplingParams)}
        neutral = {
            "n": 1,
            "best_of": 1,
            "echo": False,
            "logprobs": None,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "stop": None,
            "suffix": None,
            "logit_bias": None,
        }
        allowed = params_keys | neutral.keys() | {"model", "prompt", "stream", "stream_options"}
        if unknown := body.keys() - allowed:
            raise ValueError(f"unsupported fields: {', '.join(sorted(unknown))}")
        if not isinstance(body.get("model"), str) or not body["model"]:
            raise ValueError("model must be a nonempty string")
        if body["model"] != model:
            raise ServingError(f"model must be {model!r}", 404, "model_not_found")
        if not isinstance(body.get("prompt"), str):
            raise ValueError("prompt must be one string")
        try:
            body["prompt"].encode("utf-8")
        except UnicodeError as exc:
            raise ValueError("prompt must contain valid Unicode scalar values") from exc
        for key, default in neutral.items():
            value = body.get(key, default)
            valid_type = type(value) is type(default)
            if type(default) is float:
                valid_type = type(value) in (int, float)
            if not valid_type or value != default:
                raise ValueError(f"only {key}={default!r} is supported")
        stream = body.get("stream", False)
        if type(stream) is not bool:
            raise ValueError("stream must be a boolean")
        options = body.get("stream_options")
        include_usage = False
        if options is not None:
            if not stream or not isinstance(options, dict):
                raise ValueError("stream_options requires stream=true and an object")
            if options.keys() - {"include_usage"}:
                raise ValueError("only stream_options.include_usage is supported")
            include_usage = options.get("include_usage", False)
            if type(include_usage) is not bool:
                raise ValueError("include_usage must be a boolean")
        params = {key: body[key] for key in params_keys & body.keys()}
        for key in ("temperature", "top_p", "exit_threshold"):
            if key in params and (
                type(params[key]) not in (int, float) or not math.isfinite(params[key])
            ):
                raise ValueError(f"{key} must be a finite number")
        if "ignore_eos" in params and type(params["ignore_eos"]) is not bool:
            raise ValueError("ignore_eos must be a boolean")
        return cls(body["prompt"], SamplingParams(**params), stream, include_usage)


class IncrementalText:
    """Stable text prefixes for Ouro's byte-level tokenizer.

    Cleanup is disabled, matching the released tokenizer. Incomplete UTF-8
    prefixes are withheld until a later token completes them (or final flush).
    Each sampled ID still produces an event even when its text delta is empty.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.emitted = ""

    def decode(self, token_ids, finished):
        text = self.tokenizer.decode(
            token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        stable = text if finished else text.rstrip("\ufffd")
        if not stable.startswith(self.emitted):
            raise ValueError(
                "tokenizer rewrote emitted text; only byte-level decoding is supported"
            )
        delta = stable[len(self.emitted) :]
        self.emitted = stable
        return delta


def usage(prompt_tokens, completion_tokens):
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def completion(request_id, model, created, text, finish_reason, token_usage=None, *, summary=False):
    result = {
        "id": request_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": []
        if summary
        else [{"text": text, "index": 0, "logprobs": None, "finish_reason": finish_reason}],
    }
    if token_usage is not None:
        result["usage"] = token_usage
    return result


def sse(data):
    return ("data: " + json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n\n").encode()
