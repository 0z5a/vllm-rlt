"""Bounded HTTP completions with SSE token events and explicit readiness."""

import asyncio
import logging
import time

from aiohttp import web

from vllm_lt.serving.protocol import CompletionRequest, ServingError, completion, sse, usage
from vllm_lt.serving.worker import EngineWorker

logger = logging.getLogger(__name__)
WORKER = web.AppKey("worker", EngineWorker)


def error_response(error):
    headers = {"Retry-After": "1"} if error.status in (429, 503) else None
    return web.json_response(error.as_dict(), status=error.status, headers=headers)


def create_app(
    factory,
    *,
    model="ByteDance/Ouro-1.4B",
    max_requests=64,
    output_buffer=32,
    max_body_bytes=1024 * 1024,
    request_timeout=300,
    write_timeout=10,
    shutdown_timeout=30,
):
    for name, value in (
        ("max_body_bytes", max_body_bytes),
        ("request_timeout", request_timeout),
        ("write_timeout", write_timeout),
        ("shutdown_timeout", shutdown_timeout),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    worker = EngineWorker(factory, max_requests=max_requests, output_buffer=output_buffer)
    inflight = 0

    @web.middleware
    async def limits(request, handler):
        nonlocal inflight
        if request.path != "/v1/completions":
            return await handler(request)
        if not worker.ready:
            return error_response(ServingError("engine is not ready", 503, "not_ready"))
        if inflight >= max_requests:
            return error_response(ServingError("request capacity exhausted", 429, "overloaded"))
        inflight += 1
        try:
            return await asyncio.wait_for(handler(request), request_timeout)
        except asyncio.TimeoutError:
            # The handler aborts a prepared stream in its cancellation path.
            return error_response(ServingError("request deadline exceeded", 504, "timeout"))
        except web.HTTPRequestEntityTooLarge:
            return error_response(
                ServingError("request body is too large", 413, "invalid_request_error")
            )
        finally:
            inflight -= 1

    app = web.Application(middlewares=[limits], client_max_size=max_body_bytes)
    app[WORKER] = worker

    async def health(request):
        return web.json_response(
            {"status": "ready" if worker.ready else "unavailable"},
            status=200 if worker.ready else 503,
        )

    async def models(request):
        return web.json_response(
            {
                "object": "list",
                "data": [{"id": model, "object": "model", "created": 0, "owned_by": "vllm-lt"}],
            }
        )

    async def complete(request):
        channel = None
        response = None
        try:
            try:
                body = await request.json()
                spec = CompletionRequest.parse(body, model)
            except (ValueError, TypeError, UnicodeError) as exc:
                raise ServingError(str(exc), 400, "invalid_request_error") from exc
            channel = worker.submit(spec)
            created = int(time.time())
            # Admission/tokenizer errors and first-step failures can still use
            # an HTTP error status. Do not send an empty synthetic first token.
            event = await channel.receive()
            if spec.stream:
                response = web.StreamResponse(
                    headers={
                        "Content-Type": "text/event-stream",
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                    }
                )
                await response.prepare(request)
            texts = []
            while True:
                if spec.stream:
                    data = completion(
                        channel.request_id, model, created, event.text, event.finish_reason
                    )
                    if spec.include_usage:
                        data["usage"] = None
                    await asyncio.wait_for(response.write(sse(data)), write_timeout)
                else:
                    texts.append(event.text)
                if event.finish_reason is not None:
                    break
                event = await channel.receive()
            token_usage = usage(event.prompt_tokens, event.completion_tokens)
            logger.info(
                "generation %s: prompt_tokens=%d completion_tokens=%d finish_reason=%s",
                channel.request_id,
                event.prompt_tokens,
                event.completion_tokens,
                event.finish_reason,
            )
            if not spec.stream:
                return web.json_response(
                    completion(
                        channel.request_id,
                        model,
                        created,
                        "".join(texts),
                        event.finish_reason,
                        token_usage,
                    )
                )
            if spec.include_usage:
                data = completion(
                    channel.request_id, model, created, "", None, token_usage, summary=True
                )
                await asyncio.wait_for(response.write(sse(data)), write_timeout)
            await asyncio.wait_for(response.write(b"data: [DONE]\n\n"), write_timeout)
            await asyncio.wait_for(response.write_eof(), write_timeout)
            return response
        except ServingError as exc:
            logger.info(
                "request %s failed: %s", channel.request_id if channel else "unadmitted", exc
            )
            if response is None:
                return error_response(exc)
            # The pinned client ignores in-band error objects after a token.
            # Abort chunked HTTP without a terminating chunk so it records a
            # transport failure, rather than a successful partial completion.
            if request.transport is not None:
                request.transport.abort()
            return response
        except (asyncio.CancelledError, asyncio.TimeoutError, ConnectionError):
            if response is not None and request.transport is not None:
                request.transport.abort()
            raise
        finally:
            if channel is not None:
                worker.release(channel)

    async def start(app):
        worker.start()

    async def stop(app):
        await worker.close(shutdown_timeout)

    app.router.add_get("/health", health)
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/completions", complete)
    app.on_startup.append(start)
    app.on_shutdown.append(stop)
    return app
