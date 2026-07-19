"""
logging.py — structured logging + per-request correlation id.

A `request_id` context variable is injected into every log record so a request's
lines can be traced end to end. `RequestContextMiddleware` assigns the id, logs
the request/response with latency, and echoes it back in the `X-Request-ID` header.
"""
from __future__ import annotations

import logging
import time
import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")
log = logging.getLogger("assay.api")


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get()
        return True


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s [%(request_id)s] %(name)s: %(message)s"))
    handler.addFilter(_RequestIdFilter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
    # quieten noisy third parties
    for noisy in ("httpx", "urllib3", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        token = request_id_ctx.set(rid)
        start = time.perf_counter()
        try:
            response = await call_next(request)
            elapsed = (time.perf_counter() - start) * 1000
            response.headers["X-Request-ID"] = rid
            log.info("%s %s -> %s (%.0fms)", request.method, request.url.path,
                     response.status_code, elapsed)
            return response
        except Exception:
            elapsed = (time.perf_counter() - start) * 1000
            log.exception("%s %s failed after %.0fms", request.method, request.url.path, elapsed)
            raise
        finally:
            request_id_ctx.reset(token)
