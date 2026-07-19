"""
errors.py — typed application errors + JSON exception handlers.

Every error leaves the API in the same envelope:
    {"error": {"type": "...", "message": "...", "detail": ...}, "request_id": "..."}
so clients can branch on `type` and correlate with server logs via `request_id`.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .logging import request_id_ctx

log = logging.getLogger("assay.api")


class AppError(Exception):
    """Base for expected, client-facing errors."""
    status_code = status.HTTP_400_BAD_REQUEST
    error_type = "app_error"

    def __init__(self, message: str, detail=None):
        super().__init__(message)
        self.message = message
        self.detail = detail


class ValidationAppError(AppError):
    status_code = 422  # Unprocessable Content
    error_type = "validation_error"


class ParsingError(AppError):
    status_code = 422  # Unprocessable Content
    error_type = "parsing_error"


class PayloadTooLargeError(AppError):
    status_code = 413  # Content Too Large
    error_type = "payload_too_large"


class ModelUnavailableError(AppError):
    """A required artefact (RF model, index) isn't loaded."""
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    error_type = "model_unavailable"


class LLMUnavailableError(AppError):
    """The language model provider could not be reached or returned garbage."""
    status_code = status.HTTP_502_BAD_GATEWAY
    error_type = "llm_unavailable"


def _envelope(error_type: str, message: str, detail=None) -> dict:
    return {
        "error": {"type": error_type, "message": message, "detail": detail},
        "request_id": request_id_ctx.get(),
    }


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError):
        log.warning("%s: %s", exc.error_type, exc.message)
        return JSONResponse(status_code=exc.status_code,
                            content=_envelope(exc.error_type, exc.message, exc.detail))

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content=_envelope("validation_error", "Request validation failed",
                              jsonable_encoder(exc.errors())),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException):
        return JSONResponse(status_code=exc.status_code,
                            content=_envelope("http_error", str(exc.detail)))

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):
        log.exception("unhandled error: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_envelope("internal_error",
                              "An unexpected error occurred. Check server logs with the request_id."),
        )
