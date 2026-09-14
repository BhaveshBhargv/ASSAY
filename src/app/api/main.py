"""Creates the FastAPI app.

Swagger UI is served at /docs, ReDoc at /redoc and the schema at /openapi.json.

Run: uvicorn app.api.main:app --app-dir src --reload
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .controllers import routers
from .core.config import settings
from .core.errors import register_exception_handlers
from .core.logging import RequestContextMiddleware, configure_logging
from .dependencies import warm_up

log = logging.getLogger("assay.api")

_TAGS_METADATA = [
    {"name": "upload", "description": "Parse blood report files (CSV/JSON/PDF) for review."},
    {"name": "assessment", "description": "Rule engine + Random Forest risk assessment."},
    {"name": "recommendation", "description": "Full grounded lifestyle-recommendation pipeline."},
    {"name": "retrieval", "description": "Semantic search over NICE/NHS/WHO guideline passages."},
    {"name": "meta", "description": "Model capabilities, provenance, and health."},
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.log_level)
    log.info("starting %s v%s — warming models…", settings.app_name, settings.version)
    warm_up()
    log.info("ready")
    yield
    log.info("shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description=settings.description,
        lifespan=lifespan,
        openapi_tags=_TAGS_METADATA,
        contact={"name": "Bhavesh Bhargava — MSc Advanced Data Science"},
        license_info={"name": "Educational project — not for clinical use"},
    )

    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)
    for r in routers:
        app.include_router(r)

    @app.get("/health", tags=["meta"], summary="Liveness/readiness probe")
    def health() -> dict:
        return {"status": "ok", "app": settings.app_name, "version": settings.version}

    return app


app = create_app()
