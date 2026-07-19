"""
api — FastAPI backend for the blood-test lifestyle recommender.

Layered interface-adapter around the Phase 3-6 engine:
    controllers (HTTP routers) -> services (business logic) -> repositories (artefacts)
wired with FastAPI dependency injection. See `app.api.main:create_app`.
"""
from __future__ import annotations

__all__ = ["create_app"]


def create_app():  # lazy so importing the package doesn't pull FastAPI eagerly
    from .main import create_app as _factory
    return _factory()
