"""FastAPI backend for the recommender: controllers -> services -> repositories.

The app itself is built in app.api.main.create_app.
"""
from __future__ import annotations

__all__ = ["create_app"]


def create_app():  # imported here so importing the package doesn't load FastAPI
    from .main import create_app as _factory
    return _factory()
