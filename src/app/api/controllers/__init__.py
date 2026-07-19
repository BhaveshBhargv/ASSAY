"""controllers — FastAPI routers (one per resource)."""
from __future__ import annotations

from .model_info_controller import router as model_info_router
from .predict_controller import router as predict_router
from .recommend_controller import router as recommend_router
from .retrieve_controller import router as retrieve_router
from .upload_controller import router as upload_router

routers = [upload_router, predict_router, recommend_router, retrieve_router, model_info_router]

__all__ = ["routers"]
