"""GET /model-info"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from ..dependencies import get_model_info_service
from ..schemas.model_info import ModelInfoResponse
from ..services.model_info_service import ModelInfoService

router = APIRouter(tags=["meta"])


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Model capabilities and provenance",
    description=(
        "Reports the ruleset version, the Random Forest details (algorithm, classes, "
        "features, CV metrics), the RAG index (embedder, size), the default LLM "
        "provider, and the full biomarker catalog with valid codes."
    ),
)
def model_info(
    service: ModelInfoService = Depends(get_model_info_service),
) -> ModelInfoResponse:
    return service.info()
