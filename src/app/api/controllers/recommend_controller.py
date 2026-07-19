"""recommend_controller.py — POST /recommend."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from ..dependencies import get_recommendation_service
from ..schemas.common import ErrorResponse
from ..schemas.recommend import RecommendRequest, RecommendResponse
from ..services.recommendation_service import RecommendationService

router = APIRouter(tags=["recommendation"])


@router.post(
    "/recommend",
    response_model=RecommendResponse,
    summary="Grounded lifestyle recommendations",
    description=(
        "The full pipeline: assess → retrieve NICE/NHS/WHO evidence → generate with "
        "the LLM → verify. Every recommendation carries a rationale and cites the "
        "retrieved evidence; ungrounded or diagnostic output is stripped by the "
        "guards. Requires a reachable LLM provider (local Ollama by default). "
        "**Not a diagnosis.**"
    ),
    responses={
        422: {"model": ErrorResponse},
        502: {"model": ErrorResponse, "description": "Language model unreachable or unusable"},
    },
)
def recommend(
    payload: RecommendRequest,
    service: RecommendationService = Depends(get_recommendation_service),
) -> RecommendResponse:
    return service.recommend(payload)
