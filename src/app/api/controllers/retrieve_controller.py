"""retrieve_controller.py — POST /retrieve."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from ..dependencies import get_retrieval_service
from ..schemas.common import ErrorResponse
from ..schemas.retrieve import RetrieveRequest, RetrieveResponse
from ..services.retrieval_service import RetrievalService

router = APIRouter(tags=["retrieval"])


@router.post(
    "/retrieve",
    response_model=RetrieveResponse,
    summary="Semantic search over guideline passages",
    description=(
        "Retrieves the top-k NICE/NHS/WHO guideline passages for a free-text query, "
        "or for a severity + flagged markers (a query is then built and biomarker-"
        "boosted). Each passage returns its citation and provenance."
    ),
    responses={422: {"model": ErrorResponse}},
)
def retrieve(
    payload: RetrieveRequest,
    service: RetrievalService = Depends(get_retrieval_service),
) -> RetrieveResponse:
    return service.retrieve(payload)
