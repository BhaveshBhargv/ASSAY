"""POST /predict"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from ..dependencies import get_assessment_service
from ..schemas.common import ErrorResponse
from ..schemas.predict import PredictRequest, PredictResponse
from ..services.assessment_service import AssessmentService

router = APIRouter(tags=["assessment"])


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Risk assessment from a blood panel",
    description=(
        "Runs the clinical rule engine and the Random Forest, then fuses them into "
        "one **safety-dominant** severity (the model can only escalate the rules, "
        "never soften them). Returns per-biomarker results, flagged markers, and "
        "clinician-signpost markers. Does not call the language model. **Not a diagnosis.**"
    ),
    responses={422: {"model": ErrorResponse}},
)
def predict(
    payload: PredictRequest,
    service: AssessmentService = Depends(get_assessment_service),
) -> PredictResponse:
    assessment = service.predict(payload.demographics.to_dict(), payload.biomarkers)
    return PredictResponse(assessment=assessment)
