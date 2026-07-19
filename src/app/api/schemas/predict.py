"""predict.py — /predict request/response (rules + Random Forest + fusion)."""
from __future__ import annotations

from pydantic import BaseModel, field_validator

from .common import EXAMPLE_BIOMARKERS, AssessmentOut, Demographics, validate_biomarkers


class PredictRequest(BaseModel):
    demographics: Demographics
    biomarkers: dict[str, float]

    _v = field_validator("biomarkers")(validate_biomarkers)

    model_config = {
        "json_schema_extra": {
            "example": {
                "demographics": {"age": 54, "sex": "male", "eth_code": 3, "pir": 2.5, "educ_code": 4},
                "biomarkers": EXAMPLE_BIOMARKERS,
            }
        }
    }


class PredictResponse(BaseModel):
    assessment: AssessmentOut
