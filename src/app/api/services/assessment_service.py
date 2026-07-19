"""
assessment_service.py — the /predict use case: rules + Random Forest + fusion.

Produces the fused assessment shared by /predict and /recommend.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from app.domain.models import PatientContext
from app.recommend.fusion import fuse

from ..repositories.model_repository import ModelRepository
from ..repositories.rule_repository import RuleRepository
from ..schemas.common import AssessmentOut
from . import mappers

log = logging.getLogger("assay.api")


@dataclass
class AssessmentBundle:
    rule_result: object
    fused: object


class AssessmentService:
    def __init__(self, rules: RuleRepository, models: ModelRepository) -> None:
        self._rules = rules
        self._models = models

    def assess(self, demographics: dict, biomarkers: dict) -> AssessmentBundle:
        ctx = PatientContext(sex=demographics.get("sex"), age=demographics.get("age"))
        rule_result = self._rules.engine.evaluate(biomarkers, ctx)
        rf_output = self._models.predict(demographics, biomarkers)
        fused = fuse(rule_result, rf_output)
        log.info("assessed: severity=%s escalated=%s flagged=%d",
                 fused.severity, fused.escalated_by_rf, len(fused.flagged))
        return AssessmentBundle(rule_result, fused)

    def predict(self, demographics: dict, biomarkers: dict) -> AssessmentOut:
        b = self.assess(demographics, biomarkers)
        return mappers.assessment_out(b.rule_result, b.fused)
