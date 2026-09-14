"""Classifies biomarker readings against the configured rule set.

All thresholds and interpretation text come from the RuleSet, so new
biomarkers only need adding to the YAML file.
"""
from __future__ import annotations

import math
from typing import Optional

from ..enums import Severity, Status
from ..models import (
    BiomarkerResult,
    PatientContext,
    RuleEngineResult,
    RuleSet,
)


class RuleEngine:
    def __init__(self, ruleset: RuleSet) -> None:
        self._ruleset = ruleset

    @property
    def ruleset_version(self) -> str:
        return self._ruleset.version

    @property
    def codes(self) -> list[str]:
        return list(self._ruleset.rules.keys())

    @property
    def actionable_codes(self) -> list[str]:
        return [c for c, r in self._ruleset.rules.items() if r.tier == "actionable"]

    @property
    def core_actionable_codes(self) -> list[str]:
        return [c for c, r in self._ruleset.rules.items()
                if r.tier == "actionable" and r.label_role == "core"]

    @property
    def secondary_actionable_codes(self) -> list[str]:
        return [c for c, r in self._ruleset.rules.items()
                if r.tier == "actionable" and r.label_role == "secondary"]

    @property
    def secondary_borderline_min(self) -> int:
        return self._ruleset.secondary_borderline_min

    def _band_for(self, code: str, value: float, sex: Optional[str]):
        rule = self._ruleset.get(code)
        if rule is None:
            return None, None
        for band in rule.bands:
            if band.contains(value, sex):
                return rule, band
        return rule, None  # shouldn't happen, the bands cover every value

    def classify(
        self, code: str, value: Optional[float], context: PatientContext
    ) -> Optional[BiomarkerResult]:
        """Classify one reading. Returns None for unknown codes or missing values."""
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        rule, band = self._band_for(code, float(value), context.sex)
        if rule is None or band is None:
            return None

        text = band.interpretation.format(
            name=rule.name, value=_fmt(value), unit=rule.unit
        )
        return BiomarkerResult(
            code=code,
            name=rule.name,
            category=rule.category,
            tier=rule.tier,
            label_role=rule.label_role,
            value=float(value),
            unit=rule.unit.strip(),
            status=band.status,
            direction=band.direction,
            severity=band.severity,
            urgent=band.is_urgent(),
            reference_range=rule.reference_range(context.sex),
            interpretation=text,
            guideline=band.guideline,
        )

    def evaluate(
        self, readings: dict[str, Optional[float]], context: PatientContext
    ) -> RuleEngineResult:
        result = RuleEngineResult(
            ruleset_version=self.ruleset_version,
            context=context,
            secondary_borderline_min=self._ruleset.secondary_borderline_min,
        )
        for code, value in readings.items():
            if self._ruleset.get(code) is None:
                result.unknown_codes.append(code)
                continue
            res = self.classify(code, value, context)
            if res is not None:
                result.biomarkers.append(res)
        return result

    def layer1_severity(self, code: str, value: Optional[float], sex: Optional[str]) -> Severity:
        """Severity of one reading. Used when building the training labels, so the
        labels and the live rule engine use exactly the same thresholds."""
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return Severity.NORMAL
        _, band = self._band_for(code, float(value), sex)
        return band.severity if band is not None else Severity.NORMAL


def _fmt(value: float) -> str:
    """Format a number without a trailing .0."""
    f = float(value)
    return str(int(f)) if f.is_integer() else f"{f:g}"
