"""Combines the rule engine and Random Forest results into one severity.

The final severity is the higher of the two, so the model can raise the rule
result but never lower it. Abnormal markers are also split into actionable ones,
which get lifestyle advice, and flag-only ones, which are referred to a
clinician.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..domain.enums import SEVERITY_RANK, Severity
from ..domain.models import RuleEngineResult


@dataclass
class FusedAssessment:
    severity: str  # final combined severity
    rule_severity: str
    rf_severity: Optional[str]
    rf_probabilities: dict = field(default_factory=dict)
    escalated_by_rf: bool = False
    urgent_referral: bool = False
    # SHAP drivers of the model's prediction: [{feature, label, value, contribution}]
    rf_drivers: list[dict] = field(default_factory=list)
    # actionable abnormal markers, which the LLM gives advice on
    flagged: list[dict] = field(default_factory=list)
    # flag-only abnormal markers, which are referred to a clinician
    signpost: list[dict] = field(default_factory=list)

    @property
    def flagged_names(self) -> list[str]:
        return [f["name"] for f in self.flagged]

    def to_dict(self) -> dict:
        return {
            "severity": self.severity,
            "rule_severity": self.rule_severity,
            "rf_severity": self.rf_severity,
            "rf_probabilities": self.rf_probabilities,
            "escalated_by_rf": self.escalated_by_rf,
            "urgent_referral": self.urgent_referral,
            "rf_drivers": self.rf_drivers,
            "flagged": self.flagged,
            "signpost": self.signpost,
        }


def _max_severity(a: str, b: str) -> str:
    """The higher of two severity names (unknown names count as normal)."""
    ra = SEVERITY_RANK.get(Severity(a), 0) if a in Severity._value2member_map_ else 0
    rb = SEVERITY_RANK.get(Severity(b), 0) if b in Severity._value2member_map_ else 0
    return a if ra >= rb else b


def fuse(rule_result: RuleEngineResult, rf_output: Optional[dict]) -> FusedAssessment:
    """Combine the rule result with the model output (None means rules only)."""
    rule_sev = rule_result.overall_severity.value
    rf_sev = rf_output.get("predicted_severity") if rf_output else None
    rf_probs = rf_output.get("probabilities", {}) if rf_output else {}
    rf_drivers = rf_output.get("drivers", []) if rf_output else []

    final = _max_severity(rule_sev, rf_sev) if rf_sev else rule_sev
    escalated = bool(
        rf_sev
        and SEVERITY_RANK.get(Severity(rf_sev), 0) > SEVERITY_RANK.get(Severity(rule_sev), 0)
    )

    flagged: list[dict] = []
    signpost: list[dict] = []
    for b in rule_result.biomarkers:
        if b.severity == Severity.NORMAL:
            continue
        entry = {
            "code": b.code,
            "name": b.name,
            "status": b.status.value,
            "severity": b.severity.value,
            "interpretation": b.interpretation,
        }
        if b.tier == "actionable":
            flagged.append(entry)
        else:  # flag_only
            signpost.append(entry)

    return FusedAssessment(
        severity=final,
        rule_severity=rule_sev,
        rf_severity=rf_sev,
        rf_probabilities=rf_probs,
        escalated_by_rf=escalated,
        rf_drivers=rf_drivers,
        urgent_referral=rule_result.urgent_referral,
        flagged=flagged,
        signpost=signpost,
    )
