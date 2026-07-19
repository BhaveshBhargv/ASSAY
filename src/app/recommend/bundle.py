"""
bundle.py — the final, presentation-ready recommendation object.

Composes the LLM-generated (and guard-cleaned) report with system-owned safety
text that the LLM is NOT allowed to author: the non-diagnostic disclaimer, the
clinician signpost for flag-only markers, and the urgent-referral note. Also
carries the assessment, the evidence pack, and the groundedness audit so nothing
in the report is unattributable — the whole thing serialises to JSON for the API
and renders to Markdown for the UI / dissertation appendix.
"""
from __future__ import annotations

from dataclasses import dataclass

from .fusion import FusedAssessment
from .guards import GuardReport
from .prompt import EvidenceItem
from .report import ADVICE_SECTIONS, SECTION_TITLES, RecommendationReport

_URGENT_NOTE = (
    "One or more results fall in a range that warrants prompt clinical attention. "
    "Please contact your GP or an appropriate healthcare service without delay."
)


@dataclass
class RecommendationBundle:
    assessment: FusedAssessment
    evidence: list[EvidenceItem]
    report: RecommendationReport
    audit: GuardReport
    disclaimer: str
    provider: str = ""

    # ---- signpost / urgent text (system-owned) --------------------------- #
    @property
    def clinician_note(self) -> str:
        if not self.assessment.signpost:
            return ""
        names = ", ".join(f"{s['name']} ({s['status']})" for s in self.assessment.signpost)
        return (f"The following results are outside their typical range and should be "
                f"discussed with a clinician; they are not addressed by lifestyle advice: "
                f"{names}.")

    @property
    def urgent_note(self) -> str:
        return _URGENT_NOTE if self.assessment.urgent_referral else ""

    # ---- serialisation --------------------------------------------------- #
    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "assessment": self.assessment.to_dict(),
            "disclaimer": self.disclaimer,
            "urgent_note": self.urgent_note,
            "clinician_note": self.clinician_note,
            "report": self.report.model_dump(),
            "evidence": [
                {"id": e.id, "citation": e.citation, "title": e.title} for e in self.evidence
            ],
            "groundedness": self.audit.to_dict(),
        }

    # ---- human-readable render ------------------------------------------ #
    def render(self) -> str:
        L: list[str] = []
        L.append("# Personalised Lifestyle Recommendations\n")
        L.append(f"**Overall risk pattern:** {self.assessment.severity}")
        if self.assessment.escalated_by_rf:
            L.append("_(pattern-based risk flagged by the model even though individual "
                     "markers look near-normal)_")
        if self.urgent_note:
            L.append(f"\n> ⚠️ **Urgent:** {self.urgent_note}")
        L.append("")
        L.append("## What your results suggest")
        L.append(self.report.explanation)
        L.append("")

        for sec in ADVICE_SECTIONS:
            items = self.report.section(sec)
            if not items:
                continue
            L.append(f"## {SECTION_TITLES[sec]}")
            for it in items:
                cites = ", ".join(it.evidence)
                L.append(f"- **{it.advice}**")
                L.append(f"  - _Why:_ {it.rationale}")
                L.append(f"  - _Evidence:_ {cites}")
            L.append("")

        if self.clinician_note:
            L.append("## Discuss with your clinician")
            L.append(self.clinician_note)
            L.append("")

        L.append("## Evidence sources")
        for e in self.evidence:
            title = f" — {e.title}" if e.title else ""
            L.append(f"- **{e.id}**: {e.citation}{title}")
        L.append("")

        L.append("---")
        L.append(f"_{self.disclaimer}_")
        return "\n".join(L)
