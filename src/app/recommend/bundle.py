"""The final recommendation: the cleaned report plus text the app adds itself.

The disclaimer, clinician note and urgent-referral note are written here rather
than by the LLM. The bundle serialises to JSON for the API and renders to
Markdown.
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

    def render(self) -> str:
        lines: list[str] = []
        lines.append("# Personalised Lifestyle Recommendations\n")
        lines.append(f"**Overall risk pattern:** {self.assessment.severity}")
        if self.assessment.escalated_by_rf:
            lines.append("_(pattern-based risk flagged by the model even though individual "
                         "markers look near-normal)_")
        if self.urgent_note:
            lines.append(f"\n> ⚠️ **Urgent:** {self.urgent_note}")
        lines.append("")
        lines.append("## What your results suggest")
        lines.append(self.report.explanation)
        lines.append("")

        for sec in ADVICE_SECTIONS:
            items = self.report.section(sec)
            if not items:
                continue
            lines.append(f"## {SECTION_TITLES[sec]}")
            for it in items:
                cites = ", ".join(it.evidence)
                lines.append(f"- **{it.advice}**")
                lines.append(f"  - _Why:_ {it.rationale}")
                lines.append(f"  - _Evidence:_ {cites}")
            lines.append("")

        if self.clinician_note:
            lines.append("## Discuss with your clinician")
            lines.append(self.clinician_note)
            lines.append("")

        lines.append("## Evidence sources")
        for e in self.evidence:
            title = f" — {e.title}" if e.title else ""
            lines.append(f"- **{e.id}**: {e.citation}{title}")
        lines.append("")

        lines.append("---")
        lines.append(f"_{self.disclaimer}_")
        return "\n".join(lines)
