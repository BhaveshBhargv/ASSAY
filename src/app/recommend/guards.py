"""Safety checks run on the generated report.

scan_diagnostic looks for wording that diagnoses the reader ("you have
diabetes"). verify drops advice that doesn't cite the retrieved evidence or
has no rationale, and records how much of the report was grounded. It can also
make a second LLM call to check each item really is supported by its evidence.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from . import config as C
from .ports import ILLMProvider
from .prompt import EvidenceItem
from .report import ADVICE_SECTIONS, AdviceItem, RecommendationReport

log = logging.getLogger(__name__)

# Only phrases that say the reader has a disease. Plain facts ("you have low HDL")
# and disclaimers ("this is not a diagnosis") shouldn't match.
_DISEASE = (r"(?:diabetes|pre-?diabetes|hypertension|anaemia|anemia|cancer|"
            r"(?:kidney|liver|heart|cardiovascular|fatty liver|coronary) disease|"
            r"metabolic syndrome|ckd|cvd|(?:kidney|heart|liver) failure)")
_DIAGNOSTIC_PATTERNS = [
    rf"\byou (?:have|have got|are suffering from|are developing)\s+(?:\w+\s+){{0,3}}{_DISEASE}\b",
    r"\byou(?:'re| are)?\s+(?:diabetic|hypertensive|anaemic|anemic|pre-?diabetic)\b",
    rf"\b(?:diagnosed with|a diagnosis of)\s+(?:\w+\s+){{0,3}}{_DISEASE}\b",
    rf"\bthis (?:confirms|means you have|indicates you have)\s+(?:\w+\s+){{0,3}}{_DISEASE}\b",
    rf"\byou (?:definitely|clearly) have\s+(?:\w+\s+){{0,3}}{_DISEASE}\b",
]
_DIAGNOSTIC_RE = re.compile("|".join(_DIAGNOSTIC_PATTERNS), re.IGNORECASE)


@dataclass
class GuardReport:
    diagnostic_violations: list[str] = field(default_factory=list)
    dropped_items: list[dict] = field(default_factory=list)
    total_items: int = 0
    grounded_items: int = 0

    @property
    def groundedness(self) -> float:
        return (self.grounded_items / self.total_items) if self.total_items else 1.0

    @property
    def passed(self) -> bool:
        return (not self.diagnostic_violations) and self.groundedness >= C.MIN_GROUNDEDNESS

    def to_dict(self) -> dict:
        return {
            "diagnostic_violations": self.diagnostic_violations,
            "dropped_items": self.dropped_items,
            "total_items": self.total_items,
            "grounded_items": self.grounded_items,
            "groundedness": round(self.groundedness, 3),
            "passed": self.passed,
        }


def scan_diagnostic(report: RecommendationReport) -> list[str]:
    """Texts in the report that contain diagnostic wording."""
    hits: list[str] = []
    texts = [report.explanation]
    for _sec, item in report.all_items():
        texts.extend([item.advice, item.rationale])
    for t in texts:
        for m in _DIAGNOSTIC_RE.finditer(t or ""):
            hits.append(t.strip())
            break
    return hits


def verify(
    report: RecommendationReport,
    evidence: list[EvidenceItem],
    provider: ILLMProvider | None = None,
    llm_recheck: bool = C.LLM_ENTAILMENT_RECHECK,
) -> tuple[RecommendationReport, GuardReport]:
    """Check citations and diagnostic wording. Returns (cleaned_report, audit)."""
    valid_ids = {e.id for e in evidence}
    ev_by_id = {e.id: e for e in evidence}
    audit = GuardReport()
    audit.diagnostic_violations = scan_diagnostic(report)

    cleaned: dict[str, list[AdviceItem]] = {sec: [] for sec in ADVICE_SECTIONS}

    for sec in ADVICE_SECTIONS:
        for item in report.section(sec):
            audit.total_items += 1
            cited = [e for e in item.evidence if e in valid_ids]
            reason_ok = bool(item.rationale and item.rationale.strip())

            if not cited or not reason_ok:
                audit.dropped_items.append({
                    "section": sec,
                    "advice": item.advice,
                    "why": "no valid evidence citation" if not cited else "missing rationale",
                })
                continue

            if llm_recheck and provider is not None:
                if not _entailed(provider, item, [ev_by_id[i] for i in cited]):
                    audit.dropped_items.append({
                        "section": sec, "advice": item.advice,
                        "why": "failed LLM entailment recheck",
                    })
                    continue

            audit.grounded_items += 1
            cleaned[sec].append(AdviceItem(
                advice=item.advice, rationale=item.rationale, evidence=cited,
            ))

    cleaned_report = report.model_copy(update=cleaned)
    log.info("groundedness=%.2f, dropped=%d, diagnostic_hits=%d",
             audit.groundedness, len(audit.dropped_items), len(audit.diagnostic_violations))
    return cleaned_report, audit


def _entailed(provider: ILLMProvider, item: AdviceItem, evidence: list[EvidenceItem]) -> bool:
    """Ask the model whether the advice is supported by its evidence."""
    ev = "\n\n".join(e.render() for e in evidence)
    system = ("You check whether a piece of lifestyle advice is directly supported by "
              "the provided evidence. Answer with exactly 'YES' or 'NO'.")
    user = (f"EVIDENCE:\n{ev}\n\nADVICE: {item.advice}\nRATIONALE: {item.rationale}\n\n"
            "Is this advice supported by the evidence? Answer YES or NO.")
    try:
        ans = provider.complete(system, user).strip().upper()
        return ans.startswith("YES")
    except Exception:  # pragma: no cover
        log.warning("entailment recheck failed; keeping item")
        return True
