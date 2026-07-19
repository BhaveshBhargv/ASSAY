"""
guards.py — safety net enforcing the four hard rules after generation.

Two independent guards run on the parsed report:

1. Non-diagnostic guard  — regex scan for diagnostic phrasing ("you have X",
   "you are diabetic", "diagnosed with ...", "this confirms ..."). Violations are
   recorded (and surfaced as a Phase-9 safety metric); the prompt already forbids
   them, but we never trust the model blindly.

2. Groundedness verifier — the requested second pass. Deterministically:
     • strips any evidence id an item cites that isn't in the retrieved pack;
     • DROPS advice items left with zero valid citations (ungrounded);
     • DROPS items with an empty rationale (rule: always explain why);
   then reports a groundedness score. An optional LLM entailment recheck can add a
   semantic "is this advice actually supported by Ei?" check on top.

The verifier returns a NEW report (kept items only) plus a structured audit dict.
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

# Phrases that assert a DISEASE diagnosis about the reader. Deliberately scoped to
# disease attribution so neutral biomarker facts ("you have low HDL") and safe
# disclaimers ("this is not a diagnosis") are NOT flagged.
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
    """Return the offending snippets of any diagnostic phrasing found."""
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
    """Ground-check + non-diagnostic scan. Returns (cleaned_report, audit)."""
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
    """Optional heavier check: ask the model whether the advice is supported."""
    ev = "\n\n".join(e.render() for e in evidence)
    system = ("You check whether a piece of lifestyle advice is directly supported by "
              "the provided evidence. Answer with exactly 'YES' or 'NO'.")
    user = (f"EVIDENCE:\n{ev}\n\nADVICE: {item.advice}\nRATIONALE: {item.rationale}\n\n"
            "Is this advice supported by the evidence? Answer YES or NO.")
    try:
        ans = provider.complete(system, user).strip().upper()
        return ans.startswith("YES")
    except Exception:  # pragma: no cover - never let the recheck break generation
        log.warning("entailment recheck failed; keeping item")
        return True
