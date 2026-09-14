"""Builds the prompts for the recommendation model.

The system prompt sets the rules (no diagnosis, only use the evidence, always
explain why). The user prompt holds the assessment and the retrieved passages,
labelled E1, E2, ... so each piece of advice can cite them.
"""
from __future__ import annotations

from dataclasses import dataclass

from .fusion import FusedAssessment
from .report import format_instructions

SYSTEM_PROMPT = """You are a careful clinical-lifestyle information assistant for a \
UK health context. You turn blood-test findings into evidence-based, patient-friendly \
LIFESTYLE guidance.

You MUST obey these rules without exception:
1. NEVER diagnose. Do not state or imply the person HAS any disease or condition \
(e.g. never say "you have diabetes", "you are diabetic", "this means kidney disease"). \
Describe results as patterns and risk levels only.
2. ONLY use the EVIDENCE passages provided below. Do not add facts, thresholds, drug \
names, or claims from your own knowledge. If the evidence does not support a piece of \
advice, do not give it.
3. ALWAYS explain WHY. Every recommendation must have a rationale tied to the person's \
specific flagged results, and must cite at least one evidence id (e.g. "E1").
4. Give practical, safe, non-alarming lifestyle guidance. Encourage discussing results \
with a GP/clinician. Do not prescribe medication or doses.

Write for a layperson: clear, supportive, concrete. Return only the required JSON."""

_TASK = """Using ONLY the evidence below, produce a lifestyle recommendation report for \
this person. Cover, where the evidence supports it: a plain-English explanation of what \
the results suggest (no diagnosis), then advice for general lifestyle, diet, exercise, \
sleep, weight management, hydration, smoking, alcohol, and follow-up/monitoring.

Tailor everything to the FLAGGED RESULTS. Only advise where evidence supports it; leave a \
section's array empty rather than inventing unsupported advice."""


@dataclass
class EvidenceItem:
    """A retrieved passage with its id in the prompt (E1, E2, ...)."""
    id: str
    citation: str
    title: str
    text: str

    def render(self) -> str:
        head = f"[{self.id}] ({self.citation}"
        head += f" — {self.title})" if self.title else ")"
        return f"{head}\n{self.text.strip()}"


def build_evidence_pack(results) -> list[EvidenceItem]:
    """Number the retrieved passages E1, E2, ..."""
    pack: list[EvidenceItem] = []
    for i, res in enumerate(results, start=1):
        c = res.chunk
        pack.append(EvidenceItem(
            id=f"E{i}",
            citation=c.citation(),
            title=c.title or "",
            text=c.text,
        ))
    return pack


def _render_demographics(demographics: dict) -> str:
    parts = []
    for key in ("age", "sex"):
        if demographics.get(key) is not None:
            parts.append(f"{key}: {demographics[key]}")
    for k, v in demographics.items():
        if k not in ("age", "sex") and v is not None:
            parts.append(f"{k}: {v}")
    return ", ".join(parts) if parts else "not provided"


def _render_flags(assessment: FusedAssessment) -> str:
    if not assessment.flagged:
        return "No individual biomarker is outside its reference range."
    lines = []
    for f in assessment.flagged:
        lines.append(f"- {f['name']}: {f['status']} ({f['severity']}). {f['interpretation']}")
    return "\n".join(lines)


def build_user_prompt(demographics: dict, assessment: FusedAssessment,
                      evidence: list[EvidenceItem]) -> str:
    """The user message: the task, patient details, evidence and JSON format."""
    escalation = ""
    if assessment.escalated_by_rf:
        escalation = (
            "\nNOTE: individual markers look near-normal, but a machine-learning model "
            "flagged the overall PATTERN as higher risk. Frame advice as sensible "
            "precaution, not alarm.")

    evidence_block = "\n\n".join(e.render() for e in evidence) or "(no evidence retrieved)"

    return (
        f"{_TASK}\n\n"
        f"PATIENT (de-identified): {_render_demographics(demographics)}\n"
        f"OVERALL RISK PATTERN: {assessment.severity}{escalation}\n\n"
        f"FLAGGED RESULTS (advise on these):\n{_render_flags(assessment)}\n\n"
        f"EVIDENCE (your ONLY knowledge source — cite by id):\n{evidence_block}\n\n"
        f"{format_instructions()}"
    )


def build_messages(demographics: dict, assessment: FusedAssessment,
                   evidence: list[EvidenceItem]) -> tuple[str, str]:
    """Return the (system, user) prompt strings.

    If langchain_core is installed they're passed through a ChatPromptTemplate,
    otherwise the plain strings are returned.
    """
    user = build_user_prompt(demographics, assessment, evidence)
    try:
        from langchain_core.prompts import ChatPromptTemplate
        tmpl = ChatPromptTemplate.from_messages(
            [("system", "{system}"), ("human", "{user}")]
        )
        msgs = tmpl.format_messages(system=SYSTEM_PROMPT, user=user)
        return msgs[0].content, msgs[1].content
    except Exception:  # pragma: no cover
        return SYSTEM_PROMPT, user
