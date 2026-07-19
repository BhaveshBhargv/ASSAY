"""
report.py — the structured recommendation schema (LLM output contract).

Every piece of advice is an `AdviceItem` that MUST carry a `rationale` (the
"why") and at least one `evidence` citation id (e.g. "E1") pointing at a
retrieved guideline passage. This structure is what makes the four hard rules
enforceable downstream:

    • Only retrieved evidence  -> evidence ids are validated against the pack
    • Always explain why       -> rationale is a required, non-empty field
    • Never diagnose           -> no diagnosis field exists; a guard scans text
    • Avoid hallucination      -> structured parse + citation + groundedness check

The nine advice sections are exactly the ones requested in the Phase-6 spec.
"""
from __future__ import annotations

import json
from typing import List

from pydantic import BaseModel, Field

# The nine lifestyle advice sections (order preserved for rendering).
ADVICE_SECTIONS: tuple[str, ...] = (
    "lifestyle",
    "diet",
    "exercise",
    "sleep",
    "weight_management",
    "hydration",
    "smoking",
    "alcohol",
    "follow_up",
)

SECTION_TITLES: dict[str, str] = {
    "lifestyle": "General Lifestyle",
    "diet": "Diet",
    "exercise": "Exercise & Physical Activity",
    "sleep": "Sleep",
    "weight_management": "Weight Management",
    "hydration": "Hydration",
    "smoking": "Smoking",
    "alcohol": "Alcohol",
    "follow_up": "Follow-up & Monitoring",
}


class AdviceItem(BaseModel):
    """One concrete, actionable recommendation with its justification + evidence."""
    advice: str = Field(..., description="A single, specific, actionable recommendation.")
    rationale: str = Field(..., description="Why this is advised, tied to the patient's results.")
    evidence: List[str] = Field(
        default_factory=list,
        description="Ids of the retrieved guideline passages that support this advice, e.g. ['E1','E3'].",
    )


class RecommendationReport(BaseModel):
    """The full LLM-generated report (system-owned safety text added separately)."""
    explanation: str = Field(
        ..., description="Plain-English summary of what the results suggest, no diagnosis."
    )
    lifestyle: List[AdviceItem] = Field(default_factory=list)
    diet: List[AdviceItem] = Field(default_factory=list)
    exercise: List[AdviceItem] = Field(default_factory=list)
    sleep: List[AdviceItem] = Field(default_factory=list)
    weight_management: List[AdviceItem] = Field(default_factory=list)
    hydration: List[AdviceItem] = Field(default_factory=list)
    smoking: List[AdviceItem] = Field(default_factory=list)
    alcohol: List[AdviceItem] = Field(default_factory=list)
    follow_up: List[AdviceItem] = Field(default_factory=list)

    def section(self, name: str) -> List[AdviceItem]:
        return getattr(self, name)

    def all_items(self) -> list[tuple[str, AdviceItem]]:
        """Every (section, item) pair — used by the groundedness verifier."""
        pairs: list[tuple[str, AdviceItem]] = []
        for sec in ADVICE_SECTIONS:
            for item in self.section(sec):
                pairs.append((sec, item))
        return pairs


# --- schema helpers -------------------------------------------------------- #

def example_json() -> str:
    """A compact, valid example that anchors the model's JSON output shape."""
    example = {
        "explanation": "Your HbA1c is in the higher range and your HDL ('good') "
                       "cholesterol is low, a combination linked to raised "
                       "cardiometabolic risk. This is not a diagnosis.",
        "lifestyle": [
            {
                "advice": "Aim to reduce sitting time and take short walking breaks.",
                "rationale": "Your raised HbA1c and low HDL point to metabolic risk "
                             "that regular movement helps improve.",
                "evidence": ["E1"],
            }
        ],
        "diet": [
            {
                "advice": "Favour wholegrain carbohydrates over refined/sugary foods.",
                "rationale": "Supports steadier blood glucose given your higher HbA1c.",
                "evidence": ["E2"],
            }
        ],
        "exercise": [],
        "sleep": [],
        "weight_management": [],
        "hydration": [],
        "smoking": [],
        "alcohol": [],
        "follow_up": [
            {
                "advice": "Ask your GP about repeat blood tests to monitor these markers.",
                "rationale": "Tracking change over time is recommended for borderline results.",
                "evidence": ["E3"],
            }
        ],
    }
    return json.dumps(example, indent=2)


def format_instructions() -> str:
    """Plain-text JSON contract for the prompt (provider-agnostic; no framework)."""
    sections = ", ".join(ADVICE_SECTIONS)
    return (
        "Respond with a SINGLE valid JSON object and nothing else — no prose, no "
        "markdown fences. The object MUST have these keys:\n"
        f'  "explanation": string\n'
        f"  {sections}: each an array of items\n"
        "Each item is an object: "
        '{"advice": string, "rationale": string, "evidence": [string, ...]}.\n'
        "Every advice item MUST include a non-empty rationale and at least one "
        "evidence id (e.g. \"E1\") drawn ONLY from the EVIDENCE list provided. "
        "If a section has no evidence-supported advice, use an empty array [].\n\n"
        "Example of the required shape:\n" + example_json()
    )
