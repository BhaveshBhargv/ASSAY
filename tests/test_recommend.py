"""Recommendation pipeline tests.

These run offline: FakeProvider returns canned JSON and a fake retriever returns
fixed evidence. They cover fusion, parsing, the guards and rendering.

Run:  python tests/test_recommend.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app.domain.models import PatientContext  # noqa: E402
from app.domain.services.rule_engine import RuleEngine  # noqa: E402
from app.rag.models import GuidelineChunk, RetrievalResult  # noqa: E402
from app.recommend.engine import RecommendationEngine  # noqa: E402
from app.recommend.fusion import fuse  # noqa: E402
from app.recommend.generator import _extract_json, generate_report  # noqa: E402
from app.recommend.guards import scan_diagnostic, verify  # noqa: E402
from app.recommend.prompt import build_evidence_pack  # noqa: E402
from app.recommend.providers import FakeProvider  # noqa: E402
from app.recommend.report import RecommendationReport  # noqa: E402
from app.rules.loader import load_ruleset  # noqa: E402

_EVIDENCE = [
    RetrievalResult(chunk=GuidelineChunk(
        chunk_id="c1", text="Regular physical activity improves glycaemic control.",
        source="NICE", code="NG28", title="Type 2 diabetes",
        biomarkers=("hba1c_pct",)), score=0.9),
    RetrievalResult(chunk=GuidelineChunk(
        chunk_id="c2", text="A Mediterranean-style diet supports healthier lipids.",
        source="NICE", code="CG181", title="Cardiovascular risk",
        biomarkers=("hdl_mgdl", "total_chol_mgdl")), score=0.8),
]

# Canned model output. The item citing E9 (which doesn't exist) and the one with
# an empty rationale should both be dropped by the guard.
_CANNED = json.dumps({
    "explanation": "Your results show a higher HbA1c with low HDL, a pattern linked "
                   "to raised cardiometabolic risk. This is not a diagnosis.",
    "lifestyle": [{"advice": "Build in daily walking.", "rationale": "Helps glycaemic "
                   "control given your raised HbA1c.", "evidence": ["E1"]}],
    "diet": [
        {"advice": "Adopt a Mediterranean-style diet.", "rationale": "Supports healthier "
         "lipids given your low HDL.", "evidence": ["E2"]},
        {"advice": "Take a miracle supplement.", "rationale": "It cures everything.",
         "evidence": ["E9"]},
    ],
    "exercise": [{"advice": "Do resistance training.", "rationale": "", "evidence": ["E1"]}],
    "sleep": [], "weight_management": [], "hydration": [],
    "smoking": [], "alcohol": [],
    "follow_up": [{"advice": "Ask your GP about repeat testing.", "rationale": "Monitors "
                   "borderline results.", "evidence": ["E1"]}],
})


class _FakeRetriever:
    def retrieve_for_assessment(self, severity, flagged, k=5):
        return _EVIDENCE[:k]


def test_fusion_escalates_on_rf():
    engine = RuleEngine(load_ruleset())
    ctx = PatientContext(sex="male", age=54)
    # the rules say normal, the RF says serious
    rule_result = engine.evaluate({"hba1c_pct": 5.2, "hdl_mgdl": 60}, ctx)
    fused = fuse(rule_result, {"predicted_severity": "serious",
                               "probabilities": {"serious": 0.7}})
    assert fused.severity == "serious"
    assert fused.escalated_by_rf is True


def test_fusion_safety_dominant_takes_max():
    engine = RuleEngine(load_ruleset())
    ctx = PatientContext(sex="male", age=54)
    rule_result = engine.evaluate({"hba1c_pct": 6.1}, ctx)  # borderline by the rules
    fused = fuse(rule_result, {"predicted_severity": "normal", "probabilities": {}})
    # a lower RF read never brings the severity down
    assert fused.severity == "borderline"
    assert fused.escalated_by_rf is False


def test_extract_json_from_fenced_and_noisy():
    fenced = "Sure!\n```json\n{\"a\": 1}\n```\nhope that helps"
    assert _extract_json(fenced) == {"a": 1}
    trailing = 'prefix {"x": [1,2], "y": {"z": 3}} suffix'
    assert _extract_json(trailing) == {"x": [1, 2], "y": {"z": 3}}


def test_generate_report_parses_canned():
    provider = FakeProvider(_CANNED)
    report = generate_report(provider, "sys", "user")
    assert isinstance(report, RecommendationReport)
    assert report.explanation
    assert len(report.diet) == 2  # nothing dropped yet


def test_guard_drops_ungrounded_and_unexplained():
    report = generate_report(FakeProvider(_CANNED), "s", "u")
    evidence = build_evidence_pack(_EVIDENCE)
    cleaned, audit = verify(report, evidence)
    assert len(cleaned.diet) == 1
    assert cleaned.exercise == []
    assert len(audit.dropped_items) == 2
    assert audit.total_items == 5 and audit.grounded_items == 3
    assert 0 < audit.groundedness < 1


def test_non_diagnostic_guard_flags_disease_claim():
    bad = RecommendationReport(explanation="You have diabetes and must worry.")
    assert scan_diagnostic(bad)
    good = RecommendationReport(explanation="Your HbA1c is in a higher range.")
    assert not scan_diagnostic(good)


def test_engine_end_to_end_offline():
    engine = RecommendationEngine(
        rule_engine=RuleEngine(load_ruleset()),
        retriever=_FakeRetriever(),
        provider=FakeProvider(_CANNED),
        risk_adapter=None,  # rules only
        k=2,
    )
    bundle = engine.recommend(
        {"age": 54, "sex": "male"},
        {"hba1c_pct": 6.1, "hdl_mgdl": 34, "total_chol_mgdl": 232},
    )
    rendered = bundle.render()
    assert "Personalised Lifestyle Recommendations" in rendered
    assert "not a medical diagnosis" in bundle.disclaimer.lower()
    assert "E1" in rendered and "NICE NG28" in rendered
    assert not bundle.audit.diagnostic_violations
    # the ungrounded supplement advice shouldn't make it into the report
    assert "miracle" not in rendered.lower()


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
