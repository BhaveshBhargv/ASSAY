"""
test_rule_engine.py — Phase 3 unit tests.

Run:  python -m pytest tests/test_rule_engine.py -q
      (or)  python tests/test_rule_engine.py     # falls back to a simple runner
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app.domain.enums import Direction, Severity, Status       # noqa: E402
from app.domain.models import PatientContext                   # noqa: E402
from app.domain.services.rule_engine import RuleEngine         # noqa: E402
from app.rules.loader import load_ruleset                      # noqa: E402

ENGINE = RuleEngine(load_ruleset())
MALE = PatientContext(sex="male", age=50)
FEMALE = PatientContext(sex="female", age=50)


def test_ruleset_loads_and_validates():
    assert ENGINE.ruleset_version
    assert "hba1c_pct" in ENGINE.codes


def test_normal_hba1c():
    r = ENGINE.classify("hba1c_pct", 5.2, MALE)
    assert r.status == Status.NORMAL and r.severity == Severity.NORMAL
    assert r.urgent is False


def test_prediabetes_borderline():
    r = ENGINE.classify("hba1c_pct", 6.0, MALE)
    assert r.status == Status.BORDERLINE and r.severity == Severity.BORDERLINE
    assert r.direction == Direction.HIGH


def test_diabetes_high():
    r = ENGINE.classify("hba1c_pct", 7.1, MALE)
    assert r.status == Status.HIGH and r.severity == Severity.SERIOUS


def test_severe_hba1c_is_urgent():
    r = ENGINE.classify("hba1c_pct", 10.5, MALE)
    assert r.status == Status.SEVERE and r.urgent is True


def test_boundary_is_inclusive_lower():
    # 6.5 is the start of the diabetes band (min inclusive)
    assert ENGINE.classify("hba1c_pct", 6.5, MALE).status == Status.HIGH
    # 6.4999 stays borderline
    assert ENGINE.classify("hba1c_pct", 6.49, MALE).status == Status.BORDERLINE


def test_sex_specific_hdl():
    # 45 mg/dL: normal for a man (>=40), low for a woman (<50)
    assert ENGINE.classify("hdl_mgdl", 45, MALE).status == Status.NORMAL
    assert ENGINE.classify("hdl_mgdl", 45, FEMALE).status == Status.LOW


def test_low_direction_glucose():
    r = ENGINE.classify("fasting_glucose_mgdl", 60, MALE)
    assert r.status == Status.LOW and r.direction == Direction.LOW


def test_missing_value_returns_none():
    assert ENGINE.classify("hba1c_pct", None, MALE) is None
    assert ENGINE.classify("hba1c_pct", float("nan"), MALE) is None


def test_evaluate_summary_overall_and_urgent():
    readings = {"hba1c_pct": 10.5, "hdl_mgdl": 55}
    res = ENGINE.evaluate(readings, MALE)
    assert res.overall_severity == Severity.SERIOUS   # hba1c severe -> serious
    assert res.urgent_referral is True                # hba1c 10.5 is severe
    assert "hba1c_pct" in res.flagged
    assert "hdl_mgdl" not in res.flagged              # 55 normal for male


def test_unknown_code_recorded():
    res = ENGINE.evaluate({"unknown_marker": 1.0}, MALE)
    assert "unknown_marker" in res.unknown_codes
    assert res.overall_severity == Severity.NORMAL


def test_measurements_removed_and_all_blood():
    # BP/BMI/waist are out of scope — not in the ruleset at all.
    for c in ["sbp_mmhg", "dbp_mmhg", "bmi", "waist_cm"]:
        assert c not in ENGINE.codes
    # Every configured biomarker is a blood analyte.
    res = ENGINE.evaluate({c: 5 for c in ENGINE.codes}, MALE)
    assert all(b.category == "blood" for b in res.biomarkers)


def test_flag_only_routes_to_signpost_not_recommendations():
    # Abnormal potassium (flag-only) must NOT become a recommendation target;
    # it goes to the clinician signpost instead.
    res = ENGINE.evaluate({"potassium": 6.4, "hba1c_pct": 6.0}, MALE)
    assert "potassium" in res.clinician_signpost
    assert "potassium" not in res.recommendation_targets
    assert "hba1c_pct" in res.recommendation_targets   # actionable
    assert res.urgent_referral is True                 # K 6.4 is severe


def test_anaemia_marker_is_actionable():
    res = ENGINE.evaluate({"hemoglobin": 10.5}, FEMALE)  # low Hb for a woman
    assert "hemoglobin" in res.recommendation_targets    # actionable, drives advice


def test_expanded_panel_size():
    # Blood-only panel: 27 biomarkers (BP/BMI/waist removed); 21 actionable, 6 flag-only.
    assert len(ENGINE.codes) == 27
    assert len(ENGINE.actionable_codes) == 21


def test_weighted_core_single_core_borderline_escalates():
    # Prediabetes alone (core marker, borderline) -> overall borderline.
    res = ENGINE.evaluate({"hba1c_pct": 6.0}, MALE)
    assert res.overall_severity == Severity.BORDERLINE


def test_weighted_core_single_secondary_stays_normal():
    # One mildly-raised RDW (secondary) alone must NOT flag the whole record.
    res = ENGINE.evaluate({"rdw": 15.0}, MALE)
    assert res.overall_severity == Severity.NORMAL


def test_weighted_core_two_secondary_escalate():
    # Two secondary borderline findings (raised RDW + low vitamin D) -> borderline.
    res = ENGINE.evaluate({"rdw": 15.0, "vitamin_d": 40}, MALE)
    assert res.overall_severity == Severity.BORDERLINE


def test_output_is_json_serializable():
    import json
    res = ENGINE.evaluate({"hba1c_pct": 6.8}, MALE)
    json.dumps(res.to_dict())  # must not raise
    d = res.to_dict()
    assert d["summary"]["overall_severity"] == "serious"


def _run_all():
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"PASS {fn.__name__}")
    print(f"\n{passed}/{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
