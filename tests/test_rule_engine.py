"""Rule engine tests.

Run:  python -m pytest tests/test_rule_engine.py -q
  or  python tests/test_rule_engine.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app.domain.enums import Direction, Severity, Status  # noqa: E402
from app.domain.models import PatientContext  # noqa: E402
from app.domain.services.rule_engine import RuleEngine  # noqa: E402
from app.rules.loader import load_ruleset  # noqa: E402

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
    # the diabetes band starts at 6.5 inclusive
    assert ENGINE.classify("hba1c_pct", 6.5, MALE).status == Status.HIGH
    assert ENGINE.classify("hba1c_pct", 6.49, MALE).status == Status.BORDERLINE


def test_sex_specific_hdl():
    # 45 mg/dL is normal for a man (>=40) but low for a woman (<50)
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
    assert res.overall_severity == Severity.SERIOUS
    assert res.urgent_referral is True                # HbA1c 10.5 is severe
    assert "hba1c_pct" in res.flagged
    assert "hdl_mgdl" not in res.flagged              # 55 is normal for a man


def test_unknown_code_recorded():
    res = ENGINE.evaluate({"unknown_marker": 1.0}, MALE)
    assert "unknown_marker" in res.unknown_codes
    assert res.overall_severity == Severity.NORMAL


def test_measurements_removed_and_all_blood():
    # BP, BMI and waist aren't in the ruleset
    for c in ["sbp_mmhg", "dbp_mmhg", "bmi", "waist_cm"]:
        assert c not in ENGINE.codes
    res = ENGINE.evaluate({c: 5 for c in ENGINE.codes}, MALE)
    assert all(b.category == "blood" for b in res.biomarkers)


def test_flag_only_routes_to_signpost_not_recommendations():
    # potassium is flag-only, so it goes to the clinician signpost, not the advice
    res = ENGINE.evaluate({"potassium": 6.4, "hba1c_pct": 6.0}, MALE)
    assert "potassium" in res.clinician_signpost
    assert "potassium" not in res.recommendation_targets
    assert "hba1c_pct" in res.recommendation_targets
    assert res.urgent_referral is True                 # potassium 6.4 is severe


def test_anaemia_marker_is_actionable():
    res = ENGINE.evaluate({"hemoglobin": 10.5}, FEMALE)  # low for a woman
    assert "hemoglobin" in res.recommendation_targets


def test_expanded_panel_size():
    # 27 biomarkers: 21 actionable and 6 flag-only
    assert len(ENGINE.codes) == 27
    assert len(ENGINE.actionable_codes) == 21


def test_weighted_core_single_core_borderline_escalates():
    # one borderline core marker is enough
    res = ENGINE.evaluate({"hba1c_pct": 6.0}, MALE)
    assert res.overall_severity == Severity.BORDERLINE


def test_weighted_core_single_secondary_stays_normal():
    # a single borderline secondary marker isn't
    res = ENGINE.evaluate({"rdw": 15.0}, MALE)
    assert res.overall_severity == Severity.NORMAL


def test_weighted_core_two_secondary_escalate():
    # but two are (raised RDW and low vitamin D)
    res = ENGINE.evaluate({"rdw": 15.0, "vitamin_d": 40}, MALE)
    assert res.overall_severity == Severity.BORDERLINE


def test_output_is_json_serializable():
    import json
    res = ENGINE.evaluate({"hba1c_pct": 6.8}, MALE)
    json.dumps(res.to_dict())
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
