"""
test_parser.py — report parser: unit-anchoring, conversion, plausibility.

Uses synthetic lines that mimic a real lab-report layout (label / Method / Machine
/ value-unit-range), so it runs offline with no PDF fixture.

Run:  python tests/test_parser.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app.ingestion.report_parser import (  # noqa: E402
    extract_biomarkers_from_lines,
    parse_upload,
)
from app.ingestion.units import is_plausible, to_canonical  # noqa: E402

# label / Method / Machine / value — the layout that broke the old parser.
_LINES = [
    "Total Cholesterol", "Method: Cholesterol Oxidase", "Machine: BECKMAN AU5800",
    "204.7 mg/dl Desirable : <200",
    "Serum HDL Cholesterol", "52.1 mg/dl 40 - 60",
    "Haemoglobin (HB) :", "15.3 g/dL 13.0-17.0",
    "HbA1c", "5.90 % 4.2 - 5.7",
    "Vitamin D", "17.26 ng/ml 30 - 100",
    "Vitamin B12", "199 pg/ml 211 - 912",          # must NOT become glucose
    "Blood Glucose Fasting", "92.24 mg/dl 70 - 100",
    "Platelet count", "246 10^3/uL 150-410",
]


def test_stateful_extraction_pairs_label_and_value():
    found, _conv = extract_biomarkers_from_lines(_LINES)
    assert found["total_chol_mgdl"] == 204.7
    assert found["hdl_mgdl"] == 52.1
    assert found["hemoglobin"] == 15.3
    assert found["hba1c_pct"] == 5.9
    assert found["platelets"] == 246


def test_unit_anchoring_rejects_wrong_unit():
    # glucose must skip "199 pg/ml" (Vitamin B12) and take the mg/dl reading
    found, _ = extract_biomarkers_from_lines(_LINES)
    assert found["fasting_glucose_mgdl"] == 92.24


def test_vitamin_d_unit_conversion():
    found, conv = extract_biomarkers_from_lines(_LINES)
    assert abs(found["vitamin_d"] - 17.26 * 2.496) < 0.1     # ng/mL → nmol/L
    assert any("vitamin_d" in c for c in conv)


def test_plausibility_rejects_garbage():
    # implausible values (bad extraction) are dropped
    found, _ = extract_biomarkers_from_lines(["Haemoglobin", "1 g/dL 13-17"])
    assert "hemoglobin" not in found
    found2, _ = extract_biomarkers_from_lines(["MCV", "12 fL 83-101"])
    assert "mcv" not in found2


def test_units_helpers():
    v, frm = to_canonical("vitamin_d", 20.0, "ng/ml")
    assert abs(v - 49.92) < 0.1 and frm == "ng/ml"
    assert to_canonical("fasting_glucose_mgdl", 199, "pg/ml") == (None, None)
    assert is_plausible("hba1c_pct", 5.9) and not is_plausible("hemoglobin", 1.0)


def test_structured_json_still_works():
    payload = json.dumps({"biomarkers": {"HbA1c": 6.1, "HDL cholesterol": 34}}).encode()
    res = parse_upload("r.json", payload)
    assert res.biomarkers == {"hba1c_pct": 6.1, "hdl_mgdl": 34.0}


def test_structured_csv_long_format():
    csv = b"biomarker,value\nHbA1c,6.1\nTriglycerides,205\n"
    res = parse_upload("r.csv", csv)
    assert res.biomarkers["hba1c_pct"] == 6.1 and res.biomarkers["triglycerides_mgdl"] == 205.0


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
