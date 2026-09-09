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
    extract_demographics,
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


# --- single-row table layout: "Label  Value  Unit  Range" on ONE line --------- #
# The layout used by the summary tables at the front of most lab reports. The
# original parser only accepted values on a line of their own, so it skipped
# these entirely.
_ROW_LINES = [
    "Test Name Result Range",
    " Hemoglobin 10.7 g/dL 12.0 - 15.0",
    " TLC 12.6 10^3/µl 4 - 10",
    " Platelet Count 346 10^3/µl 150 - 410",
    " Glycosylated Hemoglobin (HbA1c) 6.8 % 0 - 5.7",
    " Estimated Average Glucose 148.46 mg/dL —",      # derived: must be ignored
    " Glucose Fasting 152.2 mg/dL 70 - 100",
    " Calcium Serum 10.2 mg/dL 8.8 - 10.0",
    " Total Cholesterol 160 mg/dL 0 - 200",
    " Triglycerides 167.9 mg/dL 0 - 150",
    " Non HDL Cholesterol 116.3 mg/dL 0 - 130",       # derived: must be ignored
    " Vitamin D 25 - Hydroxy 79.2 ng/mL 30 - 100",    # digits inside the label
]


def test_single_row_layout_is_extracted():
    found, _conv = extract_biomarkers_from_lines(_ROW_LINES)
    assert found["hemoglobin"] == 10.7
    assert found["wbc"] == 12.6                        # "TLC"
    assert found["platelets"] == 346.0
    assert found["fasting_glucose_mgdl"] == 152.2
    assert found["calcium"] == 10.2
    assert found["total_chol_mgdl"] == 160.0
    assert found["triglycerides_mgdl"] == 167.9


def test_hba1c_label_containing_haemoglobin_is_not_haemoglobin():
    """"Glycosylated Hemoglobin (HbA1c)" must resolve to HbA1c, not haemoglobin."""
    found, _conv = extract_biomarkers_from_lines(_ROW_LINES)
    assert found["hba1c_pct"] == 6.8
    assert found["hemoglobin"] == 10.7                 # the real haemoglobin row


def test_derived_rows_are_never_captured():
    found, _conv = extract_biomarkers_from_lines(_ROW_LINES)
    # 148.46 (estimated average glucose) must not be mistaken for fasting glucose,
    # and 116.3 (non-HDL) must not be mistaken for HDL.
    assert found["fasting_glucose_mgdl"] == 152.2
    assert found.get("hdl_mgdl") is None


def test_label_digits_do_not_shadow_the_reading():
    """"Vitamin D 25 - Hydroxy 79.2 ng/mL" reads 79.2, not the 25 in the name."""
    found, _conv = extract_biomarkers_from_lines(_ROW_LINES)
    assert abs(found["vitamin_d"] - 79.2 * 2.496) < 0.01


def test_abnormality_flag_between_value_and_unit():
    """A flagged result ("10.7 L* g/dL") must not be dropped for having no unit."""
    found, _conv = extract_biomarkers_from_lines(
        ["Hemoglobin", "Spectrophotometry (Cyanide Free)", "10.7 L* g/dL 12.0 - 15.0",
         "PCV", "Calculated", "33.2 L* % 36 - 46",
         "Serum Triglycerides", "Method: Enzymatic", "167.9 H* mg/dL <150"])
    assert found["hemoglobin"] == 10.7
    assert found["hematocrit"] == 33.2                 # "PCV"
    assert found["triglycerides_mgdl"] == 167.9


def test_reference_bounds_are_not_read_as_results():
    """Numbers behind a comparator or dash are ranges, never readings."""
    found, _conv = extract_biomarkers_from_lines(
        ["Glucose, Fasting", "Method: Hexokinase",
         "Normal : < 100 mg/dL", "Impaired fasting glucose : 100 - 125 mg/dL"])
    assert "fasting_glucose_mgdl" not in found


def test_value_is_not_paired_with_a_distant_label():
    """A reading may not attach to a label that a different test has superseded."""
    found, _conv = extract_biomarkers_from_lines(
        ["Serum Creatinine", "Serum Potassium", "4.13 mmol/L 3.5 - 5.5"])
    assert found.get("creatinine") is None
    assert found["potassium"] == 4.13


# --- patient demographics from the report header ----------------------------- #
_HEADER_SHAPES = [
    ("Female 61 yrs", 61, "female"),
    ("Male, 60 Yrs", 60, "male"),
    ("Gender: Female Age: 61 Yrs Patient ID: 17671820", 61, "female"),
    ("DOB/Age/Gender : 61 Y/Female", 61, "female"),
    ("Age/Gender : 60Y 0M 0D /Male", 60, "male"),
    ("Sex : Male   Age : 45 Years", 45, "male"),
]


def test_demographics_from_header_shapes():
    for line, age, sex in _HEADER_SHAPES:
        got = extract_demographics([line])
        assert got == {"age": age, "sex": sex}, (line, got)


def test_demographics_ignores_reference_and_prose_lines():
    """Guideline cut-offs mention ages too; none of them describe the patient."""
    for line in ["Non diabetic adults >=18 years <5.7",
                 "Age > 19 years", "Age < 19 years",
                 "above 20 years of age must be screened for abnormal lipid levels.",
                 "Age (Years) Male"]:
        assert extract_demographics([line]) == {}, line


def test_demographics_prefers_the_line_carrying_both():
    """A paired header wins over a stray age or sex elsewhere on the page."""
    got = extract_demographics(
        ["Total Cholesterol 160 mg/dL", "78 yrs", "Gender: Female Age: 61 Yrs"])
    assert got == {"age": 61, "sex": "female"}


def test_demographics_absent_when_unreadable():
    assert extract_demographics(["Hemoglobin 10.7 g/dL 12.0 - 15.0"]) == {}


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
