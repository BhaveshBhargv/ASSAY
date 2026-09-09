"""
report_parser.py — turn an uploaded blood report into a {code: value} draft.

Three input shapes, one output. CSV/JSON are structured and reliable. PDF uses a
stateful, unit-anchored text extractor:

  * track the "current analyte" as we scan, skipping Method/Machine/reference
    noise lines, so a value is paired with the right label even when they are
    separated in the extracted text;
  * only accept a value whose printed unit is valid for that biomarker
    (stops "199 pg/ml" being read as glucose), converting to the app's canonical
    unit where needed (e.g. Vitamin D ng/mL → nmol/L);
  * reject physiologically impossible values (plausibility bounds).

Results are always best-effort and should be reviewed before use.

    parse_upload(filename, data) -> ParseResult(biomarkers, demographics, notes)
"""
from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass, field

from .catalog import LABEL_PRIORITY, SYNONYMS, label_index
from .units import CANON_UNIT, detect_unit, is_plausible, to_canonical

# Synonyms used to detect the analyte a line is labelling.
_LABEL_TERMS: list[tuple[str, str]] = sorted(
    [(alt.lower(), code) for code, alts in SYNONYMS.items() for alt in alts],
    key=lambda t: -len(t[0]))

# Tokens that mean "this line is not an analyte result": method/instrument notes,
# reference-range and interpretation prose, and derived rows that borrow a tracked
# analyte's name (their numbers must never be captured as a primary measurement).
_NOISE_LABEL = (
    "method", "machine", "description", "department", "sample type", "barcode",
    "booking", "patient name", "biological reference", "reference interval",
    "interpretation", "impression", "ratio", "non-hdl", "non hdl", "vldl",
    "v.l.d.l", "apolipo", "/hdl", "a/g", "sgot/sgpt", "desirable", "optimal",
    "normal range",
    # derived quantities that contain a tracked analyte's name
    "estimated average glucose", "average glucose", "mean plasma glucose",
    "bilirubin direct", "bilirubin indirect", "direct bilirubin", "indirect bilirubin",
    "globulin", "bun/creatinine", "urea /", "egfr", "rdw-sd",
    "mean platelet", "mpv", "pdw", "p-lcc",
    # differential counts — not tracked analytes, but share cell-name wording
    "neutrophil", "lymphocyte", "monocyte", "eosinophil", "basophil", "immature",
)

# How many following lines may sit between a test name and its value in the
# stacked layout (method / instrument / label-continuation lines).
_LOOKAHEAD = 4

# A numeric token that is not glued to another digit or a decimal point.
_VALUE = re.compile(r"(?<![\d.])(\d{1,4}(?:\.\d+)?)(?![\d.])")

# A line whose FIRST token is the reading, optionally behind an H/L abnormality
# flag: "10.7 L* g/dL 12.0 - 15.0", "160 mg/dL <200".
_LEADING_VALUE = re.compile(r"^\s*(?:[HL]\*?\s+|\*\s*)?(\d{1,4}(?:\.\d+)?)(\D.*)?$")

# A number introduced by one of these is a bound ("< 100", "0 - 200"), not a result.
_BOUND_CHARS = "<>=≥≤-–—/±"


def match_label_span(low_line: str) -> tuple[str, int] | None:
    """Which analyte a (lowercased) line labels, and the index the label ends at.

    Returns (code, end) or None. Long prose and noise lines are rejected so an
    interpretation paragraph can never masquerade as a result row. When several
    synonyms match, the highest-priority code wins and then the longest term —
    that ordering is what keeps "Glycosylated Hemoglobin (HbA1c)" from being read
    as `hemoglobin`.
    """
    if len(low_line) > 80 or any(tok in low_line for tok in _NOISE_LABEL):
        return None
    best: tuple[tuple[int, int], str, int] | None = None
    for term, code in _LABEL_TERMS:
        m = re.search(r"(?<![a-z])" + re.escape(term) + r"(?![a-z])", low_line)
        if m is None:
            continue
        rank = (LABEL_PRIORITY.get(code, 0), len(term))
        if best is None or rank > best[0]:
            best = (rank, code, m.end())
    return (best[1], best[2]) if best else None


def match_label(low_line: str) -> str | None:
    """The biomarker code a (lowercased) label line refers to, else None."""
    hit = match_label_span(low_line)
    return hit[0] if hit else None


def _accept(code: str, value: float, rest: str):
    """Unit-anchor and plausibility-check one candidate reading."""
    canon, converted_from = to_canonical(code, value, detect_unit(rest))
    if canon is None or not is_plausible(code, canon):
        return None
    return canon, converted_from


def _same_line_value(code: str, tail: str):
    """First unit-valid reading printed after the label on the same line.

    Every numeric token is tried in order, so digits belonging to the label itself
    ("Vitamin D 25 - Hydroxy 79.2 ng/mL") are skipped when their unit doesn't
    validate. Numbers introduced by a comparator or dash are reference bounds and
    are never candidates.
    """
    for m in _VALUE.finditer(tail):
        before = tail[:m.start()].rstrip()
        if before and before[-1] in _BOUND_CHARS:
            continue
        hit = _accept(code, float(m.group(1)), tail[m.end():])
        if hit:
            return hit
    return None


def _stacked_value(code: str, line: str):
    """Reading printed on its own line, e.g. "10.7 L* g/dL 12.0 - 15.0"."""
    m = _LEADING_VALUE.match(line)
    return _accept(code, float(m.group(1)), m.group(2) or "") if m else None


def extract_biomarkers_from_lines(lines: list[str]) -> tuple[dict[str, float], list[str]]:
    """Layout-agnostic, unit-anchored extraction. Returns (biomarkers, notes).

    Handles both layouts real lab PDFs use, in a single pass:

      A. one row per test  — "Total Cholesterol 160 mg/dL 0 - 200"
      B. stacked           — a test name, optional method/instrument lines, then
                             "160 mg/dL <200" on a line of its own

    A stacked value is only looked for within `_LOOKAHEAD` lines and the search
    stops as soon as a *different* test is named, so a value can never be paired
    with a label from further up the page. The first accepted reading for a code
    wins, letting the summary table at the front of a report take precedence over
    the detail pages that repeat it.
    """
    found: dict[str, float] = {}
    conversions: list[str] = []
    n = len(lines)

    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        hit = match_label_span(line.lower())
        if hit is None:
            continue
        code, end = hit
        if code in found:
            continue

        taken = _same_line_value(code, line[end:])                      # layout A
        if taken is None:                                               # layout B
            for j in range(i + 1, min(i + 1 + _LOOKAHEAD, n)):
                nxt = lines[j].strip()
                if not nxt:
                    continue
                other = match_label_span(nxt.lower())
                if other is not None and other[0] != code:
                    break            # a different test starts here — don't reach past it
                taken = _stacked_value(code, nxt[other[1]:] if other else nxt)
                if taken:
                    break
        if taken is None:
            continue

        canon, converted_from = taken
        found[code] = round(canon, 4)
        if converted_from:
            conversions.append(
                f"{code}: {converted_from} → {round(canon, 2)} {CANON_UNIT[code]}")
    return found, conversions


# --- demographics ---------------------------------------------------------- #
# Lab headers print age and sex in many shapes, all seen in real reports:
#   "Female 61 yrs"                    "Male, 60 Yrs"
#   "Gender: Female Age: 61 Yrs ..."   "Age/Gender : 60Y 0M 0D /Male"
#   "DOB/Age/Gender : 61 Y/Female"
_SEX_RE = re.compile(r"\b(female|male)\b", re.I)          # female first: it contains "male"
_AGE_KEYED = re.compile(r"\bage\b\D{0,20}?(\d{1,3})", re.I)     # "Age: 61", "Age/Gender : 60Y"
_AGE_UNITED = re.compile(r"\b(\d{1,3})\s*(?:y|yr|yrs|years)\b", re.I)   # "61 yrs"

# Guideline and reference rows also mention ages ("Age > 19 years", "adults >=18
# years"). They are prose about cut-offs, never the patient, so they are excluded.
_DEMO_NOISE = (
    "reference", "range", "normal", "adult", "screened", "recommend", "criteria",
    "interpretation", "above", "below", "goal", "target", "table", "population",
    "risk", "guideline", "category", "classification",
    "(years)", "(yrs)",          # a column heading such as "Age (Years) Male"
)
_ADULT_AGE = (18, 120)


def _demographics_from_line(line: str) -> tuple[int | None, str | None]:
    """Age and/or sex printed on one header line, or (None, None)."""
    low = line.lower()
    if any(tok in low for tok in _DEMO_NOISE) or "<" in line or ">" in line:
        return None, None
    sex_m = _SEX_RE.search(low)
    sex = sex_m.group(1) if sex_m else None

    age = None
    m = _AGE_KEYED.search(low) or (_AGE_UNITED.search(low) if sex else None)
    if m:
        value = int(m.group(1))
        if _ADULT_AGE[0] <= value <= _ADULT_AGE[1]:
            age = value
    return age, sex


def extract_demographics(lines: list[str]) -> dict:
    """Patient age and sex from a report header.

    A line carrying BOTH is trusted first — that is how every lab header prints
    them, and requiring the pair rules out stray words like a "Age (Years) Male"
    column heading. Only then does it fall back to the first standalone age and
    the first standalone sex. Ages outside the adult range are ignored rather than
    guessed at, so the form asks the user instead.
    """
    first_age = first_sex = None
    for raw in lines:
        line = raw.strip()
        if not line or len(line) > 120:
            continue
        age, sex = _demographics_from_line(line)
        if age is not None and sex is not None:
            return {"age": age, "sex": sex}
        first_age = first_age if first_age is not None else age
        first_sex = first_sex or sex

    out = {}
    if first_age is not None:
        out["age"] = first_age
    if first_sex:
        out["sex"] = first_sex
    return out


@dataclass
class ParseResult:
    biomarkers: dict[str, float] = field(default_factory=dict)
    demographics: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def parse_upload(filename: str, data: bytes) -> ParseResult:
    name = (filename or "").lower()
    if name.endswith(".json"):
        return _parse_json(data)
    if name.endswith(".csv"):
        return _parse_csv(data)
    if name.endswith(".pdf"):
        return _parse_pdf(data)
    return ParseResult(notes=[f"Unsupported file type: {filename}. Use CSV, JSON, or PDF."])


def _coerce(value) -> float | None:
    try:
        v = float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def _match(label: str) -> str | None:
    key = re.sub(r"\s+", " ", str(label).strip().lower())
    return label_index().get(key)


def _parse_json(data: bytes) -> ParseResult:
    try:
        obj = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return ParseResult(notes=[f"Could not read JSON: {exc}"])
    res = ParseResult()
    biomarkers = obj.get("biomarkers", obj) if isinstance(obj, dict) else {}
    res.demographics = obj.get("demographics", {}) if isinstance(obj, dict) else {}
    for label, value in (biomarkers.items() if isinstance(biomarkers, dict) else []):
        code = _match(label)
        v = _coerce(value) if code else None
        if code and v is not None:
            res.biomarkers[code] = v
    res.notes.append(f"Read {len(res.biomarkers)} biomarker value(s) from JSON.")
    return res


def _parse_csv(data: bytes) -> ParseResult:
    import pandas as pd

    try:
        df = pd.read_csv(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001
        return ParseResult(notes=[f"Could not read CSV: {exc}"])

    res = ParseResult()
    cols = {c.lower().strip(): c for c in df.columns}
    label_col = next((cols[c] for c in ("biomarker", "name", "test", "analyte", "marker") if c in cols), None)
    value_col = next((cols[c] for c in ("value", "result", "reading") if c in cols), None)
    if label_col and value_col:                              # long format
        for _, row in df.iterrows():
            code = _match(row[label_col])
            v = _coerce(row[value_col]) if code else None
            if code and v is not None:
                res.biomarkers[code] = v
    else:                                                     # wide format
        row = df.iloc[0] if len(df) else {}
        for label in df.columns:
            code = _match(label)
            v = _coerce(row[label]) if code else None
            if code and v is not None:
                res.biomarkers[code] = v
    res.notes.append(f"Read {len(res.biomarkers)} biomarker value(s) from CSV.")
    return res


def _parse_pdf(data: bytes) -> ParseResult:
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as exc:  # noqa: BLE001
        return ParseResult(notes=[f"Could not read PDF: {exc}"])

    if not text.strip():
        return ParseResult(notes=["No text found in PDF (it may be a scanned image). "
                                  "Enter values manually or upload CSV/JSON."])

    lines = text.splitlines()
    found, conversions = extract_biomarkers_from_lines(lines)
    demographics = extract_demographics(lines)
    notes = [f"Best-effort extraction found {len(found)} value(s). PDF layouts vary — "
             "please review every value before use."]
    if demographics:
        read = ", ".join(f"{k} {v}" for k, v in demographics.items())
        notes.append(f"Read patient details from the report header ({read}). "
                     "Correct them in the sidebar if they are wrong.")
    notes.extend(conversions)
    return ParseResult(biomarkers=found, demographics=demographics, notes=notes)
