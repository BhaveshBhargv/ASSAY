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

from .catalog import SYNONYMS, label_index
from .units import CANON_UNIT, detect_unit, is_plausible, to_canonical

_NUM = r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?"

# A value line starts with the reading; the rest holds the unit (+ maybe a range).
_NUM_LEAD = re.compile(r"^\s*([0-9]{1,4}(?:\.[0-9]+)?)(\D.*)?$")

# Synonyms (longest first) used to detect the analyte a line is labelling.
_LABEL_TERMS: list[tuple[str, str]] = sorted(
    [(alt.lower(), code) for code, alts in SYNONYMS.items() for alt in alts],
    key=lambda t: -len(t[0]))

# Tokens that mean "this line is not an analyte label" (method/machine notes,
# reference-range descriptors, and derived/ratio rows).
_NOISE_LABEL = (
    "method", "machine", "description", "department", "sample type", "barcode",
    "booking", "patient name", "biological reference", "reference interval",
    "interpretation", "impression", "ratio", "non-hdl", "non hdl", "vldl",
    "apolipo", "/hdl", "a/g", "sgot/sgpt", "desirable", "optimal", "normal range",
)


def match_label(low_line: str) -> str | None:
    """Return the biomarker code a (lowercased) label line refers to, else None.
    Long prose lines and noise lines are ignored to avoid false labels."""
    if len(low_line) > 60 or any(tok in low_line for tok in _NOISE_LABEL):
        return None
    for term, code in _LABEL_TERMS:
        if re.search(r"(?<![a-z])" + re.escape(term) + r"(?![a-z])", low_line):
            return code
    return None


def extract_biomarkers_from_lines(lines: list[str]) -> tuple[dict[str, float], list[str]]:
    """Stateful, unit-anchored extraction. Returns (biomarkers, conversion notes)."""
    found: dict[str, float] = {}
    conversions: list[str] = []
    current: str | None = None

    for raw in lines:
        ln = raw.strip()
        if not ln:
            continue
        lab = match_label(ln.lower())
        if lab is not None:
            current = lab
        if current is None or current in found:
            continue
        m = _NUM_LEAD.match(ln)
        if not m:
            continue
        value = float(m.group(1))
        unit = detect_unit(m.group(2) or "")
        canon, converted_from = to_canonical(current, value, unit)
        if canon is None or not is_plausible(current, canon):
            continue
        found[current] = round(canon, 4)
        if converted_from:
            conversions.append(
                f"{current}: {value:g} {converted_from} → {round(canon, 2)} {CANON_UNIT[current]}")
        current = None  # consume so the next value can't refill this slot
    return found, conversions


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

    found, conversions = extract_biomarkers_from_lines(text.splitlines())
    notes = [f"Best-effort extraction found {len(found)} value(s). PDF layouts vary — "
             "please review every value before use."]
    notes.extend(conversions)
    return ParseResult(biomarkers=found, notes=notes)
