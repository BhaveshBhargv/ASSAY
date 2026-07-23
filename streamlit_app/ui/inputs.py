"""
inputs.py — sidebar controls and the blood-panel entry form.

Owns all widget state. Loading a sample or reading a file writes values into the
controlled state (`age_value`, `sex_value`, and the ``in_<code>`` keys) and reruns,
so the form reflects them on the next pass. Age and sex are empty by default and
required before analysis — except when a sample patient is loaded (which fills
them). The model-only demographics (ethnicity/income/education) use fixed sensible
defaults and are no longer asked of the user.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import streamlit as st

from sample_data.samples import SAMPLES
from services.parsing import parse_upload
from ui.metadata import ALL_CODES, PANELS, display, reference_range

# Fixed model-only demographics (used by the RF; not clinically meaningful to ask).
_DEFAULT_EXTRA = {"eth_code": 3, "pir": 2.5, "educ_code": 4}
_SEX_OPTIONS = ["male", "female"]


@dataclass
class Settings:
    analyze: bool
    generate: bool
    provider: str
    model: str
    k: int
    recheck: bool
    source: str
    demographics: dict = field(default_factory=dict)
    demographics_ok: bool = False


def init_state() -> None:
    st.session_state.setdefault("age_value", None)
    st.session_state.setdefault("sex_value", None)
    st.session_state.setdefault("demo_extra", dict(_DEFAULT_EXTRA))
    st.session_state.setdefault("parse_notes", [])
    st.session_state.setdefault("assessment", None)
    st.session_state.setdefault("recs", None)


def _apply_values(bio: dict) -> None:
    """Push loaded biomarker values into the widget state, clearing unset fields."""
    for code in ALL_CODES:
        st.session_state[f"in_{code}"] = float(bio.get(code, 0.0) or 0.0)


def _apply_patient(age, sex, extra: dict | None) -> None:
    st.session_state["age_value"] = int(age) if age else None
    st.session_state["sex_value"] = sex if sex in _SEX_OPTIONS else None
    st.session_state["demo_extra"] = {**dict(_DEFAULT_EXTRA), **(extra or {})}


def render_sidebar() -> Settings:
    sb = st.sidebar
    sb.markdown("### 🩸 Assay")
    sb.caption("Blood report → evidence-based lifestyle guidance. Not a diagnosis.")

    sb.markdown("#### Blood report")
    source = sb.radio("Source", ["Sample patient", "Upload file", "Manual entry"],
                      label_visibility="collapsed")

    # Switching report source clears the patient details so they're re-entered
    # for the new report (a sample re-fills them on load).
    if st.session_state.get("_prev_source") != source:
        st.session_state["_prev_source"] = source
        st.session_state["age_value"] = None
        st.session_state["sex_value"] = None

    if source == "Sample patient":
        name = sb.selectbox("Sample", list(SAMPLES.keys()))
        if sb.button("Load sample", use_container_width=True):
            s = SAMPLES[name]
            d = s["demographics"]
            _apply_patient(d.get("age"), d.get("sex"),
                           {k: d[k] for k in ("eth_code", "pir", "educ_code") if k in d})
            _apply_values(s["biomarkers"])
            st.session_state["parse_notes"] = [f"Loaded sample: {name}."]
            st.rerun()

    elif source == "Upload file":
        up = sb.file_uploader("CSV, JSON or PDF", type=["csv", "json", "pdf"],
                              label_visibility="collapsed")
        sb.caption("PDF extraction is best-effort — review every value before analysing.")
        if up is not None and sb.button("Read file", use_container_width=True):
            res = parse_upload(up.name, up.getvalue())
            if res.biomarkers:
                _apply_values(res.biomarkers)
            d = res.demographics or {}
            if d.get("age") or d.get("sex"):
                _apply_patient(d.get("age") or st.session_state["age_value"],
                               d.get("sex") or st.session_state["sex_value"], None)
            st.session_state["parse_notes"] = res.notes
            st.rerun()
    else:
        if sb.button("Clear all fields", use_container_width=True):
            _apply_values({})
            _apply_patient(None, None, None)
            st.session_state["parse_notes"] = []
            st.rerun()

    # --- patient (age + sex required, except for samples) ----------------- #
    sb.markdown("#### Patient")
    age = sb.number_input("Age", min_value=18, max_value=100, step=1,
                          value=st.session_state.get("age_value"), placeholder="e.g. 54")
    st.session_state["age_value"] = age

    cur_sex = st.session_state.get("sex_value")
    sex = sb.selectbox("Sex", _SEX_OPTIONS, placeholder="Select sex",
                       index=_SEX_OPTIONS.index(cur_sex) if cur_sex in _SEX_OPTIONS else None)
    st.session_state["sex_value"] = sex

    if source != "Sample patient" and (age is None or sex is None):
        sb.caption("⚠️ Age and sex are required to analyse.")

    # Recommendations generate automatically (OpenRouter by default); no controls.
    generate, provider, model, recheck, k = True, "openrouter", "", False, 10

    sb.caption("Recommendations generate automatically after the analysis. "
               "Educational project — not for clinical use.")

    demographics = {"age": age, "sex": sex, **st.session_state["demo_extra"]}
    demographics_ok = (source == "Sample patient") or (age is not None and sex in _SEX_OPTIONS)
    return Settings(False, generate, provider, model.strip(), k, recheck,
                    source, demographics, demographics_ok)


def render_biomarker_form() -> None:
    """Six panel tabs of number inputs. Values live in ``in_<code>`` widget keys."""
    sex = st.session_state.get("sex_value") or "male"
    tabs = st.tabs([p for p, _ in PANELS])
    for tab, (_panel, codes) in zip(tabs, PANELS):
        with tab:
            cols = st.columns(3)
            for i, code in enumerate(codes):
                meta = display(code)
                rng = reference_range(code, sex)
                ref = ""
                if rng["low"] is not None and rng["high"] is not None:
                    ref = f"ref {_n(rng['low'])}–{_n(rng['high'])}"
                label = f"{meta['name']} ({meta['unit']})" if meta["unit"] else meta["name"]
                cols[i % 3].number_input(
                    label, min_value=0.0, step=0.1, format="%.2f",
                    key=f"in_{code}", help=f"{ref} · 0 = not provided" if ref else "0 = not provided",
                )


def collect_biomarkers() -> dict[str, float]:
    out: dict[str, float] = {}
    for code in ALL_CODES:
        v = st.session_state.get(f"in_{code}", 0.0)
        if v and v > 0:
            out[code] = float(v)
    return out


def _n(v) -> str:
    f = float(v)
    return str(int(f)) if f.is_integer() else f"{f:g}"
