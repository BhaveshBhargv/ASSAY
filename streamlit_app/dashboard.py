"""
dashboard.py — Assay: the Streamlit dashboard.

Thin orchestration only: wire the sidebar/form (ui.inputs) to the pipeline
(services.pipeline) and render results with presentational components
(ui.components, ui.charts). All heavy lifting lives in the Phase 3-6 engine.

Analysis is progressive: the assessment (risk summary, severity table, charts) is
computed fast and shown immediately, while the LLM recommendations are generated
in a background thread and streamed into their section when ready — so the LLM's
latency never blocks the rest of the page.

The entry file is named `dashboard.py` (not `app.py`) so it can't shadow the
`app` engine package on sys.path.

Run:  streamlit run streamlit_app/dashboard.py
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "streamlit_app"))
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Assay — Blood Report Insights", page_icon="🩸",
                   layout="wide", initial_sidebar_state="expanded")

# Streamlit Community Cloud provides secrets via st.secrets (not env vars). Bridge
# them into the environment so the os.getenv-based LLM providers pick them up.
# Runs before importing the engine config so model/key overrides take effect.
import os  # noqa: E402
try:
    for _k in ("OPENROUTER_API_KEY", "ASSAY_OPENROUTER_MODEL",
               "ANTHROPIC_API_KEY", "OLLAMA_BASE_URL"):
        if _k in st.secrets and not os.environ.get(_k):
            os.environ[_k] = str(st.secrets[_k])
except Exception:  # no secrets configured locally — that's fine
    pass

from app.recommend.config import DISCLAIMER          # noqa: E402
from services import pipeline                          # noqa: E402
from services.report_pdf import build_pdf             # noqa: E402
from ui import components as C                          # noqa: E402
from ui.charts import deviation_chart, rf_probability_chart  # noqa: E402
from ui.inputs import (                                 # noqa: E402
    Settings, collect_biomarkers, init_state, render_biomarker_form, render_sidebar,
)
from ui.theme import inject                             # noqa: E402

# Plotly: hide the hover toolbar/logo for a clean chart (tooltips kept).
_PLOTLY_CONFIG = {"displayModeBar": False, "displaylogo": False}

inject()
init_state()
settings = render_sidebar()


@st.cache_resource
def _executor() -> ThreadPoolExecutor:
    """One background worker pool for the whole app (LLM generation runs here)."""
    return ThreadPoolExecutor(max_workers=2)


def _md(html: str) -> None:
    st.markdown(html, unsafe_allow_html=True)


# --- header ---------------------------------------------------------------- #
_md('<div class="eyebrow">Clinical lifestyle dashboard</div>')
st.markdown("# Blood report insights")
_md('<div class="section-note">A blood panel, read against NICE / NHS / WHO '
    'guidance, turned into grounded lifestyle recommendations — never a diagnosis.</div>')

for note in st.session_state.get("parse_notes", []):
    st.info(note)

# --- input form ------------------------------------------------------------ #
have_results = st.session_state.get("assessment") is not None
with st.expander("Blood panel & values", expanded=True):
    render_biomarker_form()

left, right = st.columns([5, 1])

with right:
    settings.analyze = st.button(
        "Analyze",
        type="primary",
        use_container_width=True,
    )

# --- run: fast assessment now, LLM in the background ----------------------- #
if settings.analyze:
    biomarkers = collect_biomarkers()
    demographics = settings.demographics
    if not settings.demographics_ok:
        st.warning("Please enter the patient's age and sex in the sidebar before analysing.")
    elif not biomarkers:
        st.warning("Enter at least one biomarker value, or load a sample from the sidebar.")
    else:
        st.session_state["demographics"] = demographics
        with st.spinner("Reading the panel against the rules, model and guidelines…"):
            assessment = pipeline.assess(demographics, biomarkers, k=settings.k)
        st.session_state["assessment"] = assessment
        st.session_state["recs"] = None
        # Kick off the LLM in a background thread; the page renders without waiting.
        st.session_state["gen_future"] = _executor().submit(
            pipeline.generate, settings.provider, settings.model or None,
            demographics, assessment, settings.recheck)


# --- recommendations (background-filled) ----------------------------------- #
def _render_recs_content(recs) -> None:
    if recs.error:
        st.warning(recs.error)
        _md(C.callout("The risk summary, charts and evidence don't need the language "
                      "model — they're ready to download now.", "info"))
        return
    _md(C.recommendations(recs.report))
    au = recs.audit
    if au is not None:
        _md(f'<div class="section-note mono">Groundedness '
            f'{round(au.groundedness*100)}% of advice cites a retrieved guideline · '
            f'{len(au.dropped_items)} ungrounded item(s) removed · '
            f'diagnostic-language flags: {len(au.diagnostic_violations)}</div>')


@st.fragment(run_every=2)
def _poll_recs() -> None:
    """Self-refreshing island: shows a working state, swaps in the report when the
    background thread finishes, then triggers one full rerun to settle the page."""
    fut = st.session_state.get("gen_future")
    if fut is None:
        return
    if fut.done():
        st.session_state["recs"] = fut.result()
        st.session_state["gen_future"] = None
        st.rerun()
    else:
        _md(C.callout("Writing grounded recommendations… the risk summary, charts and "
                      "evidence above are ready to use now.", "info"))


def _render_recommendations() -> None:
    recs = st.session_state.get("recs")
    if recs is not None:
        _render_recs_content(recs)
    elif st.session_state.get("gen_future") is not None:
        _poll_recs()
    else:
        _md(C.callout("Recommendations generate automatically after you analyse.", "info"))


# --- export + evidence ----------------------------------------------------- #
def _render_export(assessment) -> None:
    recs = st.session_state.get("recs")
    report = recs.report if recs else None
    audit_pdf = recs.audit if recs else None
    demographics = st.session_state.get("demographics", {})
    pdf_bytes = build_pdf(demographics, assessment.rule_result, assessment.fused,
                          assessment.evidence, report, audit_pdf, DISCLAIMER)
    left, right = st.columns(2)
    with left:
        st.download_button("Download PDF report", data=pdf_bytes,
                           file_name="blood_report_recommendations.pdf",
                           mime="application/pdf", use_container_width=True)
    with right:
        st.page_link("pages/guidelines.py",
                     label=f"View {len(assessment.evidence)} retrieved guidelines",
                     icon="📖", use_container_width=True)


# --- results --------------------------------------------------------------- #
def render_results(assessment) -> None:
    fused = assessment.fused
    rule_result = assessment.rule_result
    recs = st.session_state.get("recs")
    audit = recs.audit if (recs and recs.report is not None) else None

    _md(C.hero(fused))

    _md(C.eyebrow("Risk summary"))
    _md(C.stat_tiles(fused, rule_result, audit))

    drivers_html = C.rf_drivers(fused)
    if drivers_html:
        _md(drivers_html)

    _md(C.eyebrow("Severity table", "Each analyte against its reference range. "
                  "The marker shows where the value sits; the band is the normal range."))
    _md(C.severity_table(rule_result))

    _md(C.eyebrow("Interactive charts"))
    dev = deviation_chart(rule_result)
    rf = rf_probability_chart(fused)
    if dev is None and rf is None:
        _md(C.callout("No out-of-range markers and no model probabilities to plot.", "info"))
    else:
        col_l, col_r = st.columns([3, 2])
        with col_l:
            st.caption("How far each flagged marker sits outside its reference range")
            if dev is not None:
                st.plotly_chart(dev, use_container_width=True, config=_PLOTLY_CONFIG)
            else:
                st.caption("All provided markers are within range.")
        with col_r:
            st.caption("Random Forest — probability by risk class")
            if rf is not None:
                st.plotly_chart(rf, use_container_width=True, config=_PLOTLY_CONFIG)
            else:
                st.caption("Model probabilities unavailable (rules-only).")

    signpost = C.signpost_callout(fused)
    if signpost:
        _md(signpost)

    _md(C.eyebrow("Recommendations"))
    _render_recommendations()

    _md(C.eyebrow("Export & evidence"))
    _render_export(assessment)
    _md(f'<div class="section-note" style="margin-top:1.2rem;">{DISCLAIMER}</div>')


assessment = st.session_state.get("assessment")
if assessment is not None:
    render_results(assessment)
else:
    _md(C.callout("Load a sample patient or enter a blood panel in the sidebar, then "
                  "press “Analyze blood report”. Recommendations follow automatically.",
                  "info"))
