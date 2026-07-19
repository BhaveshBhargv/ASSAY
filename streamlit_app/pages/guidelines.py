"""
pages/guidelines.py — full list of retrieved guideline passages.

A dedicated page (reachable from the dashboard's "View all retrieved guidelines"
link) that renders every NICE/NHS/WHO passage retrieved for the current
assessment. Reads the assessment from session state, so run an analysis first.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "streamlit_app"))
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Assay — Guidelines", page_icon="📖", layout="wide")

from ui import components as C  # noqa: E402
from ui.theme import inject     # noqa: E402

inject()


def _md(html: str) -> None:
    st.markdown(html, unsafe_allow_html=True)


_md('<div class="eyebrow">Evidence</div>')
st.markdown("# Retrieved guideline passages")

st.page_link("dashboard.py", label="Back to dashboard", icon="⬅️")

assessment = st.session_state.get("assessment")
if assessment is None:
    _md(C.callout("Run an analysis on the dashboard first, then return here to see the "
                  "guideline passages that informed it.", "info"))
    st.stop()

fused = assessment.fused
flagged = ", ".join(f["name"] for f in fused.flagged) or "none"
_md(f'<div class="section-note">These <b>{len(assessment.evidence)}</b> passages were '
    f'retrieved for the assessed pattern (<b>{fused.severity}</b>; flagged: {flagged}). '
    "They are the only evidence the generated recommendations are allowed to cite — every "
    "piece of advice references one of these by its id.</div>")

_md(C.evidence_cards(assessment.evidence))

_md(f'<div class="section-note" style="margin-top:1.6rem;">Sources: NICE, NHS, WHO — '
    "paraphrased with attribution. Educational project; not a diagnosis.</div>")
