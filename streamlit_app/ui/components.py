"""
components.py — presentational HTML fragments for the dashboard.

Each function returns an HTML string (rendered by the caller via
st.markdown(..., unsafe_allow_html=True)). No Streamlit calls here, so the visual
layer stays pure and easy to reason about. Colours come from theme.py.
"""
from __future__ import annotations

import html

from app.recommend.report import ADVICE_SECTIONS, SECTION_TITLES
from ui.theme import SEVERITY, STATUS_COLOR

_SEV_POS = {"normal": 16.5, "borderline": 50.0, "serious": 83.0}
_SEV_WORD = {"normal": "Low risk", "borderline": "Some risk markers", "serious": "Higher risk pattern"}
_SEV_LEDE = {
    "normal": "Your results don't show a raised-risk pattern. The guidance below "
              "helps you maintain it.",
    "borderline": "A few results sit outside their usual range. Small, evidence-based "
                  "lifestyle changes can help move them back.",
    "serious": "Several results, or their combination, point to a higher-risk pattern. "
               "The guidance below is worth acting on and discussing with your GP.",
}


def _esc(s) -> str:
    return html.escape(str(s), quote=True)


def _fmt(v) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return _esc(v)
    return str(int(f)) if f.is_integer() else f"{f:g}"


def eyebrow(label: str, note: str = "") -> str:
    out = f'<div class="eyebrow">{_esc(label)}</div>'
    if note:
        out += f'<div class="section-note">{_esc(note)}</div>'
    return out


def chip(status: str) -> str:
    color = STATUS_COLOR.get(status, "#6B7B86")
    return (f'<span class="chip" style="color:{color};background:{color}14;'
            f'border-color:{color}33;"><span class="cdot" style="background:{color};">'
            f'</span>{_esc(status)}</span>')


# --- hero ------------------------------------------------------------------ #

def hero(fused) -> str:
    sev = fused.severity
    color = SEVERITY.get(sev, SEVERITY["normal"])
    pos = _SEV_POS.get(sev, 16.5)

    urgent = ('<div class="urgent-tag">▲ Some results warrant prompt clinical '
              'attention — please contact your GP.</div>') if fused.urgent_referral else ""
    esc_note = ""
    if fused.escalated_by_rf:
        esc_note = ('<div class="section-note" style="margin-top:.6rem;">The model '
                    'flagged the overall <b>pattern</b> as higher risk even though '
                    'individual markers look near-normal.</div>')

    meter = f"""
    <div class="meter">
      <div class="track">
        <div class="live" style="width:{pos}%;"></div>
        <div class="marker" style="left:{pos}%;"></div>
      </div>
      <div class="scale"><span>Low</span><span>Borderline</span><span>Higher</span></div>
    </div>"""

    return f"""
    <div class="hero" style="--sev:{color};">
      <div class="kicker">Overall risk pattern · not a diagnosis</div>
      <div class="verdict"><span class="dot"></span>{_esc(_SEV_WORD.get(sev, sev.title()))}</div>
      <div class="lede">{_esc(_SEV_LEDE.get(sev, ''))}</div>
      {urgent}{esc_note}{meter}
    </div>"""


# --- stat tiles ------------------------------------------------------------ #

def stat_tiles(fused, rule_result, audit=None) -> str:
    n_flagged = len(fused.flagged)
    n_signpost = len(fused.signpost)
    rf = fused.rf_severity or "—"
    rules = fused.rule_severity
    rf_sub = "escalated" if fused.escalated_by_rf else "agrees with rules" if fused.rf_severity else "rules only"
    ground = f'{round(audit.groundedness * 100)}%' if audit is not None else "—"

    tiles = [
        ("Fused severity", fused.severity.title(), f"rules: {rules} · model: {rf}"),
        ("Actionable flags", str(n_flagged), "markers with lifestyle actions"),
        ("Model read", (fused.rf_severity or "n/a").title(), rf_sub),
        ("Evidence grounding", ground, "advice tied to guidelines" if audit else "run recommendations"),
    ]
    cells = "".join(
        f'<div class="tile"><div class="t-label">{_esc(l)}</div>'
        f'<div class="t-value">{_esc(v)}</div><div class="t-sub">{_esc(s)}</div></div>'
        for l, v, s in tiles
    )
    _ = n_signpost
    return f'<div class="tiles">{cells}</div>'


# --- severity table with range strips (signature) -------------------------- #

def _strip(value: float, low, high, color: str) -> str:
    """Compute a reference-range strip: reference band + patient marker."""
    lo = low if isinstance(low, (int, float)) else None
    hi = high if isinstance(high, (int, float)) else None
    if lo is None and hi is None:
        dmin, dmax, blo, bhi = value * 0.5, value * 1.5 or 1, None, None
    elif lo is not None and hi is not None:
        span = max(hi - lo, 1e-6)
        dmin, dmax, blo, bhi = lo - span * 0.7, hi + span * 0.7, lo, hi
    elif hi is not None:
        dmin, dmax, blo, bhi = 0, hi * 1.8, 0, hi
    else:  # only a lower bound
        dmax = lo * 1.8 if lo else 1
        dmin, blo, bhi = 0, lo, dmax

    def pct(x):
        return max(0.0, min(100.0, (x - dmin) / (dmax - dmin) * 100))

    band = ""
    if blo is not None and bhi is not None:
        band = (f'<div class="band" style="left:{pct(blo):.1f}%;'
                f'width:{max(pct(bhi) - pct(blo), 1):.1f}%;"></div>')
    p = pct(value)
    point = f'<div class="pt" style="left:{p:.1f}%;background:{color};"></div>'
    return f'<div class="strip">{band}{point}</div>'


def severity_table(rule_result) -> str:
    rows = sorted(
        rule_result.biomarkers,
        key=lambda b: ({"serious": 0, "borderline": 1, "normal": 2}.get(b.severity.value, 3),
                       b.name),
    )
    if not rows:
        return '<div class="panel">No biomarker values were provided.</div>'

    html_rows = []
    for b in rows:
        status = b.status.value
        color = STATUS_COLOR.get(status, "#6B7B86")
        rng = b.reference_range or {}
        low, high = rng.get("low"), rng.get("high")
        cat = "flag-only · clinician" if b.tier == "flag_only" else b.category
        ref = ""
        if isinstance(low, (int, float)) and isinstance(high, (int, float)):
            ref = f'{_fmt(low)}–{_fmt(high)}'
        html_rows.append(
            f'<div class="bm-row">'
            f'  <div class="bm-name">{_esc(b.name)}<span class="bm-cat">{_esc(cat)}</span></div>'
            f'  <div class="bm-val">{_fmt(b.value)}<span class="u"> {_esc(b.unit)}</span></div>'
            f'  {_strip(b.value, low, high, color)}'
            f'  <div class="bm-status">{chip(status)}'
            f'    <div class="bm-cat mono" style="text-align:right;margin-top:.25rem;">'
            f'ref {ref}</div></div>'
            f'</div>'
        )
    return f'<div class="panel">{"".join(html_rows)}</div>'


# --- evidence -------------------------------------------------------------- #

def evidence_cards(evidence) -> str:
    if not evidence:
        return '<div class="panel">No guideline passages retrieved.</div>'
    cards = []
    for e in evidence:
        title = f' · {_esc(e.title)}' if e.title else ""
        cards.append(
            f'<div class="ev-card"><div class="ev-head">'
            f'<span class="ev-badge">{_esc(e.id)}</span>'
            f'<span class="ev-src">{_esc(e.citation)}{title}</span></div>'
            f'<div class="ev-text">{_esc(e.text)}</div></div>'
        )
    return "".join(cards)


# --- recommendations ------------------------------------------------------- #

def recommendations(report) -> str:
    blocks = [f'<div class="panel"><div class="lede">{_esc(report.explanation)}</div></div>']
    for sec in ADVICE_SECTIONS:
        items = report.section(sec)
        if not items:
            continue
        rows = []
        for it in items:
            cites = ", ".join(it.evidence)
            rows.append(
                f'<div class="rec-item"><div class="rec-advice">{_esc(it.advice)}</div>'
                f'<div class="rec-why">Why: {_esc(it.rationale)}</div>'
                f'<div class="rec-cite">Evidence: {_esc(cites)}</div></div>'
            )
        blocks.append(
            f'<div class="panel" style="margin-top:.7rem;">'
            f'<div class="rec-sec-title">{_esc(SECTION_TITLES[sec])}</div>'
            f'{"".join(rows)}</div>'
        )
    return "".join(blocks)


def callout(text: str, kind: str = "info") -> str:
    return f'<div class="callout {kind}">{_esc(text)}</div>'


def signpost_callout(fused) -> str:
    if not fused.signpost:
        return ""
    names = ", ".join(f"{s['name']} ({s['status']})" for s in fused.signpost)
    return callout(
        f"These results are outside their usual range and should be discussed with a "
        f"clinician — they are not addressed by lifestyle advice: {names}.", "clin")
