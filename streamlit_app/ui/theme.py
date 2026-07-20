"""
theme.py — the "Assay" design system: palette tokens + injected CSS.

One place owns the visual identity. `PALETTE` / `SEVERITY` are shared with the
Plotly charts so the whole app stays on-brand. `inject()` loads the fonts and the
component styles (cards, chips, the signature reference-range strip, evidence
cards, stat tiles, hero).
"""
from __future__ import annotations

import streamlit as st

# --- tokens (single source of colour truth) -------------------------------- #
PALETTE = {
    "ink": "#10202E",
    "ink2": "#3C4E5A",
    "muted": "#6B7B86",
    "paper": "#EEF2F4",
    "panel": "#FFFFFF",
    "line": "#DBE3E8",
    "teal": "#0E7C86",
    "teal_dark": "#0B646C",
    "teal_tint": "#E3F0F1",
}

# Severity is the domain's own three-colour language (+ urgent).
SEVERITY = {
    "normal": "#1F9D74",
    "borderline": "#C8871B",
    "serious": "#C24A57",
    "urgent": "#B0323F",
}

# Per-status accents used by chips / strips (maps the 5 clinical statuses).
STATUS_COLOR = {
    "normal": SEVERITY["normal"],
    "borderline": SEVERITY["borderline"],
    "low": SEVERITY["borderline"],
    "high": SEVERITY["borderline"],
    "severe": SEVERITY["serious"],
}

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700;800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
  --ink:#10202E; --ink2:#3C4E5A; --muted:#6B7B86;
  --paper:#EEF2F4; --panel:#FFFFFF; --line:#DBE3E8;
  --teal:#0E7C86; --teal-dark:#0B646C; --teal-tint:#E3F0F1;
  --normal:#1F9D74; --borderline:#C8871B; --serious:#C24A57; --urgent:#B0323F;
  --shadow-sm: 0 1px 2px rgba(16,32,46,.05), 0 1px 3px rgba(16,32,46,.04);
  --shadow-md: 0 4px 14px rgba(16,32,46,.07), 0 2px 4px rgba(16,32,46,.05);
  --shadow-lg: 0 14px 34px rgba(16,32,46,.11);
}

/* base type — a faint clinical-teal wash gives the paper depth */
html, body, .stApp, [data-testid="stAppViewContainer"] {
  background:
    radial-gradient(1100px 480px at 82% -10%, rgba(14,124,134,.06), transparent 60%),
    linear-gradient(180deg, #F1F4F6 0%, var(--paper) 42%);
  background-attachment: fixed;
  font-family: 'IBM Plex Sans', system-ui, sans-serif;
  color: var(--ink);
}
h1,h2,h3,h4 { font-family:'Archivo', sans-serif; letter-spacing:-0.01em; color:var(--ink); }
h1 { font-size: 2.15rem; }
@keyframes assay-rise { from { opacity:0; transform: translateY(10px);} to { opacity:1; transform:none; } }

/* keep the default header out of the way, tighten top padding */
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1180px; }

/* mono utility */
.mono { font-family:'IBM Plex Mono', monospace; font-variant-numeric: tabular-nums; }

/* ---- section eyebrow ---------------------------------------------------- */
.eyebrow {
  font-family:'IBM Plex Mono', monospace; font-size:.72rem; font-weight:600;
  letter-spacing:.18em; text-transform:uppercase; color:var(--teal);
  display:flex; align-items:center; gap:.55rem; margin: 1.9rem 0 .7rem;
}
.eyebrow::before { content:""; width:16px; height:2px; background:var(--teal); display:inline-block; }
.section-note { color:var(--muted); font-size:.9rem; margin:-.35rem 0 1rem; }

/* ---- panels / cards ----------------------------------------------------- */
.panel {
  background:var(--panel); border:1px solid var(--line); border-radius:16px;
  padding:1.15rem 1.3rem; box-shadow: var(--shadow-sm);
}

/* ---- hero verdict ------------------------------------------------------- */
.hero {
  background:
    radial-gradient(680px 260px at 93% -35%,
      color-mix(in srgb, var(--sev, var(--teal)) 16%, transparent), transparent 70%),
    var(--panel);
  border:1px solid var(--line); border-radius:20px;
  padding:1.7rem 1.85rem; position:relative; overflow:hidden;
  box-shadow: var(--shadow-md); animation: assay-rise .45s ease both;
}
.hero::before {
  content:""; position:absolute; left:0; top:0; bottom:0; width:6px;
  background:linear-gradient(180deg, var(--sev,var(--teal)),
    color-mix(in srgb, var(--sev,var(--teal)) 60%, #10202E));
}
.hero .kicker {
  font-family:'IBM Plex Mono',monospace; font-size:.72rem; letter-spacing:.18em;
  text-transform:uppercase; color:var(--muted);
}
.hero .verdict {
  font-family:'Archivo',sans-serif; font-weight:800; font-size:2.55rem;
  line-height:1.04; margin:.3rem 0 .4rem; color:var(--ink);
}
.hero .verdict .dot {
  display:inline-block; width:14px; height:14px; border-radius:50%;
  background:var(--sev,var(--teal)); margin-right:.55rem; vertical-align:middle;
}
.hero .lede { color:var(--ink2); font-size:1.02rem; max-width:62ch; }
.hero .urgent-tag {
  display:inline-flex; align-items:center; gap:.4rem; margin-top:.9rem;
  background:#FBEAEC; color:var(--urgent); border:1px solid #F1C7CD;
  padding:.32rem .7rem; border-radius:999px; font-size:.82rem; font-weight:600;
}

/* ---- severity meter (signature, hero) ----------------------------------- */
.meter { margin-top:1.1rem; }
.meter .track {
  position:relative; height:12px; border-radius:999px;
  background:linear-gradient(90deg, var(--normal) 0 33%, var(--borderline) 33% 66%, var(--serious) 66% 100%);
  opacity:.28;
}
.meter .track .live {
  position:absolute; top:0; bottom:0; left:0; border-radius:999px;
  background:linear-gradient(90deg, var(--normal) 0 33%, var(--borderline) 33% 66%, var(--serious) 66% 100%);
}
.meter .marker {
  position:absolute; top:-5px; width:3px; height:22px; border-radius:2px;
  background:var(--ink); box-shadow:0 0 0 3px var(--panel);
}
.meter .scale {
  display:flex; justify-content:space-between; margin-top:.5rem;
  font-family:'IBM Plex Mono',monospace; font-size:.68rem; letter-spacing:.12em;
  text-transform:uppercase; color:var(--muted);
}

/* ---- stat tiles --------------------------------------------------------- */
.tiles { display:grid; grid-template-columns:repeat(4,1fr); gap:.85rem; }
.tile {
  background:var(--panel); border:1px solid var(--line); border-radius:14px;
  padding:1rem 1.05rem; position:relative; overflow:hidden; box-shadow:var(--shadow-sm);
  transition: transform .18s ease, box-shadow .18s ease;
}
.tile::before { content:""; position:absolute; left:0; right:0; top:0; height:3px;
  background:linear-gradient(90deg, var(--teal), var(--teal-tint)); }
.tile:hover { transform: translateY(-2px); box-shadow: var(--shadow-md); }
.tile .t-label {
  font-family:'IBM Plex Mono',monospace; font-size:.68rem; letter-spacing:.14em;
  text-transform:uppercase; color:var(--muted);
}
.tile .t-value { font-family:'Archivo',sans-serif; font-weight:700; font-size:1.65rem; margin-top:.25rem; }
.tile .t-sub { font-size:.8rem; color:var(--ink2); }

/* ---- chips -------------------------------------------------------------- */
.chip {
  display:inline-flex; align-items:center; gap:.35rem; padding:.16rem .55rem;
  border-radius:999px; font-size:.74rem; font-weight:600;
  font-family:'IBM Plex Mono',monospace; letter-spacing:.02em;
  border:1px solid transparent; white-space:nowrap;
}
.chip .cdot { width:7px; height:7px; border-radius:50%; }

/* ---- biomarker range strip (signature) ---------------------------------- */
.bm-row { display:grid; grid-template-columns: 168px 96px 1fr 128px; gap:.9rem;
  align-items:center; padding:.62rem 0; border-top:1px solid var(--line); }
.bm-row:first-child { border-top:none; }
.bm-name { font-weight:600; font-size:.92rem; }
.bm-name .bm-cat { font-family:'IBM Plex Mono',monospace; font-size:.66rem;
  color:var(--muted); text-transform:uppercase; letter-spacing:.1em; display:block; }
.bm-val { font-family:'IBM Plex Mono',monospace; font-weight:600; font-size:.98rem; text-align:right; }
.bm-val .u { color:var(--muted); font-weight:400; font-size:.78rem; }
.bm-row { transition: background .15s ease; border-radius:8px; }
.bm-row:hover { background:#F6F9FA; }
.strip { position:relative; height:9px; background:#E9EEF1; border-radius:999px;
  box-shadow: inset 0 1px 2px rgba(16,32,46,.05); }
.strip .band { position:absolute; top:0; bottom:0; background:var(--teal-tint);
  border-left:1px solid #BFD9DB; border-right:1px solid #BFD9DB; border-radius:2px; }
.strip .pt { position:absolute; top:-4px; width:15px; height:17px; border-radius:5px;
  transform:translateX(-7px);
  box-shadow:0 0 0 3px var(--panel), 0 2px 5px rgba(16,32,46,.30); }
.bm-status { text-align:right; }

/* ---- evidence cards ----------------------------------------------------- */
.ev-card { background:var(--panel); border:1px solid var(--line); border-radius:13px;
  padding:.95rem 1.1rem; margin-bottom:.75rem; border-left:3px solid var(--teal);
  box-shadow:var(--shadow-sm); transition: transform .18s ease, box-shadow .18s ease; }
.ev-card:hover { transform: translateY(-2px); box-shadow: var(--shadow-md); }
.ev-head { display:flex; align-items:center; gap:.6rem; margin-bottom:.4rem; }
.ev-badge { font-family:'IBM Plex Mono',monospace; font-size:.7rem; font-weight:600;
  background:var(--ink); color:#fff; padding:.14rem .5rem; border-radius:6px; letter-spacing:.05em; }
.ev-src { font-family:'IBM Plex Mono',monospace; font-size:.74rem; color:var(--teal-dark); font-weight:600; }
.ev-title { font-weight:600; font-size:.92rem; }
.ev-text { color:var(--ink2); font-size:.9rem; line-height:1.5; }

/* ---- recommendation items ---------------------------------------------- */
.rec-sec-title { font-family:'Archivo',sans-serif; font-weight:700; font-size:1.05rem;
  margin:.2rem 0 .5rem; display:flex; align-items:center; gap:.5rem; }
.rec-item { border-left:2px solid var(--line); padding:.15rem 0 .7rem .9rem; margin-left:.2rem; }
.rec-advice { font-weight:600; font-size:.96rem; }
.rec-why { color:var(--ink2); font-size:.88rem; margin-top:.15rem; }
.rec-cite { font-family:'IBM Plex Mono',monospace; font-size:.7rem; color:var(--teal-dark);
  margin-top:.3rem; }
.callout { border-radius:12px; padding:.9rem 1.05rem; font-size:.9rem; }
.callout.clin { background:#FFF6E9; border:1px solid #F0DCBA; color:#7A5A16; }
.callout.info { background:var(--teal-tint); border:1px solid #C3E0E2; color:var(--teal-dark); }

/* ---- model drivers (per-patient SHAP) ---------------------------------- */
.panel.drivers { margin-top:.7rem; border-left:3px solid var(--borderline); }
.drv-row { display:grid; grid-template-columns: minmax(9rem, 40%) 1fr; align-items:center;
  gap:.9rem; padding:.4rem 0; }
.drv-row + .drv-row { border-top:1px dashed var(--line); }
.drv-name { font-weight:600; font-size:.92rem; color:var(--ink);
  display:flex; align-items:baseline; gap:.5rem; }
.drv-val { font-family:'IBM Plex Mono',monospace; font-size:.78rem; font-weight:500;
  color:var(--muted); font-variant-numeric: tabular-nums; }
.drv-bar { height:8px; border-radius:999px; background:color-mix(in srgb, var(--borderline) 12%, transparent);
  overflow:hidden; }
.drv-fill { height:100%; border-radius:999px;
  background:linear-gradient(90deg, color-mix(in srgb, var(--borderline) 70%, #E8B85C), var(--borderline));
  animation: drv-grow .5s ease both; }
@keyframes drv-grow { from { transform: scaleX(0); transform-origin:left; } to { transform:none; } }

/* ---- streamlit widget nudges ------------------------------------------- */
.stButton>button, .stDownloadButton>button {
  font-family:'IBM Plex Sans',sans-serif; font-weight:600; border-radius:11px;
  border:1px solid var(--teal-dark);
  background:linear-gradient(180deg, #12909B, var(--teal)); color:#fff; padding:.55rem 1.05rem;
  box-shadow:var(--shadow-sm);
  transition: transform .15s ease, box-shadow .15s ease, filter .15s ease;
}
.stButton>button:hover, .stDownloadButton>button:hover {
  filter:brightness(1.06); transform:translateY(-1px); box-shadow:var(--shadow-md); color:#fff;
}
/* page-link styled as a secondary button (sits next to Download) */
[data-testid="stPageLink"] a {
  display:flex; align-items:center; justify-content:center; gap:.45rem;
  font-family:'IBM Plex Sans',sans-serif; font-weight:600; border-radius:11px;
  border:1px solid #BFD9DB; color:var(--teal-dark)!important; text-decoration:none;
  background:var(--teal-tint); padding:.55rem 1.05rem; box-shadow:var(--shadow-sm);
  transition: transform .15s ease, box-shadow .15s ease;
}
[data-testid="stPageLink"] a:hover { transform:translateY(-1px); box-shadow:var(--shadow-md); }
/* hide the automatic multipage nav — navigation is via explicit in-page links */
[data-testid="stSidebarNav"] { display:none !important; }
[data-testid="stSidebar"] { background:var(--panel); border-right:1px solid var(--line); }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 { font-size:1.05rem; }
.stTabs [data-baseweb="tab"] { font-family:'IBM Plex Mono',monospace; font-size:.78rem;
  letter-spacing:.06em; text-transform:uppercase; }
hr { border-color:var(--line); }

@media (max-width: 900px){
  .tiles{grid-template-columns:repeat(2,1fr);}
  .bm-row{grid-template-columns: 1fr 84px; grid-auto-rows:min-content;}
  .bm-row .strip{grid-column:1 / -1;}
  .hero .verdict{font-size:1.9rem;}
}
@media (prefers-reduced-motion: reduce){
  *{scroll-behavior:auto!important;}
  .hero{animation:none!important;}
  .drv-fill{animation:none!important;}
  .tile,.ev-card,.bm-row,.stButton>button,.stDownloadButton>button,
  [data-testid="stPageLink"] a{transition:none!important;}
}
</style>
"""


def inject() -> None:
    """Inject fonts + component styles. Call once, right after set_page_config."""
    st.markdown(_CSS, unsafe_allow_html=True)
