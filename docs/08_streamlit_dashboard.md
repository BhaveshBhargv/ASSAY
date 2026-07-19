# Phase 8 — Streamlit Dashboard ("Assay")

**Author:** Bhavesh Bhargava — MSc Advanced Data Science
**Status:** Implemented (`streamlit_app/`), verified in-browser.
**Run:** `streamlit run streamlit_app/dashboard.py`

> **Role in the project.** The presentation layer. It wires the sidebar/form to the
> Phase 3-6 engine and renders the result — assessment, evidence, and grounded
> recommendations — as a clean clinical dashboard. It calls the engine directly
> (Clean Architecture: the UI depends on the use-case layer), so a future Phase 7
> FastAPI service can wrap the same engine without changing this layer.

---

## 1. Requested features → where they live

| Feature | Implementation |
|---|---|
| Upload blood report | `services/parsing.py` — CSV / JSON (structured) + **best-effort PDF** extraction, with a review-and-correct step (values populate the form) |
| Risk summary | `ui/components.hero` (verdict + severity meter) + `stat_tiles` |
| Severity table | `ui/components.severity_table` — every analyte with its reference-range strip |
| Interactive charts | `ui/charts.py` (Plotly) — deviation-outside-range + RF class probabilities |
| Retrieved guideline passages | `ui/components.evidence_cards` — cited `[E1]…[En]` evidence |
| Generated recommendations | `services/pipeline.generate` + `ui/components.recommendations` |
| Download PDF report | `services/report_pdf.py` (fpdf2) — works with or without the LLM step |

---

## 2. Design system (frontend-design)

- **Identity — "Assay," a clinical instrument dashboard.** Deep slate ink on cool
  paper, a single **clinical teal** primary (deliberately not medical-blue or the
  AI-default cream/terracotta). Severity uses the domain's own three-colour
  language (green / amber / red).
- **Type:** `Archivo` (display) + `IBM Plex Sans` (body) + `IBM Plex Mono` (all
  data/values) — the Plex family's instrument heritage makes biomarker values read
  like a lab readout.
- **Signature element:** the **biomarker reference-range strip** — each analyte as a
  horizontal track with its reference band and the patient's value marker, coloured
  by severity. Reused in the hero severity meter and the severity table.
- **Restraint:** boldness spent on the strips + hero; chrome kept quiet. Responsive
  to mobile (tiles and strip rows reflow, no horizontal overflow); reduced-motion
  respected.
- Styling is injected CSS (`ui/theme.py`); the theme palette is shared with the
  Plotly charts so the whole app stays on-brand.

---

## 3. Architecture (modular)

```
streamlit_app/
  dashboard.py            entry — orchestration only (named to not shadow the `app` package)
  ui/
    theme.py              palette tokens + injected CSS (single source of visual truth)
    metadata.py           biomarker panels + PDF synonyms (names/units/ranges from the ruleset)
    inputs.py             sidebar controls + blood-panel form (owns widget state)
    components.py          presentational HTML fragments (no Streamlit calls)
    charts.py             themed Plotly figures
  services/
    parsing.py            CSV / JSON / best-effort PDF -> {code: value}
    pipeline.py           cached engine wrapper: assess() (always) + generate() (LLM, optional)
    report_pdf.py         fpdf2 PDF export
  sample_data/samples.py  three built-in demo patients
```

- **Separation:** `services` talk to the engine; `ui` only renders. `components.py`
  is pure (returns HTML strings), so the visual layer is trivial to reason about.
- **Two-step pipeline:** `assess()` (rules + RF + fusion + RAG) always runs and is
  useful on its own; `generate()` (the LLM) is optional and off by default, so the
  dashboard works with **no model running**. Provider/connection failures are shown
  as a clean empty state, never a crash.
- **Caching:** the rule engine, FAISS retriever, and RF load once via
  `st.cache_resource`.
- **RF at inference:** `RiskAdapter` rebuilds the model's feature vector with the
  same stateless Phase-2 transforms; it degrades to rules-only if the panel is too
  sparse.

---

## 4. Safety carried through the UI

- The hero always reads **"not a diagnosis"**; the fixed disclaimer is rendered by
  the app (system-owned), not the model.
- **Flag-only** markers (electrolytes, WBC, platelets) route to a **clinician
  signpost** callout, never to lifestyle advice.
- **Urgent** results surface a prompt-attention banner.
- The recommendations footer reports the **groundedness score** and how many
  ungrounded items the verifier removed — the Phase-6 guarantees made visible.

---

## 5. Verification

Verified live in-browser (localhost:8502) with the built models:
- Metabolic sample → verdict **"Higher risk pattern"** (serious); 15 biomarker rows
  each with a positioned range strip; 4 risk tiles; 2 Plotly charts; 6 cited
  evidence cards; recommendations empty-state when the LLM is off; PDF export
  produces a valid document.
- Design tokens confirmed applied (Archivo/Plex fonts, teal + severity colours,
  strip marker/band positions, severity-meter position).
- Best-effort parsing confirmed on CSV/JSON (synonym mapping) and PDF text.
- Responsive at 375 px with no horizontal overflow.
- PDF builds for both the assessment-only and full-recommendation paths.

**Phase 8 exit criteria met.** Remaining: Phase 7 (FastAPI backend — can wrap the
same engine), Phase 9 (evaluation & explainability), 10 (deployment), 11 (writeup).
