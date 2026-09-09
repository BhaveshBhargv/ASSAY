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
| Upload blood report | `app.ingestion.report_parser` (re-exported by `services/parsing.py`) — CSV / JSON (structured) + **best-effort PDF** extraction, including the patient's age and sex from the report header, with a review-and-correct step (values populate the form) |
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
    parsing.py            thin re-export of the shared engine parser (app.ingestion)
    pipeline.py           cached engine wrapper: assess() (always) + generate() (LLM)
    report_pdf.py         fpdf2 PDF export
  sample_data/samples.py  three built-in demo patients
```

- **Separation:** `services` talk to the engine; `ui` only renders. `components.py`
  is pure (returns HTML strings), so the visual layer is trivial to reason about.
- **Two-step pipeline:** `assess()` (rules + RF + fusion + RAG) always runs and is
  useful on its own; `generate()` (the LLM) runs **in a background thread** so its
  latency never blocks the page. The assessment, charts and evidence render
  immediately; a self-refreshing `st.fragment` swaps the recommendations in when
  the thread finishes. Provider/connection failures are shown as a clean empty
  state, never a crash — the dashboard is fully usable with **no model running**.
- **Shared parser:** report parsing, the biomarker catalog and unit conversion live
  in `app.ingestion`, so the dashboard and the FastAPI backend read a report
  identically. `services/parsing.py` and `ui/metadata.py` are thin adapters over it.
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

## 4a. Report parsing (`app.ingestion`)

PDF extraction is layout-agnostic rather than tied to one lab's house style. It
reads the two layouts real reports use, in one pass:

| Layout | Example |
|---|---|
| **A — one row per test** | `Total Cholesterol 160 mg/dL 0 - 200` |
| **B — stacked** | `Serum Triglycerides` / `Method: Enzymatic` / `167.9 H* mg/dL <150` |

Four defences keep a best-effort read from becoming a wrong read:

1. **Unit anchoring.** A value is accepted only if its printed unit is valid for
   that analyte (`units.CONVERSIONS`), then converted to the app's canonical unit —
   so `199 pg/mL` of vitamin B12 can never be read as fasting glucose, and
   vitamin D in ng/mL is converted to nmol/L.
2. **Abnormality flags are stripped.** Labs print `10.7 L* g/dL`; without handling
   the flag the unit match fails and the value is dropped — losing precisely the
   *abnormal* results.
3. **Reference bounds are not readings.** Numbers introduced by a comparator or
   dash (`< 100`, `0 - 200`) are never candidates, and a stacked value may not be
   paired with a label that a different test has superseded.
4. **Derived rows are blocked.** `Estimated Average Glucose`, `Non-HDL`, `VLDL`,
   any `X/Y Ratio`, `eGFR`, `MPV`/`PDW` and differential percentages borrow a
   tracked analyte's name and are excluded by `_NOISE_LABEL`.

Label matching resolves by **priority then longest term**, which is what keeps
`Glycosylated Hemoglobin (HbA1c)` mapping to `hba1c_pct` rather than to
`hemoglobin` — a collision that silently cost the most important marker in the
panel before it was fixed.

**Demographics** are read from the report header, accepting the shapes labs
actually print (`Female 61 yrs`, `Gender: Female Age: 61 Yrs`,
`Age/Gender : 60Y 0M 0D /Male`, `DOB/Age/Gender : 61 Y/Female`). A line carrying
*both* age and sex is trusted first, so guideline prose that also mentions ages
(`Age > 19 years`, `Non diabetic adults >=18 years`) can never be mistaken for the
patient. Ages outside the adult range are ignored rather than guessed, leaving the
form to ask.

Everything here runs **locally** — no network call, no third-party service.
Extraction remains best-effort by design, and the UI insists every value is
reviewed before analysis.

---

## 5. Verification

Verified live in-browser with the built models:
- Metabolic sample → verdict **"Higher risk pattern"** (serious); 15 biomarker rows
  each with a positioned range strip; 4 risk tiles; 2 Plotly charts; 6 cited
  evidence cards; recommendations empty-state when the LLM is off; PDF export
  produces a valid document.
- Design tokens confirmed applied (Archivo/Plex fonts, teal + severity colours,
  strip marker/band positions, severity-meter position).
- Responsive at 375 px with no horizontal overflow.
- PDF builds for both the assessment-only and full-recommendation paths.

**Parsing verified against two real lab PDFs** of different house styles:

| Report | Layout | Values extracted | Demographics |
|---|---|---|---|
| A (26 pp) | single-row summary tables + stacked detail pages | **24 / 24**, hand-checked against the source; 0 wrong, 0 spurious | age 61, female |
| B (38 pp) | stacked, two-line labels (`(ALT/SGPT) :`) | 27 | age 60, male |

Report A also demonstrates why the parsing defences matter clinically: before the
layout and label-priority fixes it yielded 17 values, missing HbA1c (6.8%, the
diabetes range) and fasting glucose (152 mg/dL). The resulting assessment graded
the panel **borderline**; with the full read it grades **serious**. Regression
tests in `tests/test_parser.py` (18) pin each defence.

**Phase 8 exit criteria met.** Remaining: Phase 11 (dissertation write-up).
