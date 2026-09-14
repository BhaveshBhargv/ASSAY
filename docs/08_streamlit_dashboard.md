# Phase 8: Streamlit dashboard

**Author:** Bhavesh Bhargava, MSc Advanced Data Science
**Code:** `streamlit_app/`, checked in the browser
**Run:** `streamlit run streamlit_app/dashboard.py`

The dashboard connects the sidebar and entry form to the engine and shows the
result: the assessment, the evidence and the recommendations. It calls the
engine directly rather than going through the API, so both can use the same
engine without depending on each other.

---

## 1. Features and where they're implemented

| Feature | Implementation |
|---|---|
| Upload a blood report | `app.ingestion.report_parser` (re-exported by `services/parsing.py`). Reads CSV and JSON, and PDFs on a best-effort basis, including the patient's age and sex from the report header. The values fill the form so they can be checked and corrected. |
| Risk summary | `ui/components.hero` (verdict and severity meter) and `stat_tiles` |
| Severity table | `ui/components.severity_table`: every biomarker with a reference range bar |
| Interactive charts | `ui/charts.py` (Plotly): distance outside the reference range, and the model's class probabilities |
| Retrieved guideline passages | `ui/components.evidence_cards`, labelled `[E1]…[En]` |
| Recommendations | `services/pipeline.generate` and `ui/components.recommendations` |
| PDF report download | `services/report_pdf.py` (fpdf2), which works with or without the LLM output |

---

## 2. Visual design

- **Look.** Dark slate text on a light grey-blue background, with one teal
  accent colour. Severity uses green, amber and red.
- **Fonts.** `Archivo` for headings, `IBM Plex Sans` for body text and `IBM Plex
  Mono` for values, so biomarker results read like a lab printout.
- **Reference range bars.** Each biomarker is shown as a horizontal bar with the
  normal range shaded and the patient's value marked in its severity colour. The
  same idea is used for the severity meter at the top.
- **Layout.** Most of the visual weight is on the bars and the summary, and the
  rest is kept plain. On narrow screens the tiles and rows wrap without
  scrolling sideways, and animations are turned off for users who prefer reduced
  motion.
- The CSS is in `ui/theme.py`, and the same colours are used in the Plotly charts.

---

## 3. Structure

```
streamlit_app/
  dashboard.py            entry point (not called app.py, so it doesn't shadow the `app` package)
  pages/guidelines.py     page listing all retrieved passages
  ui/
    theme.py              colours and CSS
    metadata.py           biomarker names, units and panels, re-exported from app.ingestion
    inputs.py             sidebar controls and the biomarker form (owns the widget state)
    components.py         HTML snippets (no Streamlit calls)
    charts.py             Plotly charts
  services/
    parsing.py            re-exports the shared report parser from app.ingestion
    pipeline.py           cached engine wrapper: assess() and generate()
    report_pdf.py         PDF export with fpdf2
  sample_data/samples.py  three sample patients
```

- **Separation.** `services` talk to the engine and `ui` only displays things.
  `components.py` just returns HTML strings, which keeps it easy to follow.
- **Two steps.** `assess()` runs the rules, model, fusion and retrieval, and is
  useful on its own. `generate()` runs the LLM **in a background thread**. The
  assessment, charts and evidence appear straight away, and an `st.fragment` that
  refreshes every two seconds shows the recommendations once the thread has
  finished. If the provider fails or can't be reached, a message is shown instead,
  so the dashboard still works with **no model running**.
- **Shared parser.** Report parsing, the biomarker catalog and unit conversion
  live in `app.ingestion`, so the dashboard and the API read a report the same
  way. `services/parsing.py` and `ui/metadata.py` are thin wrappers around it.
- **Caching.** The rule engine, FAISS retriever and Random Forest are loaded once
  with `st.cache_resource`.
- **Model input.** `RiskAdapter` builds the model's features with the same
  transformations used in data preparation, and falls back to rules only if too
  few biomarkers are provided.

---

## 4. Safety in the UI

- The summary always says **"not a diagnosis"**, and the disclaimer is fixed text
  added by the app, not written by the model.
- **Flag-only** markers (electrolytes, WBC, platelets) appear in a separate
  "discuss with your clinician" box and are never turned into lifestyle advice.
- **Urgent** results show a warning asking the user to contact their GP.
- Under the recommendations, the dashboard shows the **groundedness score** and
  how many uncited items the checks removed.

---

## 4a. Report parsing (`app.ingestion`)

The PDF reader isn't tied to one lab's format. It handles the two layouts lab
reports commonly use, in a single pass:

| Layout | Example |
|---|---|
| **A: one row per test** | `Total Cholesterol 160 mg/dL 0 - 200` |
| **B: stacked** | `Serum Triglycerides` / `Method: Enzymatic` / `167.9 H* mg/dL <150` |

Four checks stop a best-effort read turning into a wrong read:

1. **Units must match.** A value is only accepted if its unit is valid for that
   biomarker (`units.CONVERSIONS`), and it's then converted to the app's unit. So
   a vitamin B12 result of `199 pg/mL` can't be read as fasting glucose, and
   vitamin D in ng/mL is converted to nmol/L.
2. **Abnormal flags are removed.** Labs print results like `10.7 L* g/dL`. Without
   removing the flag the unit wouldn't match and the value would be lost, and
   those are exactly the abnormal results that matter most.
3. **Reference ranges aren't results.** Numbers after a comparison sign or dash
   (`< 100`, `0 - 200`) are never used as values, and a value on its own line
   can't be matched to a test name if another test name came in between.
4. **Calculated rows are ignored.** `Estimated Average Glucose`, `Non-HDL`,
   `VLDL`, any `X/Y Ratio`, `eGFR`, `MPV`/`PDW` and differential percentages share
   names with tracked biomarkers, so `_NOISE_LABEL` excludes them.

When a label matches more than one biomarker, the higher-priority and then longer
match wins. That's why `Glycosylated Hemoglobin (HbA1c)` maps to `hba1c_pct`
rather than `hemoglobin`. Before this was fixed, HbA1c, the most important
marker in the panel, was being lost.

**Age and sex** come from the report header, in the forms labs actually print
(`Female 61 yrs`, `Gender: Female Age: 61 Yrs`, `Age/Gender : 60Y 0M 0D /Male`,
`DOB/Age/Gender : 61 Y/Female`). A line that has both age and sex is preferred,
so reference text that mentions ages (`Age > 19 years`,
`Non diabetic adults >=18 years`) isn't mistaken for the patient. Ages outside
the adult range are ignored, and the form asks for them instead.

All of this runs locally, with no network requests or outside services.
Extraction is still best-effort, so the UI asks for every value to be checked
before analysis.

---

## 5. Verification

Checked in the browser with the trained model and index:
- The metabolic sample patient gives **"Higher risk pattern"** (serious), with 15
  biomarker rows and their range bars, 4 summary tiles, 2 charts and 6 evidence
  cards. With no LLM running the recommendations show a message, and the PDF
  export still produces a valid document.
- The fonts, colours, range bar positions and severity meter position render as
  intended.
- At 375 px wide there's no sideways scrolling.
- The PDF builds both with and without recommendations.

**Parsing was tested on two real lab reports** in different formats:

| Report | Layout | Values extracted | Age and sex |
|---|---|---|---|
| A (26 pages) | one-row summary tables and stacked detail pages | **24 of 24**, checked by hand against the report, none wrong or extra | 61, female |
| B (38 pages) | stacked, with labels over two lines (`(ALT/SGPT) :`) | 27 | 60, male |

Report A shows why these checks matter. Before the layout and label fixes, only
17 values were read, and HbA1c (6.8%, in the diabetes range) and fasting glucose
(152 mg/dL) were missing. The panel was then graded **borderline**; with all the
values it's graded **serious**. `tests/test_parser.py` has 18 tests covering
each of these cases.
