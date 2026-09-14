# Phase 3: Clinical rule engine

**Author:** Bhavesh Bhargava, MSc Advanced Data Science
**Code:** `src/app/domain/` and `src/app/rules/`, 19 unit tests
**Guidelines:** NICE, NHS, WHO, NCEP ATP III and standard clinical reference ranges (each band names its source in the YAML)
**Panel:** 27 blood biomarkers: 21 actionable (used for the label and recommendations) and 6 flag-only.
Blood pressure, BMI and waist aren't included.

The rule engine is the deterministic part of the pipeline. There are no clinical
numbers in the code: every threshold, status, interpretation and citation is in
`config/clinical_rules.yaml`.

---

## 1. Design decisions

| Decision | Choice | Reason |
|---|---|---|
| **Severity model** | Each biomarker gets a `status` (normal, low, high, borderline, severe), a separate `severity` used for the overall result (normal, borderline, serious), and an `urgent` flag | `low`/`high` say which way a result is off, and `borderline`/`severe` say by how much. Keeping status and severity separate means a result can be clinically high but only count as borderline overall, e.g. potassium at 5.4 mmol/L. |
| **One config file** | `clinical_rules.yaml` is used by the engine and by the Layer 1 training labels | The app and the training labels can't disagree. Checked by re-running data preparation. |
| **Urgent results** | Any band marked urgent (by default every `severe` band) sets `summary.urgent_referral = true` | Serious results are always escalated. Can be changed per band with `urgent:`. |
| **Adding biomarkers** | A new YAML block, with no Python changes | |

---

## 2. Statuses

| Status | Meaning | Example |
|---|---|---|
| `normal` | inside the reference range | HbA1c 5.2% |
| `borderline` | slightly outside the range, in either direction | HbA1c 6.0% (prediabetes) |
| `low` | below the range | HDL 45 mg/dL in a woman; fasting glucose 60 mg/dL |
| `high` | above the range | HbA1c 7.1%; potassium 5.4 mmol/L |
| `severe` | far outside the range, needs prompt attention | HbA1c 10.5%; potassium 6.0 mmol/L or more |

The severity and urgent flag for each status are set **per band in the config**,
so the clinical meaning comes from the data and can be checked there.

---

## 3. How the config works (`clinical_rules.yaml`)

Each biomarker has an ordered list of `bands`. A value goes in the first band
where `min <= value < max`; leave out `min` or `max` for an open end. Either can
be a number **or** a per-sex map `{male, female}`, resolved when the value is
evaluated.

```yaml
hdl_mgdl:
  name: HDL cholesterol
  unit: " mg/dL"
  reference_range: {low: {male: 40, female: 50}, high: 200}
  bands:
    - {status: severe, max: 30, severity: serious, direction: low, urgent: true,
       guideline: {source: NICE, code: CG181},
       interpretation: "{name} {value}{unit} is very low (<30); a strong ... risk factor."}
    - {status: low, min: 30, max: {male: 40, female: 50}, severity: borderline, direction: low, ...}
    - {status: normal, min: {male: 40, female: 50}, severity: normal, direction: in_range, ...}
```

**Checked when loading.** Numeric bands must be in ascending order with no gaps
and cover every possible value, and unknown statuses or severities are rejected.
A mistake in the YAML stops the app at startup instead of giving wrong results
later.

**The 27 biomarkers.** Every result includes `category` (always `blood`), `tier`
and `label_role`:
- **Panels:** cardiometabolic (HbA1c, glucose, lipids), full blood count (Hb, Hct,
  RBC, MCV, MCH, MCHC, RDW, WBC, platelets), liver (ALT, AST, ALP, albumin,
  bilirubin), kidney (creatinine, BUN), electrolytes (Na, K, Cl, Ca) and vitamin D.
- **Not included:** blood pressure, BMI and waist, because they aren't blood tests
  and wouldn't be on a blood report. Weight and blood pressure are left to the GP.
- **`tier`** is `actionable` for 21 markers, which drive the recommendations and
  the label, and `flag_only` for 6 (electrolytes, WBC, platelets), which are
  reported and escalated but never turned into lifestyle advice. Those go to a
  "discuss with your clinician" signpost instead.
- **`label_role`** is `core` for the cardiometabolic markers, `secondary` for the
  markers listed in `meta.label_secondary_markers` (blood count, liver, kidney,
  vitamin D), and `none` for flag-only markers. See §3b.

### 3b. Overall severity

With this many actionable markers, a simple "any borderline marker makes the
record borderline" rule left almost no normal records, because a single slightly
raised RDW or a low vitamin D was enough. The rule used instead (set in `meta`, so
it can be tuned) is:

- any actionable **serious** marker → **serious**
- a **core** marker at borderline → **borderline**
- **secondary** markers → borderline only when at least
  `secondary_borderline_min` (2) of them are borderline
- otherwise **normal**

`RuleEngineResult.overall_severity` and the Layer 1 training label both use this
same rule, so they can't drift apart. Flag-only markers don't affect the overall
severity; they can only set `urgent_referral`.

---

## 4. Structure

```
config/clinical_rules.yaml               ← all the clinical content
        │  loaded and validated by
        ▼
src/app/rules/loader.py                  → RuleSet
        │  passed into
        ▼
src/app/domain/services/rule_engine.py   (no file access, no hard-coded values)
        │  uses
        ▼
src/app/domain/models.py + enums.py      (plain data classes and enums)
```

- The engine only depends on the `RuleSet` it's given, so a different config
  file means a different clinical policy with no code change.
- The engine only classifies. Fusion, retrieval and explanations happen
  elsewhere.
- `domain/` has no external imports.

---

## 5. JSON output

`engine.evaluate(readings, context).to_dict()`. This is the output of
`scripts/run_rule_engine.py` for its sample panel, with one of the eleven
biomarkers shown:

```json
{
  "ruleset_version": "1.1",
  "context": {"age": 54, "sex": "male"},
  "biomarkers": [
    {
      "code": "hba1c_pct", "name": "HbA1c", "category": "blood",
      "tier": "actionable", "label_role": "core",
      "value": 6.8, "unit": "%",
      "status": "high", "direction": "high", "severity": "serious", "urgent": false,
      "reference_range": {"low": 4.0, "high": 5.6},
      "interpretation": "HbA1c 6.8% is in the diabetes range (>=6.5%).",
      "guideline": {"source": "NICE", "code": "NG28"}
    }
  ],
  "summary": {
    "overall_severity": "serious",
    "urgent_referral": false,
    "flagged": ["hba1c_pct", "fasting_glucose_mgdl", "total_chol_mgdl", "hdl_mgdl",
                "ldl_mgdl", "triglycerides_mgdl", "alt", "hemoglobin", "vitamin_d",
                "potassium"],
    "recommendation_targets": ["hba1c_pct", "fasting_glucose_mgdl", "total_chol_mgdl",
                               "hdl_mgdl", "ldl_mgdl", "triglycerides_mgdl", "alt",
                               "hemoglobin", "vitamin_d"],
    "clinician_signpost": ["potassium"],
    "status_counts": {"high": 3, "borderline": 5, "low": 2, "normal": 1},
    "unknown_codes": []
  }
}
```

`overall_severity` (worked out as in §3b) and `urgent_referral` are what the
fusion step uses.

---

## 6. Modules

| File | What it does |
|---|---|
| `domain/enums.py` | `Status`, `Severity`, `Direction`, severity ranking and `max_severity` |
| `domain/models.py` | `Band`, `BiomarkerRule`, `RuleSet`, `BiomarkerResult`, `RuleEngineResult`, resolving per-sex thresholds |
| `domain/services/rule_engine.py` | The engine: `classify`, `evaluate`, `layer1_severity` |
| `rules/loader.py` | Reads and validates `clinical_rules.yaml` into a `RuleSet` |
| `scripts/run_rule_engine.py` | Command-line demo that prints the JSON output |
| `tests/test_rule_engine.py` | 19 unit tests |

---

## 7. Using the engine for the training labels

Layer 1 labelling in data preparation calls `engine.layer1_severity(...)`
instead of its own threshold code. Re-running data preparation when this change
was made gave:

| Label | Previous labelling code | Using the engine |
|---|---|---|
| serious | 12,752 | 12,753 |
| borderline | 7,016 | 7,577 |
| normal | 2,030 | 1,468 |

These counts are from before blood pressure, BMI and waist were removed, so they
don't match the current dataset. `serious` stayed the same, and about 560 records
moved from normal to borderline because the engine includes mild flags the old
code didn't have (low blood pressure, low glucose, and the increased-risk waist
band). The rules and the labels have been a single definition since then.

---

## 8. Verification

- **19 unit tests pass.** They cover loading and validating the config, each
  status band, band boundaries (6.5 is diabetes, 6.49 is prediabetes), per-sex HDL
  (45 mg/dL is normal for a man and low for a woman), low glucose, missing and NaN
  values returning `None`, the summary from `evaluate`, unknown codes, JSON
  output, the panel size (27 markers, 21 actionable), blood pressure, BMI and
  waist being absent, flag-only markers going to the signpost, an anaemia marker
  being actionable, and the three overall-severity cases from §3b.
- **Demo:** `run_rule_engine.py` prints valid JSON. Its sample panel is `serious`
  overall, with potassium (5.4 mmol/L, flag-only) sent to the clinician signpost
  and no urgent referral.
