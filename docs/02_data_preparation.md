# Phase 2 — NHANES Data Preparation

**Author:** Bhavesh Bhargava — MSc Advanced Data Science
**Status:** Implemented (`src/data_prep/`), verified on a synthetic smoke test.
**Cycles used:** NHANES 2013–2014, 2015–2016, 2017–March 2020 (pre-pandemic).

> Every decision below was explicitly approved before implementation. This
> document is the audit trail a viva examiner will ask for.

---

## 1. Approved decisions (the contract)

| Area | Decision | Rationale |
|---|---|---|
| **Biomarker scope** | **Blood-only panel (27 markers)** | Cardiometabolic + CBC + liver + kidney + electrolytes + Vitamin D. **BP/BMI/waist removed entirely** (not blood tests; unavailable from a report at inference). Markers tiered: 21 *actionable* vs 6 *flag-only* (electrolytes/WBC/platelets — reported & escalated, never lifestyle advice). Weight/BP → GP signpost (documented limitation). |
| **Cohort** | Adults only (age ≥ 18) | Paediatric records use percentile-based reference ranges (not adult thresholds) and lack the adult diagnosis/medication questionnaires; their labels would be clinically invalid. |
| **Sample** | Full (non-fasting) sample; HbA1c as glycaemic marker | Keeps the full ~30k sample instead of collapsing to the fasting subsample. |
| **Blood biomarkers — mandatory** (drop if missing) | HbA1c, total cholesterol, HDL | Full-sample blood analytes; label-defining, can't fabricate ground truth. |
| **Blood biomarkers — supplementary** (imputed) | Triglycerides, LDL, fasting glucose | Fasting-subsample blood analytes; used in Layer-1 *only when observed*. |
| **Clinical measurements — optional** (imputed, never dropped) | SBP, DBP, BMI, waist | NOT on a blood report — provided separately (UI/demographics). Used by the rule engine/model *when present*; absence never drops a row. Keeps cardiometabolic signal while honouring the blood-report scope. |
| **Missing values** | Hybrid: drop-if-mandatory-missing, median-impute supplementary + soft predictors | Balances integrity and retention. |
| **Target label** | 3-layer scheme → `Normal / Borderline / Serious` | See §4. Layer 2 is the project's novelty. |
| **RF features** | Biomarkers + demographics **only** | Diagnosis/med vars are label-only → no leakage. |
| **Outliers** | Plausibility→NaN, then winsorise 1st/99th pct | Removes data errors, tames real extremes, deletes no rows. |
| **Scaling** | Unscaled matrix for RF **+** train-fit StandardScaler copy | Trees are scale-invariant; scaled copy kept for distance-based use. |
| **Encoding** | One-hot sex & ethnicity; ordinal education/BMI/age bands | Avoids false ordinality on nominal variables. |
| **Leakage control** | Split first; fit winsor/imputer/scaler on train only | Standard, defensible. |

---

## 2. Source files pulled per cycle

Naming differs by cycle (`_H`, `_I`, `P_`); the loader resolves this automatically.

| Domain | Base file | Canonical variables |
|---|---|---|
| Demographics | `DEMO` | age (`RIDAGEYR`), sex (`RIAGENDR`), ethnicity (`RIDRETH3`), education (`DMDEDUC2`), income ratio (`INDFMPIR`), weights, survey design |
| HbA1c | `GHB` | `LBXGH` |
| Total cholesterol | `TCHOL` | `LBXTC` |
| HDL | `HDL` | `LBDHDD` |
| Triglycerides / LDL | `TRIGLY` | `LBXTR`, `LBDLDL` (fasting) |
| Fasting glucose | `GLU` | `LBXGLU` (fasting) |
| Body measures | `BMX` | BMI, waist, height, weight |
| Blood pressure | `BPX` (13–16) / `BPXO` (17–20) | systolic/diastolic replicates |
| Diabetes Q | `DIQ` | `DIQ010/050/070` — **label-only** |
| BP/cholesterol Q | `BPQ` | `BPQ080/090D/020/040A` — **label-only** |
| Medical conditions | `MCQ` | `MCQ160C/E/F` — **label-only** |
| Medications | `RXQ_RX` | statin/metformin flag — **label-only** |

---

## 3. Two NHANES-specific reconciliations

**Blood-pressure methodology break** (`bp_harmonize.py`). 2013–16 used manual
auscultation (`BPXSY/DI 1–4`); 2017–20 used oscillometric devices
(`BPXOSY/ODI 1–3`). We average replicate readings per person (undetectable
diastolic `0` → missing first), emit a single `sbp_mmhg`/`dbp_mmhg`, and keep a
`bp_method` flag so the methodology effect can be audited rather than silently
absorbed.

**Fasting-subsample problem.** Fasting glucose, triglycerides, and LDL exist
only in the morning fasting subsample (~⅔ missing sample-wide). Making them
*supplementary* (imputed, non-mandatory) preserves the full sample while still
using them in Layer-1 labelling whenever they are genuinely observed.

---

## 4. The three-layer label (methodological core)

Implemented in `labeling.py`. Severity ordinal: `normal(0) < borderline(1) < serious(2)`.

**Layer 1 — rule-based single-biomarker labelling.** Each biomarker is compared
to NICE/WHO/NHS thresholds (`config/clinical_thresholds.yaml`). The record's
Layer-1 label is the *max* severity across biomarkers. This mirrors a GP reading
a report.

**Layer 2 — diagnosis & medication proxy labelling (the novelty).** For records
where no single biomarker breaches a threshold, questionnaire evidence raises the
label: diagnosed/borderline diabetes or glucose-lowering meds → ≥ Borderline;
diagnosed high cholesterol or lipid meds → ≥ Borderline; treated hypertension →
≥ Borderline; established CVD (CHD/MI/stroke) → Serious; statin/metformin use →
≥ Borderline.

**Layer 3 — conflict resolution.** `final = max(Layer1, Layer2)`. Clinical
evidence can only *raise* severity, never lower it. Documented explicitly because
a reviewer will challenge it.

**Leakage guard (critical).** The Layer-2 variables build the label and are then
**dropped from the feature matrix**. The Random Forest therefore never sees the
diagnosis — it must recover the hidden-risk signal from *biomarker patterns
alone*. This is exactly the "combinations no single rule catches" requirement.
The pipeline flags every Layer-2 upgrade (`label_upgraded_by_l2`) so the novelty
can be quantified as its own evaluation slice in Phase 4/9. An automated test
asserts no label-only column ever appears in the feature matrix.

---

## 5. Pipeline order (leakage-safe)

```
merge cycles (SEQN)
  → plausibility bounds → NaN            (stateless)
  → BUILD LABELS                         (on observed values, pre-imputation)
  → drop rows missing mandatory labs
  → stratified train/test split          ← nothing is fit before this line
  → winsorise            (fit on train)
  → impute supplementary (fit on train)
  → feature engineering  (stateless)
  → build UNSCALED matrix (RF) + SCALED copy (fit on train)
  → persist data + artifacts + methods report
```

Labels are built **before** imputation so Layer-1 severity is only ever driven by
genuinely observed biomarkers.

## 6. Feature set produced (27 columns)

Continuous: `age, pir, hba1c_pct, total_chol_mgdl, hdl_mgdl, sbp_mmhg, dbp_mmhg,
bmi, waist_cm, triglycerides_mgdl, ldl_mgdl, fasting_glucose_mgdl` + engineered
`tc_hdl_ratio, tg_hdl_ratio, waist_to_height, tyg_index`.
Ordinal: `educ_code, bmi_category, age_band`.
One-hot: `sex_1/2`, `eth_1/2/3/4/6/7`.
**Excluded from features:** all `diq_*`, `bpq_*`, `mcq_*`, `statin_or_metformin`
(label-only) and `psu/strata/wtmec/cycle` (design/provenance).

---

## 7. Module map (`src/data_prep/`)

| Module | Responsibility (single) |
|---|---|
| `config.py` | Paths, canonical variable map, feature/label column contract, YAML loaders |
| `download.py` | Fetch `.XPT` for all cycles (idempotent, per-cycle naming) |
| `load.py` | Read `.XPT` → pandas, canonical rename, weight harmonisation |
| `bp_harmonize.py` | Reconcile manual vs oscillometric blood pressure |
| `merge.py` | Join components on SEQN per cycle, concat cycles, RXQ flag |
| `labeling.py` | Three-layer target construction + novelty tracking |
| `outliers.py` | Plausibility → NaN; train-fit winsorisation |
| `missing.py` | Hybrid drop + train-fit imputation |
| `features.py` | Clinically-motivated engineered features |
| `encode_scale.py` | One-hot + ordinal encoding; train-fit scaler; column alignment |
| `pipeline.py` | Orchestrates all stages in leakage-safe order; persists artifacts |
| `scripts/run_data_prep.py` | CLI entrypoint (`--download`, `--impute`, …) |

---

## 8. How to run

```bash
pip install -r requirements.txt
python -m scripts.run_data_prep --download     # first run: pulls XPT then builds
python -m scripts.run_data_prep                # subsequent runs use cached XPT
```

Outputs in `data/processed/`: `train.csv`, `test.csv`, `X_train*.csv`,
`X_test*.csv`, `y_*.csv`, and `artifacts/` (fitted imputer, scaler, winsor
bounds, feature names, and a `prep_summary.json` methods report).

---

## 9. Verification & real-run results

`py_compile` passes for all modules. A synthetic smoke test drives
labelling → outliers → missing → features → encoding and asserts: (a) train/test
columns identical, (b) no residual NaNs in features, (c) **no label-only column
present in the feature matrix** (leakage guard), (d) Layer-2 upgrades are counted.
All pass.

**Real NHANES run (all 36 files downloaded as validated XPORT):**

| Stage | Rows |
|---|---|
| Merged (2013-14 + 2015-16 + 2017-20) | 35,706 |
| After adult filter (age ≥ 18) | 21,798 |
| After dropping rows missing a mandatory **blood** analyte | 19,308 |
| Train / Test split (stratified, 80/20) | 15,446 / 3,862 |
| Features | 35 (actionable blood + demographics + engineered; flag-only & Vit-D excluded) |

- **Label distribution (adult cohort, blood-only, weighted-core):** borderline 11,416 · serious 6,855 · normal 3,527.
- **Post-drop class balance (train):** borderline 8,372 · serious 5,245 · normal 1,829 (~11.8%).
- **Removing BP/BMI/waist substantially improved the study** (not just scope): (a) `normal`
  grew 2.9% → **11.8%**, largely resolving the imbalance; (b) `serious` nearly halved
  (12,898 → 6,855) — confirming blood pressure had been *dominating* the "serious" label,
  which had quietly undermined the blood-biomarker thesis; (c) the **Layer-2 novelty slice
  more than doubled (5.1% → 11.9%)** — with BP no longer flagging everyone, far more records
  have normal-looking bloods but a hidden diagnosis/medication signal, which is exactly the
  combinatorial "hidden risk" the Random Forest exists to catch.
- **Documented limitation:** obesity & hypertension are not assessed; the system signposts
  weight/BP to the GP.
- **Blood-report scope fix:** requiring only the 3 mandatory *blood* analytes (not BP/BMI/waist)
  cut the missing-data drop from 4,400 → 2,490 and grew the retained sample ~11%.
- **Layer-2 novelty slice:** 1,164 records (**5.3%**) upgraded by diagnosis/medication
  evidence beyond what biomarker thresholds alone flag — the RF's target value-add.
- **BP harmonisation:** 9,397 manual + 6,706 oscillometric readings reconciled.
- The adult filter cut the missing-mandatory drop rate from 44% → 20%, confirming
  most excluded rows were minors lacking adult labs (a validity gain, not just N loss).

Download robustness: the CDC migrated its URL scheme to
`…/Public/{startyear}/DataFiles/{FILE}.XPT` and serves an HTTP-200 HTML page for
missing files. The downloader now validates the XPORT magic header on every file
and rejects soft-404 HTML, so silent corruption cannot recur.

---

## 10. Known limitations to note in the dissertation

1. **Class balance.** With single-serious-biomarker → Serious, the population
   skews toward Borderline/Serious (high cardiometabolic prevalence in NHANES).
   Phase 4 will apply class weighting / resampling and report macro-F1 & PR-AUC,
   not accuracy.
2. **US population, UK thresholds.** The RF learns risk *patterns* from US data;
   clinical *labels/cut-offs* are UK-guideline-anchored. The RF is never used
   diagnostically — the rule engine owns thresholds at inference.
3. **Survey weights not used in training.** Weights are retained as columns for
   transparency; weighted ML is out of scope for the RF and documented as such.
4. **BP methodology.** Harmonised by averaging with a method flag rather than an
   unvalidated numeric correction.

**Phase 2 exit criteria met — ready for Phase 3 (Clinical Rule Engine) on approval.**
