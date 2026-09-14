# Phase 2: NHANES data preparation

**Author:** Bhavesh Bhargava, MSc Advanced Data Science
**Code:** `src/data_prep/`
**Data:** NHANES 2013–2014, 2015–2016 and 2017–March 2020 (the pre-pandemic release)

How the NHANES survey data was turned into the training dataset, and the
reasoning behind each choice.

---

## 1. Decisions

| Area | Decision | Reason |
|---|---|---|
| **Biomarkers** | **Blood tests only, 27 markers** | Cardiometabolic, full blood count, liver, kidney, electrolytes and vitamin D. Blood pressure, BMI and waist were removed completely, since they aren't blood tests and wouldn't be on a blood report. 21 markers are *actionable* and 6 are *flag-only* (electrolytes, WBC, platelets): flag-only markers are reported and escalated but never turned into lifestyle advice. Weight and blood pressure are left to the GP, which is a stated limitation. |
| **Cohort** | Adults only (18 and over) | Children's results are judged against age-based ranges rather than adult thresholds, and children don't answer the adult diagnosis and medication questions, so their labels wouldn't be valid. |
| **Sample** | Full (non-fasting) sample, with HbA1c as the glucose marker | Keeps the whole sample (35,706 people before the adult filter) instead of shrinking to the fasting subsample. |
| **Mandatory biomarkers** (rows dropped if missing) | HbA1c, total cholesterol, HDL | Measured across the full sample and needed to build the label, which can't be imputed. |
| **Supplementary biomarkers** (imputed) | The other model biomarkers: triglycerides, LDL and fasting glucose, plus the blood count, liver and kidney markers | The fasting tests only exist for the fasting subsample and the others have gaps. Layer 1 only uses them where they were actually measured, because labels are built before imputation. |
| **Clinical measurements** | SBP, DBP, BMI and waist are **not loaded** | Not blood tests and not on a blood report, so they were removed from the file list rather than imputed. See §3 and §10. |
| **Missing values** | Drop rows missing a mandatory marker; median-impute the supplementary markers plus income ratio and education | Keeps as many rows as possible without inventing labels. |
| **Target label** | Three layers → `normal / borderline / serious` | See §4. |
| **Model features** | Blood markers plus **age and sex only** | Diagnosis and medication answers are only used for the label, so they can't leak into the features. Ethnicity, income and education were dropped because they aren't on a blood report and lifestyle can't change them. See §6. |
| **Outliers** | Implausible values set to NaN, then winsorised at the 1st/99th percentile | Removes data errors and limits real extremes without deleting rows. |
| **Scaling** | Unscaled matrix for the Random Forest, plus a copy scaled with a StandardScaler fitted on the training set | Trees don't need scaling; the scaled copy is kept for distance-based methods. |
| **Encoding** | Sex one-hot encoded, age band as an ordinal | Avoids implying an order between the sex codes. |
| **Leakage** | Split first, then fit winsorising, imputation and scaling on the training set only | Standard practice. |

---

## 2. Files downloaded for each cycle

File names differ between cycles (`_H`, `_I`, `P_`); the loader works this out
from the config.

| Area | Base file | Variables |
|---|---|---|
| Demographics | `DEMO` | age (`RIDAGEYR`), sex (`RIAGENDR`), ethnicity (`RIDRETH3`), education (`DMDEDUC2`), income ratio (`INDFMPIR`), weights, survey design |
| HbA1c | `GHB` | `LBXGH` |
| Total cholesterol | `TCHOL` | `LBXTC` |
| HDL | `HDL` | `LBDHDD` |
| Triglycerides / LDL | `TRIGLY` | `LBXTR`, `LBDLDL` (fasting) |
| Fasting glucose | `GLU` | `LBXGLU` (fasting) |
| Full blood count | `CBC` | `LBXHGB`, `LBXHCT`, `LBXRBCSI`, `LBXMCVSI`, `LBXMCHSI`, `LBXMC`, `LBXRDW`, `LBXWBCSI`, `LBXPLTSI` |
| Biochemistry (liver, kidney, electrolytes) | `BIOPRO` | `LBXSATSI`, `LBXSASSI`, `LBXSAPSI`, `LBXSAL`, `LBXSTB`, `LBXSCR`, `LBXSBU`, `LBXSNASI`, `LBXSKSI`, `LBXSCLSI`, `LBXSCA` |
| Vitamin D | `VID` | `LBXVIDMS`, 2013–16 only (not in the `P_` release, so skipped there) |
| Diabetes questionnaire | `DIQ` | `DIQ010/050/070`, **label only** |
| BP/cholesterol questionnaire | `BPQ` | `BPQ080/090D/020/040A`, **label only** |
| Medical conditions | `MCQ` | `MCQ160C/E/F`, **label only** |
| Medications | `RXQ_RX` | statin/metformin flag, **label only** |

---

## 3. NHANES quirks

**Body measures and blood pressure aren't downloaded.** An early version loaded
`BMX` and `BPX`/`BPXO`, which meant reconciling manual blood pressure readings in
2013–16 (`BPXSY/DI 1–4`) with the automatic readings in 2017–20
(`BPXOSY/ODI 1–3`). Once the project was limited to blood test reports, these
could never be inputs, so they were removed from `config/nhanes_files.yaml`
together with the reconciliation code. Weight and blood pressure are left to the
GP. This is a limitation, but it also improved the labels (see §9).

**Cycle names.** Files end in `_H` (2013–14) and `_I` (2015–16) but start with
`P_` in the 2017–20 release. `Cycle.filename()` builds the name from the config,
so a new cycle doesn't need code changes.

**Survey weight names.** The examination weight is `WTMEC2YR` in the two-year
cycles and `WTMECPRP` in the pre-pandemic release. `load_component` renames both
to `wtmec`.

**Fasting subsample.** Fasting glucose, triglycerides and LDL are only measured
in the morning fasting subsample, so about two thirds of people don't have them.
Treating them as supplementary (imputed, not mandatory) keeps the full sample,
and Layer 1 still uses them whenever they were measured.

---

## 4. The three-layer label

Built in `labeling.py`. Severity order: `normal (0) < borderline (1) < serious (2)`.

**Layer 1: blood results.** Each record is labelled by the same rule engine the
app uses (`config/clinical_rules.yaml`, see Phase 3), with the same rule for the
overall result: any serious actionable marker makes the record serious, and one
borderline core marker or at least two borderline secondary markers make it
borderline. Flag-only markers are ignored. This is roughly what a GP would read
off the report.

**Layer 2: diagnoses and medication.** Questionnaire answers can raise the label
for people whose bloods don't show the problem:
- diagnosed or borderline diabetes, or taking insulin or diabetes tablets → at least borderline
- diagnosed high cholesterol, or taking cholesterol medication → at least borderline
- high blood pressure that's being treated → at least borderline
- coronary heart disease, heart attack or stroke → serious
- a statin or metformin in the prescription data → at least borderline

These rules are written in `labeling.py`. The `diagnosis_proxies` section of
`config/clinical_thresholds.yaml` describes the same rules but isn't read by the
code.

**Layer 3: combining them.** `final = max(Layer 1, Layer 2)`, so questionnaire
answers can raise the severity but never lower it.

**Keeping the label out of the features.** The Layer 2 columns are used to build
the label and nothing else. The feature matrix is built from a fixed list of
columns (`encode_scale.py`) that doesn't include any of them, so the Random
Forest never sees a diagnosis. The idea was that it would have to find this
hidden risk from the blood results alone. Every record that Layer 2 raised is
marked (`label_upgraded_by_l2`) so it can be evaluated separately; Phase 4 §5
reports how the model did on these records.

---

## 5. Pipeline order

```
merge cycles (SEQN)
  → implausible values → NaN             (no fitting)
  → build labels                         (on measured values, before imputation)
  → drop rows missing mandatory markers
  → stratified train/test split          ← nothing is fitted before this point
  → winsorise            (fitted on train)
  → impute supplementary (fitted on train)
  → engineered features  (no fitting)
  → build the unscaled matrix (RF) and a scaled copy (fitted on train)
  → save the data, fitted objects and summary
```

Labels are built **before** imputation, so Layer 1 only ever uses measured
values.

## 6. Features (27 columns)

The model only uses clinical inputs: biomarkers, age and sex. A blood report
doesn't include social or demographic details and lifestyle can't change them,
so ethnicity, income ratio and education aren't features.

**Continuous (24):** `age`, the 20 actionable blood markers
(`hba1c_pct, fasting_glucose_mgdl, total_chol_mgdl, ldl_mgdl, hdl_mgdl,
triglycerides_mgdl, hemoglobin, hematocrit, rbc, mcv, mch, mchc, rdw, alt, ast,
alp, albumin, total_bilirubin, creatinine, bun`) and the engineered
`tc_hdl_ratio, tg_hdl_ratio, tyg_index`.
**Ordinal (1):** `age_band`.
**One-hot (2):** `sex_1`, `sex_2`.

**Not used as features:**

| Columns | Reason |
|---|---|
| `diq_*`, `bpq_*`, `mcq_*`, `statin_or_metformin` | label only, to avoid leakage |
| `psu`, `strata`, `wtmec`, `cycle` | survey design and bookkeeping |
| `vitamin_d` | missing for a whole cycle |
| `wbc`, `platelets`, `sodium`, `potassium`, `chloride`, `calcium` | flag-only markers: reported and escalated by the rule engine, never modelled or advised on |
| `eth_code`, `pir`, `educ_code` | not on a blood report, and lifestyle can't change them |

This list matches `feature_names` in `src/app/ml/registry/rf_model_metadata.json`
and the header of `data/processed/X_train.csv`.

---

## 7. Modules (`src/data_prep/`)

| Module | What it does |
|---|---|
| `config.py` | Paths, NHANES variable names, which columns are features and which are label only, YAML loading |
| `download.py` | Downloads the `.XPT` files for every cycle, skipping ones already there |
| `load.py` | Reads `.XPT` files into pandas, renames columns, unifies the weight column |
| `merge.py` | Joins components on SEQN within each cycle, stacks the cycles, adds the statin/metformin flag |
| `labeling.py` | Builds the three-layer label and marks Layer 2 upgrades |
| `outliers.py` | Implausible values to NaN; winsorising fitted on train |
| `missing.py` | Drops rows missing mandatory markers; imputation fitted on train |
| `features.py` | Engineered features |
| `encode_scale.py` | Encoding, feature matrix, scaler fitted on train |
| `pipeline.py` | Runs every step in order and saves the outputs |
| `scripts/run_data_prep.py` | Command-line entry point (`--download`, `--impute`, …) |

---

## 8. Running it

```bash
pip install -r requirements.txt
python -m scripts.run_data_prep --download     # first run: downloads the XPT files, then builds
python -m scripts.run_data_prep                # later runs reuse the downloaded files
```

Outputs go to `data/processed/`: `train.csv`, `test.csv`, `X_train*.csv`,
`X_test*.csv`, `y_*.csv`, and `artifacts/` (the fitted imputer and scaler,
winsorising bounds, feature names and `prep_summary.json`).

---

## 9. Results

During development, a synthetic test of labelling, outliers, missing values,
features and encoding confirmed that train and test have the same columns, no
NaNs remain in the features, no label-only column is in the feature matrix, and
Layer 2 upgrades are counted.

**Full NHANES run (all 36 files downloaded and checked as valid XPORT):**

| Step | Rows |
|---|---|
| Merged (2013-14, 2015-16, 2017-20) | 35,706 |
| Adults only (18 and over) | 21,798 |
| After dropping rows missing a mandatory blood marker | 19,308 |
| Train / test (stratified 80/20) | 15,446 / 3,862 |
| Feature columns | 27 |

- **Labels for all 21,798 adults:** borderline 11,416, serious 6,855, normal 3,527.
  Layer 1 on its own gives borderline 10,923, serious 5,659, normal 5,216.
- **Training set:** borderline 8,372, serious 5,245, normal 1,829 (11.8% normal).
- **Layer 2 raised the label for 2,587 adults (11.9%).** These records form the
  novelty slice evaluated in Phase 4.
- **Removing blood pressure, BMI and waist changed the labels a lot.** The normal
  share rose from 2.9% to 11.8%, which mostly fixed the class imbalance. Serious
  labels nearly halved (12,898 → 6,855), so blood pressure had been behind most
  of them, which didn't fit a study about blood biomarkers. The share of records
  raised by Layer 2 went up from 5.1% to 11.9%, because far more people now have
  normal-looking bloods alongside a diagnosis or medication.
- **Needing only the three mandatory blood markers** cut the rows dropped for
  missing data from 4,400 to 2,490, keeping about 11% more of the sample.
- **Limitation:** obesity and high blood pressure aren't assessed; the app refers
  weight and blood pressure to the GP.

The CDC changed its download URLs to `…/Public/{startyear}/DataFiles/{FILE}.XPT`,
and it returns an HTML page with status 200 for files that don't exist. The
downloader therefore checks the XPORT header of every file and rejects HTML
pages, so a broken download can't slip through unnoticed.

---

## 10. Limitations

1. **Class balance.** Most records are borderline or serious, because
   cardiometabolic problems are common in NHANES. Phase 4 uses balanced class
   weights and reports macro-F1 as well as accuracy.
2. **US data, UK thresholds.** The model learns patterns from US data while the
   thresholds come from UK guidelines. The model is never used to diagnose, and
   the rule engine owns the thresholds in the app.
3. **Survey weights aren't used in training.** They're kept as columns, but
   weighted training is out of scope for this model.
