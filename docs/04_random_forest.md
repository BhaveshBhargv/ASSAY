# Phase 4: Random Forest risk model

**Author:** Bhavesh Bhargava, MSc Advanced Data Science
**Code:** `src/app/ml/`, trained and evaluated on the Phase 2 dataset
**Model:** scikit-learn `RandomForestClassifier`, saved with joblib

The aim was to find combinations of blood results that point to higher risk even
when each result looks normal or borderline by itself. The model is the
probabilistic counterpart to the rule engine.

---

## 1. Design decisions

| Decision | Choice | Reason |
|---|---|---|
| Model | RandomForestClassifier | Works well on tabular data, picks up interactions between features, and can be explained. |
| Class imbalance | `class_weight="balanced"` | Weights classes by inverse frequency without creating synthetic data. |
| Tuning | GridSearchCV over a small grid | Exhaustive and reproducible. |
| Cross-validation | StratifiedKFold(5) | Keeps the class proportions in every fold. |
| Selection metric | macro-F1 | Gives all three classes equal weight, which suits imbalanced classes. |
| Scaling | none | Trees don't need scaled inputs. |
| Features | 27, **clinical only** (biomarkers, age, sex and engineered features) | A blood report doesn't include social or demographic details, so the model shouldn't depend on them. Blood pressure, BMI and waist aren't used either. |

**Best hyperparameters (CV):** `max_depth=20, max_features=0.5,
min_samples_leaf=1, n_estimators=500`, with a CV macro-F1 of **0.868**.

---

## 2. Features (27)

The actionable blood biomarkers (including the imputed fasting tests), **`age`**,
one-hot **`sex`**, and features derived from the bloods (`tc_hdl_ratio`,
`tg_hdl_ratio`, `tyg_index`, `age_band`). The flag-only markers (electrolytes,
WBC, platelets) and vitamin D **aren't used** (see Phases 2 and 3).

**No social or demographic inputs.** Ethnicity, income-to-poverty ratio and
education (`eth_*`, `pir`, `educ_code`) were removed because a blood report
doesn't include them and lifestyle can't change them. Retraining without them
left the headline metrics practically unchanged (accuracy 0.890 → 0.889,
macro-F1 0.868 → 0.867), so they weren't adding anything. The Random Forest can
combine biomarkers by itself, and the ratio features give it the most useful
combinations directly.

---

## 3. Test set results (n = 3,862)

| Metric | Value |
|---|---|
| Accuracy | **0.889** |
| Macro-F1 | **0.867** |
| Weighted-F1 | 0.891 |
| ROC-AUC (macro, one-vs-rest) | **0.973** |

**By class:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| normal | 0.707 | 0.897 | 0.791 | 457 |
| borderline | 0.888 | 0.912 | 0.900 | 2,094 |
| serious | 0.983 | 0.848 | 0.911 | 1,311 |

**Confusion matrix** (rows are the true class, columns the prediction):

| | → normal | → borderline | → serious |
|---|---|---|---|
| **normal** | 410 | 47 | 0 |
| **borderline** | 165 | 1,910 | 19 |
| **serious** | 5 | 194 | 1,112 |

Only **5 of the 1,311 serious cases (0.4%)** were predicted normal. Almost all
errors are one class away; the largest groups are serious predicted as
borderline (194) and borderline predicted as normal (165).

---

## 4. What the headline numbers mean

For most records the label is close to a fixed function of the Layer 1
thresholds, and the model sees the same biomarkers those thresholds use. The
model agrees with the rule engine on **93.5%** of test records, and **97.4%** of
its serious predictions were already serious by the rules. Most of the 0.89
accuracy is therefore the forest learning the rule engine back. That shows it
reproduces the guideline-based severity closely, but it isn't evidence that it
predicts clinical risk on its own.

The test that could show that is the novelty slice (§5).

---

## 5. Novelty slice

The **novelty slice** is the test records whose blood results looked normal or
borderline by the rules, but whose diagnosis or medication answers put them
higher (`label_upgraded_by_l2`). The model never sees those answers, so this
slice tests whether it can find the hidden risk from the blood results alone.

There are 306 such records. None of them is normal (true labels: borderline 127,
serious 179), and the rules never grade any of them as serious, because that's
what puts them in the slice.

| Predictor | Exact accuracy (95% CI) |
|---|---|
| **Random Forest** | **18.6%** (14.7–23.4) |
| Always "serious" | 58.5% (52.9–63.9) |
| Random draw from the training class mix | 42.3% (37.3–47.7) |
| Always "borderline" | 41.5% (36.1–47.1) |
| Always "normal" | 0.0% |

**The model does worse than predicting "serious" for everyone.** Against that
baseline it gets 47 records right that the baseline gets wrong, and 169 wrong
that the baseline gets right (exact McNemar p = 2.4 × 10⁻¹⁷). It predicts normal
for 85 records, borderline for 211 and serious for only 10.

An earlier version of this document gave a **"risk detected" rate of 72.2%** (any
prediction other than normal) as the main result. That rate means nothing on this
slice: it has no normal records, so predicting any single non-normal class for
everyone scores 100%.

**Is it because medication brings the blood results back to normal?** The slice
was split by medication:

| Group | n | Model accuracy | Best constant prediction |
|---|---|---|---|
| On medication that lowers a measured marker (lipids, glucose) | 184 | 17.4% (12.6–23.5) | 76.6% (always serious) |
| On blood pressure medication only | 70 | 27.1% (18.1–38.5) | 74.3% (always borderline) |
| No medication | 52 | 11.5% (5.4–23.0) | 61.5% (always borderline) |

People on no medication aren't graded any better than people whose medication
lowers the measured markers (11.5% against 17.4%), and accuracy for the
marker-lowering group doesn't differ significantly from everyone else in the
slice (Fisher's exact p = 0.55). Treatment masking the blood results doesn't
explain the result.

The simpler explanation is how the slice is built. Layer 1 never calls these
records serious, the model has closely learned Layer 1, and the labels that make
these records different come from questionnaire answers the model can't see. It
predicts serious for only 10 of the 179 records that are truly serious.

Phase 9 also looks at what this means once the model is fused with the rules in
the app.

---

## 6. Explainability

- **Gini importance and SHAP (TreeExplainer)** agree on the most important
  features. By mean |SHAP|: `hba1c_pct`, `total_chol_mgdl`,
  `fasting_glucose_mgdl`, `age`, `hdl_mgdl`, `tg_hdl_ratio`, `tc_hdl_ratio`,
  `age_band`, `ldl_mgdl`, then blood count markers such as `mch`.
- The cardiometabolic markers matter most, which makes clinical sense, and the
  ratio features and TyG index contribute as well. These are the same markers the
  Layer 1 thresholds use, which fits with §4.
- Outputs in `reports/phase4/`: `confusion_matrix.png`, `roc_curves.png`,
  `feature_importance.{png,csv}`, `shap_summary_{normal,borderline,serious}.png`,
  `classification_report.txt`, `metrics.json`. The novelty-slice baselines and
  plot (`novelty_baselines.png`) are in `reports/phase9/`.

---

## 7. Modules (`src/app/ml/`)

| File | What it does |
|---|---|
| `data.py` | Loads the Phase 2 matrices, encodes labels, provides the novelty mask and medication groups |
| `train.py` | GridSearchCV with StratifiedKFold; saves the model and its metadata (joblib) |
| `evaluate.py` | Metrics, confusion matrix, ROC curves, novelty-slice baselines and medication groups, agreement with the rules, fusion analysis, plots |
| `explain.py` | Gini and SHAP importance, SHAP plots per class |
| `predict.py` | `RiskModel`, used by the API and fusion to make predictions |
| `registry/` | `rf_model.joblib` and `rf_model_metadata.json` |
| `scripts/train_model.py` | Train, evaluate and explain in one go |
| `tests/test_ml.py` | End-to-end smoke test on synthetic data |

**Run:** `python scripts/train_model.py` (or add `--quick` for a small grid).

---

## 8. Limitations

1. **Labels and features overlap.** The label is mostly built from the same
   biomarkers the model sees, which inflates the headline metrics (§4).
2. **The model doesn't find the hidden risk.** On the novelty slice it's worse
   than a constant prediction (§5).
3. **US training data with UK-guideline labels.** The model picks up risk
   patterns and is never used to diagnose; the rule engine owns the thresholds.
4. **No survey weights in training**, as decided in Phase 2.

The trained model and its metadata are committed to `src/app/ml/registry/`, so
the app runs without retraining. Phase 9 evaluates the same model and writes the
results to `reports/phase9/`.
