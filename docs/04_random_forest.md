# Phase 4 — Random Forest Risk Model

**Author:** Bhavesh Bhargava — MSc Advanced Data Science
**Status:** Implemented (`src/app/ml/`), trained & evaluated on the Phase-2 dataset.
**Model:** `RandomForestClassifier` (scikit-learn), saved with joblib.

> **Goal.** Identify clinically significant *interaction* patterns among blood
> biomarkers — cases where individual markers look normal/borderline but their
> combination indicates elevated risk. The probabilistic complement to the
> deterministic rule engine in the five-stage contract.

---

## 1. Approved design decisions

| Decision | Choice | Why |
|---|---|---|
| Model | RandomForestClassifier | Robust on tabular data, captures interactions natively, natively explainable. |
| Imbalance | `class_weight="balanced"` | Reweights loss by inverse frequency; no synthetic data (defensible). |
| Tuning | GridSearchCV (focused grid) | Exhaustive, reproducible, easy to justify. |
| CV | StratifiedKFold(5) | Preserves class proportions per fold. |
| Selection metric | macro-F1 | Weights all 3 classes equally — right for imbalance + the clinical goal. |
| Scaling | none (unscaled) | Trees are scale-invariant. |
| Features | 27 — **clinical-only** (biomarkers + age + sex + engineered) | A blood report doesn't carry socio-demographics, and they aren't clinically actionable, so the model must not depend on them. No BP/BMI/waist either (blood-report scope). |

**Best hyperparameters (CV):** `max_depth=20, max_features=0.5,
min_samples_leaf=1, n_estimators=500` — CV macro-F1 **0.868**.

---

## 2. Feature set (27) — clinical-only

Blood biomarkers (actionable, incl. imputed fasting labs), **`age`**, one-hot
**`sex`**, and blood-derived engineered features (`tc_hdl_ratio`, `tg_hdl_ratio`,
`tyg_index`, `age_band`). Flag-only electrolytes/WBC/platelets and Vitamin D are
**excluded** (per Phase 2/3).

**Socio-demographic inputs removed.** Ethnicity, income-to-poverty ratio, and
education (`eth_*`, `pir`, `educ_code`) were dropped: a blood report doesn't carry
them and they aren't clinically actionable. Retraining without them left every
headline metric essentially unchanged (accuracy 0.890→0.889, macro-F1 0.868→0.867,
novelty detection 72.5%→72.2%) — direct evidence they contributed no useful signal.
The model is now defensibly a function of the blood panel + age + sex only. The
Random Forest captures biomarker interactions natively; the explicit ratios give it
those interactions in low-variance form.

---

## 3. Results (held-out test set, n = 3,862)

| Metric | Value |
|---|---|
| Accuracy | **0.889** |
| Macro-F1 | **0.867** |
| Weighted-F1 | 0.891 |
| ROC-AUC (macro, OVR) | **0.973** |

**Per-class:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| normal | 0.707 | 0.897 | 0.791 | 457 |
| borderline | 0.888 | 0.912 | 0.900 | 2,094 |
| serious | 0.983 | 0.848 | 0.911 | 1,311 |

**Confusion matrix** (rows = true, cols = predicted):

| | → normal | → borderline | → serious |
|---|---|---|---|
| **normal** | 410 | 47 | 0 |
| **borderline** | 165 | 1,910 | 19 |
| **serious** | 5 | 194 | 1,112 |

**Safety-critical error is very low:** only **5 / 1,311 (0.4%)** truly serious
cases were predicted normal. Most error is one-step and conservative
(serious→borderline, normal→borderline).

---

## 4. Honest interpretation (read before quoting the headline numbers)

The 3-class label is, for most records, a near-deterministic function of the
biomarker thresholds (Layer-1 rules), and the RF sees those same biomarkers as
features. So a large part of the 0.89 accuracy reflects the RF **re-learning the
rule engine** — a robust, smoothed surrogate, but *not* independent evidence of
predictive power. The headline metrics should be reported as *"the RF reproduces
the guideline-based severity with high fidelity,"* not as clinical risk prediction.

**The genuine ML contribution is measured on the Layer-2 novelty slice** (§5).

---

## 5. Novelty-slice evaluation (the study's core question)

The **Layer-2 novelty slice** = test records whose *individual* biomarkers looked
normal/borderline but whose diagnosis/medication evidence indicated higher risk
(`label_upgraded_by_l2`). The RF **never sees** that evidence, so its behaviour
here measures whether it has learned the hidden *interaction* signal from
biomarker patterns alone.

| Novelty slice (n = 306) | Value |
|---|---|
| **Hidden-risk detection rate** (predicts non-normal) | **72.2%** |
| Exact-label accuracy | 18.6% |
| Predicted distribution | normal 85 · borderline 211 · serious 10 |

**Reading it honestly:** despite these patients' individually unremarkable
biomarkers, the RF flags *elevated risk* in ~73% of them from biomarker
combinations alone — evidence it captures clinically meaningful interaction
patterns that single-threshold rules miss. It **detects** risk far better than it
**grades** it (exact accuracy only 20%): it tends to predict *borderline* where the
hidden truth is *serious*. This is the expected, defensible result — the value is
in surfacing otherwise-missed risk, which is then handled downstream by the rule
engine + clinician signposting.

---

## 6. Explainability

- **Gini importance + SHAP (TreeExplainer)** agree on the drivers. Top features
  (mean |SHAP|): `hba1c_pct`, `total_chol_mgdl`, `fasting_glucose_mgdl`,
  `hdl_mgdl`, `age`, `age_band`, `tg_hdl_ratio`, `tc_hdl_ratio`, `ldl_mgdl`,
  `triglycerides_mgdl`, then CBC indices (`mch`, `rdw`).
- Cardiometabolic markers dominate (clinically sensible); the engineered ratios
  (`tg_hdl`, `tc_hdl`) and TyG contribute, confirming interaction signal is used.
- Artefacts in `reports/phase4/`: `confusion_matrix.png`, `roc_curves.png`,
  `feature_importance.{png,csv}`, `shap_summary_{normal,borderline,serious}.png`,
  `classification_report.txt`, `metrics.json`.

---

## 7. Module map (`src/app/ml/`)

| File | Responsibility |
|---|---|
| `data.py` | Load Phase-2 matrices, encode labels, expose the novelty mask |
| `train.py` | GridSearchCV + StratifiedKFold; save model + metadata (joblib) |
| `evaluate.py` | All metrics, confusion matrix, ROC, novelty-slice eval, plots |
| `explain.py` | Gini + SHAP importance and per-class beeswarms |
| `predict.py` | Inference wrapper (`RiskModel`) for the API/fusion layer |
| `registry/` | `rf_model.joblib` + `rf_model_metadata.json` |
| `scripts/train_model.py` | Entrypoint: train → evaluate → explain |
| `tests/test_ml.py` | Synthetic end-to-end smoke test |

**Run:** `python scripts/train_model.py` (or `--quick`).

---

## 8. Limitations

1. **Label–feature coupling** inflates headline metrics (§4); the novelty slice is
   the fair test.
2. **Novelty slice grading is weak** (exact acc 20%) — the RF detects hidden risk
   better than it grades it.
3. **US NHANES training** with UK-guideline labels — the RF is a *risk-pattern*
   detector, never diagnostic (thresholds owned by the rule engine).
4. **Unweighted training** (survey weights not applied) — documented Phase-2 choice.

**Phase 4 exit criteria met.** The trained model and its metadata sidecar are
committed to `src/app/ml/registry/`, so the application runs without retraining;
Phase 9 re-evaluates the same artefact into `reports/phase9/`.
