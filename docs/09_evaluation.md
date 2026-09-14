# Phase 9: Evaluation and explainability

**Author:** Bhavesh Bhargava, MSc Advanced Data Science
**Code:** `src/app/eval/`, with results in `reports/phase9/`
**Run:** `python scripts/evaluate_system.py` (`--no-llm` skips the LLM step)

One script evaluates the Random Forest, its explanations, the guideline
retrieval and the LLM recommendations, and writes a JSON summary
(`evaluation_summary.json`) and plots.

---

## 1. Random Forest (test set, n = 3,862)

| Metric | Value |
|---|---|
| Accuracy | **0.889** |
| Precision (macro) | **0.859** |
| Recall (macro) | **0.886** |
| F1 (macro) | **0.867** |
| ROC-AUC (macro, one-vs-rest) | **0.973** |

Per-class results and the confusion matrix are in `reports/phase9/metrics.json`,
with plots in `ml_headline_metrics.png`, `confusion_matrix.png` and
`roc_curves.png`. Only **5 of 1,311 (0.4%)** serious cases were predicted normal.

### Agreement with the rule engine

The model agrees with the rule engine on **93.5%** of test records, and **97.4%**
of its serious predictions were already serious by the rules (it calls only 29
records serious that the rules don't). The headline numbers above mostly measure
how well it has learned the rules. Phase 4 §4 goes into this.

### Novelty slice

On the 306 records where the diagnosis or medication answers raised the label,
the model's exact accuracy is **18.6%** (95% CI 14.7–23.4%). Predicting
"serious" for all of them scores **58.5%**, and the model is significantly worse
(exact McNemar p = 2.4 × 10⁻¹⁷). Splitting by medication doesn't change the
picture. The full breakdown is in Phase 4 §5, and the comparison is plotted in
`novelty_baselines.png`.

The "risk detected" rate of 72.2% in `metrics.json` shouldn't be used as a
result: the slice has no normal records, so any constant non-normal prediction
scores 100% on it.

### Does fusing the model with the rules help?

The app uses `max(rules, model)`, so the model can only raise the severity. On
the test set:

| Configuration | Accuracy (95% CI) |
|---|---|
| Rule engine only | **92.1%** (91.2–92.9) |
| Random Forest only | 88.9% (87.8–89.8) |
| Fused, `max(rules, model)` | 91.8% (90.9–92.7) |

The model raised the severity for 135 records:
- **69 were in the novelty slice**, where raising it was right. 57 of those
  reached the true label and 12 were still too low.
- **66 were false escalations** outside the slice (47 normal → borderline, 19
  borderline → serious). That's a false escalation rate of **1.9%** (1.5–2.4%)
  of the 3,556 other records.

Compared with the rules alone, fusion fixes 57 records and breaks 66, a net
change of −9 (exact McNemar p = 0.47). So adding the model doesn't significantly
improve the result, and false escalations are the only harm this design can
cause.

### Explainability

- **SHAP (TreeExplainer) and Gini importance** agree on the most important
  features: `hba1c_pct, total_chol_mgdl, fasting_glucose_mgdl, age, hdl_mgdl,
  tg_hdl_ratio, tc_hdl_ratio, age_band`. These are the cardiometabolic markers,
  age, and the ratio features built from them.
- Outputs: `feature_importance.{png,csv}`, `shap_summary_{normal,borderline,serious}.png`.

---

## 2. Guideline retrieval (7 test queries)

| K | Precision@K | Recall@K |
|---|---|---|
| 1 | **0.857** | 0.24 |
| 3 | **0.667** | 0.44 |
| 5 | 0.543 | 0.57 |

**Context relevance:** mean top-K cosine similarity **0.52**, tag overlap **0.54**.
Plot: `rag_precision_recall_at_k.png`.

**How relevance is judged:** a passage counts as relevant if its biomarker tags
include one of the query's target biomarkers. This is reproducible and needs no
manual labelling, but it's only an approximation of human relevance judgements.

A **Precision@1 of 0.86** means the top passage is almost always on topic.
Precision falls as K grows because the corpus only has a few passages for each
condition, so after about three there aren't many more relevant ones to find.
Recall goes up with K, as expected.

### A problem the evaluation found

Looking at individual queries, the cardiometabolic ones scored perfectly (P@3 =
1.0) but **fatty liver scored 0.0**. `build_query` had the phrase
"cardiometabolic risk" written into every query, which pulled the embeddings
towards metabolic passages. Removing it and starting the query with the flagged
markers raised fatty liver P@3 from 0.00 to 0.33, overall P@3 from 0.62 to 0.67,
and R@5 from 0.48 to 0.57. Liver and anaemia queries are still harder, since
there are fewer and more general passages for them. A larger corpus would help.

---

## 3. LLM recommendations

These metrics are calculated over every advice item generated:

| Metric | Definition |
|---|---|
| **Groundedness** | share of advice items citing a passage that was actually retrieved |
| **Faithfulness** | share whose text matches its cited evidence, measured as embedding cosine similarity ≥ τ (0.35). Deterministic, with no LLM acting as judge |
| **Hallucination rate** | share that are uncited **or** unfaithful (`1 − faithful share`) |

`score_recommendation` and `aggregate` are plain functions **tested offline**
with a fake embedder (`tests/test_eval.py`). A cited item that matches its
evidence counts as faithful, an item citing a passage that doesn't exist counts
as a hallucination, and a cited item that doesn't match its evidence counts as
unfaithful.

**Results:**

| | |
|---|---|
| Provider | OpenRouter, `nex-agi/nex-n2.5-pro:free` |
| Sample | 5 evaluation patients, 38 advice items |
| Groundedness | 1.00 |
| Faithfulness | 1.00 (mean cosine 0.69) |
| Hallucination rate | 0.00 |

Plot: `llm_quality.png`.

These numbers need to be read with care. Items are scored **after** the Phase 6
checks have run (`run_llm_eval` calls `verify` first), so uncited items have
already been removed and groundedness is 1.0 by design. The faithfulness
threshold of 0.35 is fairly lenient, and the sample is small: five reports from
one free model.

To run it again you need an LLM. Without one, the script reports the step as
skipped instead of failing:

```
python scripts/evaluate_system.py                 # with Ollama running
python scripts/evaluate_system.py --provider openrouter
```

---

## 4. Modules (`src/app/eval/`)

| File | What it does |
|---|---|
| `config.py` | Evaluation patients, retrieval test queries, thresholds, the `reports/phase9` path |
| `ml_eval.py` | Reuses Phase 4's `evaluate` and `explain`, and adds the headline metrics chart |
| `rag_eval.py` | Precision@K, Recall@K, context relevance and the plot |
| `llm_eval.py` | Groundedness, faithfulness and hallucination rate, and runs the pipeline to produce reports |
| `scripts/evaluate_system.py` | Runs everything and writes `evaluation_summary.json` and the plots |
| `tests/test_eval.py` | Offline tests for the LLM metrics and the retrieval evaluation |

---

## 5. Limitations

1. **Retrieval relevance comes from the corpus tags**, not from human judgements.
2. **Faithfulness uses embedding similarity** as a stand-in for real entailment.
   It's fast and deterministic, but an LLM judge could be added as a second
   opinion.
3. **The LLM results come from a small sample** (5 reports, one model) and are
   scored after the checks have removed uncited items.
4. **The model's headline metrics mostly reflect agreement with the rule engine.**
   On the novelty slice it's worse than a constant prediction, and fusing it with
   the rules doesn't significantly change accuracy.
