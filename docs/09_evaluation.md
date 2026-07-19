# Phase 9 — System Evaluation & Explainability

**Author:** Bhavesh Bhargava — MSc Advanced Data Science
**Status:** Implemented (`src/app/eval/`), run end-to-end; artefacts in `reports/phase9/`.
**Run:** `python scripts/evaluate_system.py` (`--no-llm` to skip the LLM step).

> Evaluates all four moving parts of the pipeline — the Random Forest, its
> explainability, the RAG retriever, and the LLM recommendations — with one script
> that writes a consolidated JSON summary and publication-ready plots.

---

## 1. Machine learning (held-out test set, n = 3,862)

| Metric | Value |
|---|---|
| Accuracy | **0.889** |
| Precision (macro) | **0.859** |
| Recall (macro) | **0.886** |
| F1 (macro) | **0.867** |
| ROC-AUC (macro, OVR) | **0.973** |

Per-class precision/recall/F1 and the confusion matrix are in
`reports/phase9/metrics.json`; plots: `ml_headline_metrics.png`,
`confusion_matrix.png`, `roc_curves.png`. Safety-critical error (true *serious* →
predicted *normal*) is **5 / 1,311 (0.4%)**. The **Layer-2 novelty slice** (the
study's core question) is reported too: hidden-risk detection **72.2%**.

### Explainability
- **SHAP (TreeExplainer) + Gini importance** agree on the drivers. Top features:
  `hba1c_pct, total_chol_mgdl, fasting_glucose_mgdl, age, hdl_mgdl, tg_hdl_ratio,
  tc_hdl_ratio, age_band` — all cardiometabolic markers, age, and the engineered
  ratios (confirming interaction signal is used).
- Artefacts: `feature_importance.{png,csv}`, `shap_summary_{normal,borderline,serious}.png`.

---

## 2. RAG retrieval (7 gold queries)

| K | Precision@K | Recall@K |
|---|---|---|
| 1 | **0.857** | 0.24 |
| 3 | **0.667** | 0.44 |
| 5 | 0.543 | 0.57 |

**Context relevance:** mean top-K cosine **0.52**; tag-overlap **0.49**.
Plot: `rag_precision_recall_at_k.png`.

**Relevance judgement (silver standard):** a passage is relevant to a query if its
biomarker tags intersect the query's target biomarkers — reproducible, no manual
annotation (a documented limitation vs. human relevance labels).

**Precision@1 = 0.86** means the single best passage is almost always on-topic;
precision falls with K (the small corpus has only a handful of passages per
condition, so beyond K≈3 there aren't more relevant ones to find), while recall
rises as expected.

### A weakness the evaluation caught (and a fix)
Per-query, cardiometabolic queries scored perfectly (P@3 = 1.0) but **fatty-liver
retrieval scored 0.0**. Root cause: `build_query` hard-coded the phrase
*"cardiometabolic risk"* into **every** query, biasing the embedding toward
metabolic passages. Removing it (leading with the flagged markers instead) raised
fatty-liver P@3 0.00 → 0.33 and aggregate P@3 0.62 → 0.67, R@5 0.48 → 0.57.
Liver/anaemia remain harder (fewer, more general passages) — an honest limit of a
small curated corpus, addressable by expanding it.

---

## 3. LLM recommendations

Metrics, all over the generated advice items:

| Metric | Definition |
|---|---|
| **Groundedness** | share of advice items that cite a real retrieved passage |
| **Faithfulness** | share whose text is *entailed* by its cited evidence — judged by embedding cosine ≥ τ (0.35); deterministic, no LLM judge required |
| **Hallucination rate** | share that are ungrounded **or** unfaithful (`1 − faithful share`) |

`score_recommendation` / `aggregate` are pure functions, **unit-tested offline**
with a controlled fake embedder (`tests/test_eval.py`): a grounded+matching item
scores faithful, an item citing a non-existent passage counts as hallucination,
and a grounded-but-unrelated item is caught as unfaithful. So the metric *logic*
is verified.

Producing the *numbers* requires generating recommendations, which needs a
reachable LLM (local Ollama by default). Without one, the script reports the step
as skipped rather than failing:

```
python scripts/evaluate_system.py                 # with Ollama running
python scripts/evaluate_system.py --provider anthropic
```

Groundedness is additionally enforced *at generation time* by the Phase-6 guards
(ungrounded/diagnostic items are stripped), so the deployed system's groundedness
is high by construction; this evaluation measures faithfulness beyond mere
citation presence.

---

## 4. Module map (`src/app/eval/`)

| File | Responsibility |
|---|---|
| `config.py` | Test patients, RAG gold queries, thresholds, `reports/phase9` path |
| `ml_eval.py` | Reuses Phase-4 `evaluate` + `explain`; adds the headline-metrics chart |
| `rag_eval.py` | Precision@K / Recall@K / context relevance + plot |
| `llm_eval.py` | Groundedness / faithfulness / hallucination (+ pipeline driver) |
| `scripts/evaluate_system.py` | Orchestrator → `evaluation_summary.json` + plots |
| `tests/test_eval.py` | Offline tests for the LLM metric logic + RAG eval |

---

## 5. Limitations
1. **RAG relevance is a corpus-tag silver standard**, not human-annotated.
2. **Faithfulness is an embedding-cosine proxy** for entailment (fast, deterministic);
   an LLM-judge variant could be added for a second opinion.
3. **LLM numbers need a live model** — the framework and logic are complete and tested;
   the metrics populate when run against Ollama/Anthropic.
4. **ML headline metrics reflect label–feature coupling** (see Phase 4 §4); the
   novelty slice remains the fair test of genuine ML contribution.

**Phase 9 exit criteria met** — ML, explainability, RAG, and LLM evaluation are
implemented, visualised, and (where offline-possible) verified.
