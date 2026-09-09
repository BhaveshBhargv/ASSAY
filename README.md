# Assay — Blood-Test Lifestyle Recommender

**Personalised lifestyle recommendations from blood test reports, using machine
learning, retrieval-augmented generation (RAG), and clinical guidelines.**

> MSc Advanced Data Science dissertation project — Bhavesh Bhargava.
> **Not a diagnostic device.** It produces evidence-based *lifestyle* guidance
> grounded in NICE / NHS / WHO material, never a diagnosis.

---

## What it does

A blood panel goes through a five-stage pipeline:

```
Rules  →  Random Forest  →  Fusion  →  RAG retrieval  →  LLM
(thresholds) (hidden-pattern (one severity) (cited NICE/NHS/  (grounded, cited
              risk)                           WHO passages)     recommendations)
```

1. **Clinical rule engine** — classifies each biomarker against configurable
   NICE/NHS/WHO thresholds (`config/clinical_rules.yaml`; no code changes to add a
   biomarker).
2. **Random Forest** — catches combined-risk patterns single thresholds miss
   (clinical-only: biomarkers + age + sex).
3. **Fusion** — one safety-dominant severity (`max(rules, model)`); the model can
   only *escalate*, never soften.
4. **RAG** — retrieves the top-k cited guideline passages for the assessment.
5. **LLM** — turns that evidence into grounded, non-diagnostic lifestyle advice;
   every item carries a rationale and a citation, and ungrounded or diagnostic
   output is stripped by guards.

**Interfaces:** a Streamlit **dashboard** and a FastAPI **REST API** (Swagger at
`/docs`). Both call the same engine directly.

---

## Quick start

```bash
git clone <repo-url> && cd ASSAY

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only, optional
pip install -r requirements.txt

streamlit run streamlit_app/dashboard.py
```

The dashboard opens at <http://localhost:8501>. To run the REST API as well, in a
second shell:

```bash
uvicorn app.api.main:app --app-dir src --reload      # API + Swagger at :8000/docs
```

The trained Random Forest and the FAISS guideline index are committed to the
repo, so it runs out of the box. The assessment (rules + model + retrieval) works
with **no API key**; only the LLM recommendation step needs one.

---

## Installation

**Requirements**

| | |
|---|---|
| Python | 3.11 – 3.13 (3.12 recommended) |
| Disk | ~2 GB (PyTorch + the embedding model) |
| RAM | 8 GB is comfortable; 16 GB if you also run a local LLM |
| GPU | Not required — everything runs on CPU |

**Steps**

1. Clone the repository and change into it.
2. Create and activate a virtual environment (see Quick start).
3. Install dependencies. `sentence-transformers` pulls PyTorch; installing the
   **CPU wheel first** avoids downloading the multi-GB CUDA build:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```
4. The first run downloads the `all-MiniLM-L6-v2` embedding model (~90 MB) and
   caches it locally. Everything after that works offline, apart from the LLM step.
5. *(Optional)* Configure an LLM provider — see **Configuration**.
6. *(Optional)* Rebuild artefacts if you change the data or config:
   ```bash
   make build-index      # RAG FAISS index
   make train            # Random Forest (train → evaluate → explain)
   ```

`make help` lists the convenience commands. On Windows, `make` needs Git Bash or
WSL; otherwise run the underlying commands directly.

---

## Configuration

Configuration comes from environment variables, or from a git-ignored `.env` file
in the project root which is loaded automatically. **Nothing is required to run
the assessment** — rules, Random Forest and retrieval all work unconfigured.

| Variable | Purpose | Default |
|---|---|---|
| `OPENROUTER_API_KEY` | OpenRouter key for the LLM step | — |
| `ASSAY_OPENROUTER_MODEL` | OpenRouter model slug | see `src/app/recommend/config.py` |
| `ANTHROPIC_API_KEY` | If using `provider=anthropic` | — |
| `OLLAMA_BASE_URL` | Reach an Ollama daemon elsewhere | local daemon |
| `ASSAY_LOG_LEVEL` | API log level | `INFO` |
| `ASSAY_CORS_ORIGINS` | Allowed API origins (comma-separated) | `localhost:8501,8502` |
| `ASSAY_MAX_UPLOAD_BYTES` | Upload size limit | 10 MB |

Example `.env`:

```
OPENROUTER_API_KEY=sk-or-...
```

**LLM providers** sit behind one port (`ILLMProvider`) and are swappable per
request:

- **OpenRouter** (default) — cloud, free-tier model slugs available.
- **Anthropic** — cloud, needs `ANTHROPIC_API_KEY`.
- **Ollama** — fully local, nothing leaves the machine. Install Ollama, run
  `ollama pull llama3.1`, then select `provider=ollama`. Expect minutes rather
  than seconds per report on a CPU-only laptop.

---

## Running locally

```bash
# Dashboard (default port 8501)
streamlit run streamlit_app/dashboard.py

# REST API + Swagger UI at http://localhost:8000/docs
uvicorn app.api.main:app --app-dir src --reload
```

The two are independent — the dashboard imports the engine directly and does not
call the API. Run either, or both.

Other entry points:

```bash
python scripts/run_rule_engine.py        # classify a sample panel, print JSON
python scripts/build_rag_index.py        # rebuild the FAISS guideline index
python scripts/train_model.py            # train → evaluate → explain the RF
python scripts/run_data_prep.py          # rebuild the NHANES dataset
python scripts/evaluate_system.py        # full ML + RAG + LLM evaluation
```

---

## Using the dashboard

1. Choose a report source in the sidebar: **Sample patient**, **Upload file**, or
   **Manual entry**.
2. Uploads accept **CSV**, **JSON** and **PDF**. PDF extraction reads both common
   lab layouts (one row per test, and the stacked name/method/value form), pulls
   the patient's age and sex from the report header, converts units to the app's
   canonical ones, and rejects implausible readings.
   **Every extracted value is best-effort — review the form before analysing.**
3. Confirm age and sex (auto-filled from a PDF header when present; required
   otherwise), then press **Analyze**.
4. The assessment — risk summary, severity table, charts, retrieved evidence —
   appears immediately. Recommendations are generated in a background thread and
   stream in when ready, so a slow or unavailable LLM never blocks the rest.
5. **Download PDF report** exports the assessment, with recommendations when
   present.

---

## Testing

Each suite is a self-contained runner — no pytest required:

```bash
make test        # runs them all
```

Or individually:

```bash
python tests/test_rule_engine.py   # 19 — rule engine, bands, aggregation
python tests/test_parser.py        # 18 — report parsing: layouts, units, demographics
python tests/test_rag.py           #  5 — splitter, corpus, retrieval
python tests/test_ml.py            #  1 — RF train/evaluate/explain smoke
python tests/test_recommend.py     #  7 — fusion, generation, guards
python tests/test_api.py           #  8 — FastAPI endpoints (TestClient)
python tests/test_eval.py          #  5 — evaluation metrics
```

**63 tests, all offline.** The LLM is exercised through a fake provider, so no API
key and no running model are needed. `/recommend` is asserted to fail *gracefully*
(a typed 502) when no LLM is reachable.

**System evaluation:** `python scripts/evaluate_system.py` writes metrics and
plots to `reports/phase9/` — ML accuracy / precision / recall / F1 / ROC-AUC, SHAP
and feature importance, RAG Precision@K / Recall@K, and (with a live LLM)
groundedness / faithfulness / hallucination rate.

**CI:** `.github/workflows/ci.yml` runs `ruff` (informational) and all seven
suites on every push and pull request.

---

## Project structure

```
src/app/
  domain/        clinical rule engine (entities, enums, services)
  rules/         YAML ruleset loader + validation
  ml/            Random Forest: data, train, evaluate, explain, predict, registry/
  rag/           embedder, FAISS store, retriever, index/
  recommend/     fusion + LLM engine (providers behind a port) + guards
  ingestion/     shared biomarker catalog, unit conversion, report parser
  eval/          ML / RAG / LLM evaluation
  api/           FastAPI: controllers → services → repositories (DI)
src/data_prep/   NHANES download, merge, labelling, features, pipeline
streamlit_app/   dashboard.py, pages/, ui/, services/, sample_data/
config/          clinical_rules.yaml (single source of truth), nhanes_files.yaml
data/guidelines/ curated NICE/NHS/WHO corpus (paraphrased, cited)
scripts/         build_rag_index, train_model, evaluate_system, run_api, ...
tests/           per-phase test runners
docs/            01–10 phase documentation
```

---

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/upload` | Parse a blood report (CSV/JSON/PDF) → biomarkers for review |
| POST | `/predict` | Rules + Random Forest + fusion → severity assessment |
| POST | `/recommend` | Full grounded pipeline → cited recommendations |
| POST | `/retrieve` | Semantic search over guideline passages |
| GET | `/model-info` | Ruleset, RF metrics, RAG index, biomarker catalog |
| GET | `/health` | Liveness / readiness probe |

Interactive documentation at `/docs` (Swagger) and `/redoc`.

---

## Privacy

Everything except the LLM step runs entirely on your machine. Report parsing, the
rule engine, the Random Forest and guideline retrieval never make a network call.

When recommendations are generated with a **cloud** provider (OpenRouter or
Anthropic), the request contains the patient's **age, sex and flagged biomarker
values** — de-identified; no name, address or ID is sent. Use the **Ollama**
provider if you need the LLM step to stay local too.

---

## Disclaimer

Educational research project — **not for clinical use** and **not a medical
diagnosis**. Always consult a qualified healthcare professional. Guideline content
is paraphrased with attribution; the model is trained on US NHANES data with
UK-guideline labels and detects *risk patterns*, not disease.
