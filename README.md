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
   NICE/NHS/WHO thresholds (YAML; no code changes to add a biomarker).
2. **Random Forest** — catches combined-risk patterns single thresholds miss
   (clinical-only: biomarkers + age + sex).
3. **Fusion** — one safety-dominant severity (`max(rules, model)`); the model can
   only *escalate*.
4. **RAG** — retrieves the top-k cited guideline passages for the assessment.
5. **LLM** — turns that evidence into grounded, non-diagnostic lifestyle advice;
   every item carries a rationale and a citation, and ungrounded/diagnostic output
   is stripped by guards.

**Interfaces:** a Streamlit **dashboard** and a FastAPI **REST API** (Swagger at
`/docs`). Both call the same engine.

---

## Quick start

### Option A — Docker (recommended)

```bash
cp .env.example .env          # optional: add OPENROUTER_API_KEY for the LLM step
docker compose up --build
```
- Dashboard → http://localhost:8502
- API + Swagger → http://localhost:8000/docs

### Option B — Local Python

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run streamlit_app/dashboard.py             # dashboard :8502
# in another shell:
uvicorn app.api.main:app --app-dir src --reload      # API :8000
```

The trained model and RAG index are included in the repo, so it runs out of the
box. The assessment (rules + model + retrieval) works with **no API key**; only
the LLM recommendation step needs one.

---

## Installation guide

**Requirements:** Python 3.11–3.13 (3.12 recommended), ~2 GB disk (PyTorch +
embedding model), Docker (optional).

1. Clone the repo and enter it.
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
   The first run downloads the `all-MiniLM-L6-v2` embedding model (~90 MB).
3. (Optional) Configure the LLM — copy `.env.example` to `.env` and set a key
   (see **Environment configuration**).
4. (Optional) Rebuild artefacts if you change the data/config:
   ```bash
   make build-index      # RAG FAISS index
   make train            # Random Forest (train → evaluate → explain)
   ```

`make help` lists all convenience commands.

---

## Environment configuration

Config comes from environment variables (or a git-ignored `.env`, loaded
automatically). Nothing is required to run the assessment.

| Variable | Purpose | Default |
|---|---|---|
| `OPENROUTER_API_KEY` | OpenRouter key for the LLM step | — |
| `ASSAY_OPENROUTER_MODEL` | OpenRouter model slug | `tencent/hy3:free` |
| `ANTHROPIC_API_KEY` | If using `provider=anthropic` | — |
| `OLLAMA_BASE_URL` | Reach a local/remote Ollama | local daemon |
| `ASSAY_LOG_LEVEL` | API log level | `INFO` |
| `ASSAY_CORS_ORIGINS` | Allowed API origins (comma-sep) | localhost:8501,8502 |
| `ASSAY_MAX_UPLOAD_BYTES` | Upload size limit | 10 MB |

**LLM providers** (all behind one port, swappable per request): **OpenRouter**
(default, cloud, free-tier models available), **Anthropic**, or **Ollama** (local).

---

## Deployment guide

### Docker Compose (two services, one image)

```bash
cp .env.example .env          # add keys as needed
docker compose up --build -d  # api :8000, dashboard :8502
docker compose logs -f
docker compose down
```

- The image bakes in the embedding model, RF model, and FAISS index, so
  containers start ready to serve.
- To use a **local LLM** instead of OpenRouter:
  ```bash
  docker compose --profile ollama up --build -d
  docker compose exec ollama ollama pull llama3.1
  # set in .env:  OLLAMA_BASE_URL=http://ollama:11434
  ```
  and choose `provider=ollama` in the API / dashboard.

### CI/CD (GitHub Actions)

`.github/workflows/ci.yml` runs on every push/PR:
1. **lint** — `ruff` (informational),
2. **test** — installs CPU PyTorch + deps and runs all test suites,
3. **docker** — builds the image and **pushes to GHCR** (`ghcr.io/<owner>/<repo>`)
   on pushes to `main`.

No secrets needed — it uses the built-in `GITHUB_TOKEN`.

### Single container (either service)

```bash
docker build -t assay .
docker run -p 8000:8000 --env-file .env assay          # API
docker run -p 8502:8502 --env-file .env assay \
  streamlit run streamlit_app/dashboard.py --server.port 8502 --server.address 0.0.0.0 --server.headless true
```

---

## Testing instructions

Each suite is a self-contained runner (no pytest required):

```bash
make test        # runs them all
# or individually:
python tests/test_rule_engine.py   # 19 — rule engine
python tests/test_parser.py        #  7 — report parsing (units, plausibility)
python tests/test_rag.py           #  5 — retrieval
python tests/test_ml.py            #  1 — RF train/evaluate/explain smoke
python tests/test_recommend.py     #  7 — fusion, guards, generation
python tests/test_api.py           #  8 — FastAPI endpoints (TestClient)
python tests/test_eval.py          #  5 — evaluation metrics
```

The LLM step is exercised offline with a fake provider, so tests need **no API
key** and no running model. `/recommend` is asserted to fail *gracefully* (typed
502) when no LLM is reachable.

**System evaluation** (Phase 9): `python scripts/evaluate_system.py` writes
metrics + plots to `reports/phase9/` (ML accuracy/precision/recall/F1/ROC-AUC,
SHAP + importance, RAG Precision@K/Recall@K, and — with a live LLM —
groundedness/faithfulness/hallucination).

---

## Project structure

```
src/app/
  domain/        clinical rule engine (entities, services)
  rules/         YAML ruleset loader
  ml/            Random Forest: data, train, evaluate, explain, predict, registry/
  rag/           embedder, FAISS store, retriever, index/
  recommend/     fusion + LangChain LLM engine (providers behind a port) + guards
  ingestion/     shared biomarker catalog + report parser (CSV/JSON/PDF, units)
  eval/          Phase-9 ML / RAG / LLM evaluation
  api/           FastAPI: controllers → services → repositories (DI)
streamlit_app/   dashboard.py, pages/, ui/, services/
config/          clinical_rules.yaml (single source of truth) + thresholds
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
| GET | `/health` | Liveness/readiness |

Interactive docs at `/docs` (Swagger) and `/redoc`.

---

## Disclaimer

Educational research project — **not for clinical use** and **not a medical
diagnosis**. Always consult a qualified healthcare professional. Guideline content
is paraphrased with attribution; the model is trained on US NHANES data with
UK-guideline labels and detects *risk patterns*, not disease.
