# Assay: blood test lifestyle recommender

Assay reads a blood test report and gives lifestyle recommendations based on
NICE, NHS and WHO guidance. It combines a clinical rule engine, a Random Forest,
guideline retrieval (RAG) and an LLM.

MSc Advanced Data Science dissertation project by Bhavesh Bhargava.

> **This is not a diagnostic tool.** It gives general lifestyle guidance based on
> published guidelines and never makes a diagnosis.

---

## How it works

A blood panel goes through five steps:

```
Rules          →  Random Forest        →  Fusion          →  Retrieval                 →  LLM
(thresholds)      (combined patterns)     (one severity)     (NICE/NHS/WHO passages)      (cited recommendations)
```

1. **Rule engine.** Each biomarker is checked against the thresholds in
   `config/clinical_rules.yaml`. A new biomarker only needs a new entry there.
2. **Random Forest.** Looks for risky combinations of results that no single
   threshold catches. It uses the biomarkers, age and sex only.
3. **Fusion.** The overall severity is `max(rules, model)`, so the model can
   raise the severity but never lower it.
4. **Retrieval.** Pulls the most relevant guideline passages for the assessment.
5. **LLM.** Writes lifestyle advice from those passages. Every item needs a
   reason and a citation, and guards remove anything uncited or diagnostic.

There's a Streamlit dashboard and a FastAPI REST API (Swagger at `/docs`). Both
use the same engine code.

---

## Quick start

```bash
git clone <repo-url> && cd ASSAY

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional, CPU-only build
pip install -r requirements.txt

streamlit run streamlit_app/dashboard.py
```

The dashboard opens at <http://localhost:8501>. To run the API as well, use a
second terminal:

```bash
uvicorn app.api.main:app --app-dir src --reload      # API and Swagger at :8000/docs
```

The trained Random Forest and the FAISS index are committed, so nothing needs
training or building first. The assessment (rules, model and retrieval) works
without an API key. Only the LLM step needs one.

---

## Installation

**Requirements**

| | |
|---|---|
| Python | 3.11 to 3.13 (3.12 recommended) |
| Disk | about 2 GB (mostly PyTorch and the embedding model) |
| RAM | 8 GB is enough; 16 GB if you also run a local LLM |
| GPU | not needed, everything runs on the CPU |

**Steps**

1. Clone the repository and `cd` into it.
2. Create and activate a virtual environment (see Quick start).
3. Install the dependencies. `sentence-transformers` installs PyTorch, so install
   the CPU build first to avoid downloading the multi-GB CUDA version:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```
4. The first run downloads the `all-MiniLM-L6-v2` embedding model (about 90 MB)
   and caches it. After that everything except the LLM step works offline.
5. *(Optional)* Set up an LLM provider, see **Configuration**.
6. *(Optional)* If you change the data or config, rebuild the index and model:
   ```bash
   make build-index      # FAISS guideline index
   make train            # Random Forest: train, evaluate, explain
   ```

`make help` lists the other commands. On Windows `make` needs Git Bash or WSL,
or you can run the commands from the `Makefile` directly.

---

## Configuration

Settings are read from environment variables, or from a `.env` file in the
project root (git-ignored, loaded automatically). None of them are needed for
the assessment itself.

| Variable | Purpose | Default |
|---|---|---|
| `OPENROUTER_API_KEY` | OpenRouter key for the LLM step | none |
| `ASSAY_OPENROUTER_MODEL` | OpenRouter model to use | see `src/app/recommend/config.py` |
| `ANTHROPIC_API_KEY` | needed for `provider=anthropic` | none |
| `OLLAMA_BASE_URL` | Ollama server address | local server |
| `ASSAY_LOG_LEVEL` | API log level | `INFO` |
| `ASSAY_CORS_ORIGINS` | allowed API origins, comma-separated | `localhost:8501,8502` |
| `ASSAY_MAX_UPLOAD_BYTES` | upload size limit | 10 MB |

Example `.env`:

```
OPENROUTER_API_KEY=sk-or-...
```

**LLM providers.** All three implement the same `ILLMProvider` interface and can
be chosen per request:

- **Ollama** runs locally, so nothing leaves your machine. It's the default for
  the API and the scripts. Install Ollama, run `ollama pull llama3.1`, then use
  `provider=ollama`. Without a GPU, expect a report to take minutes.
- **OpenRouter** is a cloud service with some free models. The dashboard uses it.
- **Anthropic** is a cloud service and needs `ANTHROPIC_API_KEY`.

---

## Running locally

```bash
# dashboard, port 8501
streamlit run streamlit_app/dashboard.py

# REST API, Swagger UI at http://localhost:8000/docs
uvicorn app.api.main:app --app-dir src --reload
```

The dashboard imports the engine directly and doesn't need the API running. You
can run either one, or both.

Other scripts:

```bash
python scripts/run_rule_engine.py        # classify a sample panel and print JSON
python scripts/build_rag_index.py        # rebuild the FAISS guideline index
python scripts/train_model.py            # train, evaluate and explain the model
python scripts/run_data_prep.py          # rebuild the NHANES dataset
python scripts/evaluate_system.py        # ML, RAG and LLM evaluation
```

---

## Using the dashboard

1. Pick a source in the sidebar: **Sample patient**, **Upload file** or
   **Manual entry**.
2. Uploads can be **CSV**, **JSON** or **PDF**. The PDF reader handles the two
   common lab layouts (one row per test, and name, method and value on separate
   lines). It also reads the patient's age and sex from the report header,
   converts units, and throws away values that aren't plausible.
   **Extraction isn't perfect, so check the values in the form before analysing.**
3. Check age and sex (filled in from the PDF when found, otherwise required),
   then press **Analyze**.
4. The risk summary, severity table, charts and evidence appear straight away.
   Recommendations are generated in the background and show up when they're
   ready, so a slow or unavailable LLM doesn't hold up the page.
5. **Download PDF report** saves the assessment, including the recommendations
   if there are any.

---

## Testing

The test files run as plain scripts, no pytest needed:

```bash
make test        # runs all of them
```

Or one at a time:

```bash
python tests/test_rule_engine.py   # 19  rule engine, bands, overall severity
python tests/test_parser.py        # 18  report parsing: layouts, units, demographics
python tests/test_rag.py           #  5  splitter, corpus, retrieval
python tests/test_ml.py            #  1  model train/evaluate/explain smoke test
python tests/test_recommend.py     #  7  fusion, generation, guards
python tests/test_api.py           #  8  API endpoints (TestClient)
python tests/test_eval.py          #  5  evaluation metrics
```

That's 63 tests, and they all run offline. The LLM is replaced by a fake
provider, so no API key or running model is needed. `/recommend` is expected to
return a 502 error when no LLM can be reached.

**Evaluation.** `python scripts/evaluate_system.py` writes metrics and plots to
`reports/phase9/`: model accuracy, precision, recall, F1 and ROC-AUC, SHAP and
feature importance, the novelty-slice baselines, retrieval Precision@K and
Recall@K, and (with an LLM available) groundedness, faithfulness and
hallucination rate.

**CI.** `.github/workflows/ci.yml` runs `ruff` (warnings only) and all seven test
files on every push and pull request.

---

## Project structure

```
src/app/
  domain/        rule engine (models, enums, engine)
  rules/         loads and validates the YAML ruleset
  ml/            Random Forest: data, train, evaluate, explain, predict, registry/
  rag/           embedder, FAISS store, retriever, index/
  recommend/     fusion, LLM providers, prompts, guards
  ingestion/     biomarker catalog, unit conversion, report parser
  eval/          ML, RAG and LLM evaluation
  api/           FastAPI: controllers, services, repositories
src/data_prep/   NHANES download, merge, labels, features, pipeline
streamlit_app/   dashboard.py, pages/, ui/, services/, sample_data/
config/          clinical_rules.yaml, clinical_thresholds.yaml, nhanes_files.yaml
data/guidelines/ curated NICE/NHS/WHO passages (paraphrased, with sources)
scripts/         build_rag_index, train_model, evaluate_system, run_api, ...
tests/           test scripts
docs/            design and phase write-ups (01 to 10)
```

---

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/upload` | Parse a blood report (CSV/JSON/PDF) into biomarker values to review |
| POST | `/predict` | Rules, Random Forest and fusion: the severity assessment |
| POST | `/recommend` | The whole pipeline, returning cited recommendations |
| POST | `/retrieve` | Search the guideline passages |
| GET | `/model-info` | Ruleset, model metrics, index details, biomarker list |
| GET | `/health` | Health check |

Interactive docs are at `/docs` (Swagger) and `/redoc`.

---

## Privacy

Everything apart from the LLM step runs on your own machine. Report parsing, the
rule engine, the Random Forest and retrieval don't make any network requests.

If recommendations are generated with a cloud provider (OpenRouter or
Anthropic), the request includes the patient's age, sex and flagged biomarker
values. No name, address or ID is sent. Use Ollama if the LLM step has to stay
local too.

---

## Disclaimer

This is an educational research project. It is not for clinical use and does
not give a medical diagnosis, so always talk to a qualified healthcare
professional. Guideline content is paraphrased with its sources. The model was
trained on US NHANES data with labels based on UK guidelines, and it picks up
risk patterns rather than diagnosing disease.
