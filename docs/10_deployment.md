# Phase 10: Packaging and running locally

**Author:** Bhavesh Bhargava, MSc Advanced Data Science

The whole system (rule engine, Random Forest, retrieval and LLM, with the API
and the dashboard) runs as a local Python application from a fresh clone. It
doesn't use containers; §5 explains why.

---

## 1. Files

| File | Purpose |
|---|---|
| `requirements.txt` | Dependencies with minimum versions, grouped by part of the system |
| `Makefile` | `install / test / lint / api / dashboard / build-index / train / eval` |
| `.github/workflows/ci.yml` | CI: `ruff` (warnings only), then all seven test files |
| `.streamlit/config.toml` | Dashboard theme, upload limit, sidebar navigation |
| `README.md` | Overview, installation, configuration, running and testing |
| `.gitignore` | Keeps `.env`, caches and virtual environments out of git |

**Committed artefacts**, which are why a fresh clone runs straight away:

| Artefact | Path | Built by |
|---|---|---|
| Random Forest and its metadata | `src/app/ml/registry/` | `scripts/train_model.py` |
| FAISS index and passages | `src/app/rag/index/` | `scripts/build_rag_index.py` |
| Processed NHANES dataset | `data/processed/` | `scripts/run_data_prep.py` |
| Clinical ruleset | `config/clinical_rules.yaml` | written by hand |

The only download on first run is the `all-MiniLM-L6-v2` embedding model (about
90 MB), which is then cached.

---

## 2. How it runs

- **Two entry points, one engine.** The dashboard imports the engine directly
  and the API wraps it in HTTP. Neither needs the other, so either can run alone.
- **No database or outside services.** The index is a file (`IndexFlatIP`), the
  model is a joblib file and the rules are YAML, so there's nothing to set up.
- **Configuration from the environment**, optionally through a git-ignored `.env`
  file loaded by `src/app/recommend/config.py`. The assessment doesn't need any
  settings; only the LLM step needs a provider key.
- **Fallbacks.** Without the Random Forest, fusion uses the rules only. Without an
  LLM, the assessment, charts, evidence and PDF export all still work, and the
  API returns a `502` error rather than a stack trace.
- **CPU only.** No GPU is needed. Installing the CPU build of PyTorch first avoids
  downloading the multi-GB CUDA version.

---

## 3. Running it

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

streamlit run streamlit_app/dashboard.py            # dashboard  :8501
uvicorn app.api.main:app --app-dir src --reload     # API+Swagger :8000/docs
```

Rebuilding the artefacts (only needed after changing the data or config):

```bash
make build-index    # FAISS guideline index
make train          # Random Forest: train, evaluate, explain
make eval           # full evaluation into reports/phase9/
```

Keeping the LLM step local as well:

```bash
ollama pull llama3.1        # then use provider=ollama
```

The README has the full installation, configuration and testing instructions.

---

## 4. CI

`.github/workflows/ci.yml` runs on every push and pull request:

1. **lint:** `ruff` over `src`, `streamlit_app`, `scripts` and `tests`. It only
   reports problems and never fails the build.
2. **test:** installs the CPU build of PyTorch and the dependencies, then runs
   all seven test files (**63 tests**). No secrets are needed, since the LLM is
   faked and the retrieval and model tests use the committed index and model.

---

## 5. Why not containers

An earlier version had a `Dockerfile`, a `docker-compose.yml` and a CI job that
pushed an image to GHCR. They were removed for these reasons:

- **There was nothing to orchestrate.** With no database, queue or vector
  service, the system is one Python process per interface. Compose was only
  starting two commands that are one line each.
- **The image was mostly PyTorch.** Including CPU PyTorch and the embedding model
  made a large image, and all it saved over `pip install` was a one-off 90 MB
  download.
- **It was more to keep in sync.** The Dockerfile, the compose file and the CI job
  repeated paths, ports and environment variables already defined in
  `requirements.txt`, the `Makefile` and `config.py`, so they kept drifting out
  of date.
- **It's run locally.** For someone reproducing a dissertation project,
  `pip install -r requirements.txt` is quicker than installing a container
  runtime first.

Reproducibility comes from the minimum dependency versions, the committed model
and index, a fixed random seed, and CI that installs everything from scratch and
runs the full test suite on every push.

---

## 6. Checklist

- [x] Runs from a fresh clone: create a venv, `pip install`, run one command
- [x] CI (lint and 63 tests) on GitHub Actions, with no secrets needed
- [x] Secrets in a git-ignored `.env`, never in code or config
- [x] Health endpoint, request logging and consistent error responses in the API
- [x] Keeps working when the Random Forest or the LLM isn't available
- [x] Committed model and index, so the assessment starts immediately and works offline
- [x] Documentation: README, phase write-ups (01 to 10), Swagger API docs
