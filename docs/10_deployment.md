# Phase 10 — Packaging & Local Deployment

**Author:** Bhavesh Bhargava — MSc Advanced Data Science
**Status:** Implemented. The repository runs from a clean clone with one command.

> Packages the whole system (rule engine + RF + RAG + LLM, exposed via a FastAPI
> API and a Streamlit dashboard) as a reproducible **local** Python application.
> Containerisation was deliberately dropped — see §5.

---

## 1. Artefacts

| File | Purpose |
|---|---|
| `requirements.txt` | Dependency set with version floors, grouped by phase |
| `Makefile` | `install / test / lint / api / dashboard / build-index / train / eval` |
| `.github/workflows/ci.yml` | CI: `ruff` (informational) → all seven test suites |
| `.streamlit/config.toml` | Dashboard theme, upload limit, sidebar navigation |
| `README.md` | Overview + installation, configuration, running and testing guides |
| `.gitignore` | Excludes secrets (`.env`), caches, virtual environments |

**Committed artefacts** — the reason a clean clone runs immediately:

| Artefact | Path | Built by |
|---|---|---|
| Random Forest + metadata | `src/app/ml/registry/` | `scripts/train_model.py` |
| FAISS index + chunks | `src/app/rag/index/` | `scripts/build_rag_index.py` |
| Processed NHANES dataset | `data/processed/` | `scripts/run_data_prep.py` |
| Clinical ruleset | `config/clinical_rules.yaml` | authored |

Only the `all-MiniLM-L6-v2` embedding model is fetched at first run (~90 MB, then
cached). Everything else ships with the repository.

---

## 2. Runtime design

- **Two independent entry points, one engine.** The dashboard imports the engine
  directly; the API wraps the same engine behind HTTP. Neither depends on the
  other, so either can be run alone.
- **No database, no external services.** The vector index is a file
  (`IndexFlatIP`), the model is a joblib artefact, the rules are YAML. Nothing to
  provision.
- **Config via environment**, optionally through a git-ignored `.env` loaded by
  `src/app/recommend/config.py`. Nothing is required to run the assessment; only
  the LLM step needs a provider key.
- **Graceful degradation.** A missing RF model falls back to rules-only fusion; an
  unreachable LLM still serves the full assessment, charts, evidence and PDF
  export. The API returns a typed `502` rather than a stack trace.
- **CPU-only.** No GPU is required anywhere. Installing the CPU PyTorch wheel
  first avoids pulling the multi-GB CUDA build.

---

## 3. Running it

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

streamlit run streamlit_app/dashboard.py            # dashboard  :8501
uvicorn app.api.main:app --app-dir src --reload     # API+Swagger :8000/docs
```

Rebuilding artefacts (only needed after changing data or config):

```bash
make build-index    # RAG FAISS index
make train          # Random Forest: train → evaluate → explain
make eval           # full system evaluation → reports/phase9/
```

Fully local LLM (nothing leaves the machine):

```bash
ollama pull llama3.1        # then select provider=ollama
```

See `README.md` for the complete installation, configuration and testing guides.

---

## 4. CI

`.github/workflows/ci.yml` on every push and pull request:

1. **lint** — `ruff` over `src`, `streamlit_app`, `scripts`, `tests`
   (informational, non-blocking).
2. **test** — installs the CPU PyTorch wheel plus dependencies and runs all seven
   suites (**63 tests**). No secrets are required: the LLM is faked, and the
   retrieval and model tests use the committed index and model.

---

## 5. Why not containers

An earlier revision shipped a `Dockerfile`, a `docker-compose.yml` and a GHCR
push job. These were removed. The reasoning, recorded here because it is a design
decision rather than an omission:

- **Nothing needed orchestrating.** There is no database, queue or vector service
  — the "stack" is one Python process per interface. Compose was coordinating two
  commands that each run in one line.
- **The image was mostly PyTorch.** Baking CPU torch plus the embedding model
  produced a large image whose only advantage over `pip install` was avoiding a
  one-off 90 MB download.
- **It added a second thing to keep true.** The Dockerfile, compose file and CI
  push job all restated paths, ports and environment variables that already live
  in `requirements.txt`, the `Makefile` and `config.py` — a standing source of
  documentation drift.
- **The audience runs it locally.** For an examiner reproducing a dissertation
  project, `pip install -r requirements.txt` is a shorter path than installing a
  container runtime.

Reproducibility is preserved by the things that actually carry it: pinned
dependency floors, committed model and index artefacts, a fixed random seed, and
CI that installs from scratch and runs the full suite on every push.

---

## 6. Readiness checklist

- [x] Runs from a clean clone: venv → `pip install` → one command
- [x] CI (lint + 63 tests) on GitHub Actions, no secrets required
- [x] Secrets via git-ignored `.env`; no keys in code or config
- [x] Health endpoint, structured logging, typed error envelopes (API)
- [x] Graceful degradation when the RF or the LLM is unavailable
- [x] Committed model + index → immediate start, offline-capable assessment
- [x] Documented: README, per-phase docs (01–10), Swagger API docs

**Phase 10 exit criteria met.** Remaining: Phase 11 (dissertation write-up).
