# Phase 10 — Deployment & GitHub

**Author:** Bhavesh Bhargava — MSc Advanced Data Science
**Status:** Implemented. The repository is production-ready and containerised.

> Packages the whole system (rule engine + RF + RAG + LLM, exposed via a FastAPI
> API and a Streamlit dashboard) into a reproducible, one-command deployment.

---

## 1. Artefacts created

| File | Purpose |
|---|---|
| `Dockerfile` | One image for both services; bakes in CPU torch, the embedding model, the RF model, and the FAISS index |
| `docker-compose.yml` | Runs `api` (:8000) + `dashboard` (:8502) from that image; optional `ollama` profile |
| `requirements.txt` | Pinned (floor) dependency set, grouped by phase |
| `.dockerignore` | Keeps the image lean (excludes raw/processed data, reports, git) |
| `.env.example` | Documented environment template (`.env` is git-ignored) |
| `.github/workflows/ci.yml` | CI/CD: lint → test → build & push image to GHCR |
| `Makefile` | `install / test / api / dashboard / build-index / train / eval / docker-up` |
| `README.md` | Overview + install / deploy / testing guides |
| `.gitignore` | Excludes secrets, caches, venvs |

---

## 2. Container design

- **Single image, two commands.** The API and dashboard share one image; the
  command is chosen per service in compose. Keeps builds simple and images consistent.
- **Baked artefacts.** The Dockerfile pre-downloads `all-MiniLM-L6-v2` and copies
  the committed RF model (`src/app/ml/registry`) and FAISS index
  (`src/app/rag/index`), so containers start **ready to serve** — no build-time
  training or network model pulls at first request.
- **CPU-only torch** is installed from the PyTorch CPU index before the rest, so
  the image doesn't pull the multi-GB CUDA build.
- **Config via env** (`.env`), read automatically. Nothing is required to run the
  assessment; only the LLM step needs a provider key. `OLLAMA_BASE_URL` lets a
  container reach a local/compose Ollama.
- **Healthcheck** on the API (`/health`).

---

## 3. CI/CD (GitHub Actions)

`ci.yml` on every push/PR:
1. **lint** — `ruff` (informational, non-blocking).
2. **test** — installs CPU torch + deps, runs all 7 test suites (52 tests). The
   LLM is faked, so no keys are needed; retrieval/model tests use the committed
   index and model.
3. **docker** — builds the image and, on pushes to `main`, pushes it to the GitHub
   Container Registry (`ghcr.io/<owner>/<repo>:latest`) using the built-in
   `GITHUB_TOKEN` (no extra secrets).

---

## 4. Deploy

```bash
cp .env.example .env            # add OPENROUTER_API_KEY for the LLM step (optional)
docker compose up --build -d    # dashboard :8502 · API+Swagger :8000/docs
```

Local LLM instead of OpenRouter:
```bash
docker compose --profile ollama up --build -d
docker compose exec ollama ollama pull llama3.1
# .env:  OLLAMA_BASE_URL=http://ollama:11434  → choose provider=ollama
```

See `README.md` for the full installation, deployment, and testing guides.

---

## 5. Production-readiness checklist

- [x] Containerised (Dockerfile + compose), reproducible builds
- [x] CI (lint + 52 tests) and CD (image → GHCR) on GitHub Actions
- [x] Secrets via `.env` (git-ignored); template committed; no keys in code
- [x] Health endpoint + structured logging + typed error envelopes (API)
- [x] Graceful degradation when the LLM is unreachable (assessment still served)
- [x] Documented: README, per-phase docs (01–10), Swagger API docs
- [x] Baked model + index → fast cold start, offline-capable assessment

**Phase 10 exit criteria met.** Remaining: Phase 11 (dissertation write-up).
