# Phase 7 — FastAPI Backend ("Assay API")

**Author:** Bhavesh Bhargava — MSc Advanced Data Science
**Status:** Implemented (`src/app/api/`), verified live and via TestClient (8/8).
**Run:** `python scripts/run_api.py` → Swagger at `http://localhost:8000/docs`
(or `uvicorn app.api.main:app --app-dir src --reload`).

> **Role in the project.** The HTTP interface to the Phase 3-6 engine. It is a thin
> interface-adapter: controllers handle HTTP, services orchestrate the engine,
> repositories own the loaded artefacts. The Streamlit dashboard could later call
> these endpoints instead of the engine directly, with no change to the engine.

---

## 1. Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/upload` | Parse a blood report (CSV/JSON/PDF) → recognised biomarkers + notes for review |
| `POST` | `/predict` | Rules + Random Forest + fusion → one severity + per-biomarker results |
| `POST` | `/recommend` | Full pipeline: assess → retrieve → LLM → verify → grounded report |
| `POST` | `/retrieve` | Semantic search over NICE/NHS/WHO passages (query, or severity + flags) |
| `GET`  | `/model-info` | Ruleset version, RF details/metrics, RAG index, biomarker catalog |
| `GET`  | `/health` | Liveness/readiness probe |

Interactive docs: **Swagger `/docs`**, ReDoc `/redoc`, schema `/openapi.json`.

---

## 2. Architecture (controllers → services → repositories)

```
src/app/api/
  main.py                 app factory: lifespan warm-up, middleware, CORS, handlers, routers
  dependencies.py         DI wiring (repositories = lru_cache singletons; services per request)
  core/
    config.py             settings (env prefix ASSAY_)
    logging.py            request-id context + RequestContextMiddleware
    errors.py             typed AppError hierarchy + JSON exception handlers
  schemas/                Pydantic request/response models (validation lives here)
    common.py  predict.py  recommend.py  retrieve.py  upload.py  model_info.py
  repositories/           own the loaded artefacts (one responsibility each)
    rule_repository.py    RuleEngine + ruleset
    model_repository.py   Random Forest (RiskAdapter) + metadata
    retrieval_repository.py  FAISS GuidelineRetriever
    llm_repository.py     builds LLM providers (behind the port)
  services/               business logic / orchestration
    assessment_service.py  recommendation_service.py  retrieval_service.py
    upload_service.py      model_info_service.py      mappers.py
  controllers/            FastAPI routers (one per resource)
```

- **Dependency injection.** Controllers depend only on service getters
  (`Depends(get_assessment_service)`). Services receive repositories by
  constructor injection. Repositories are process-singletons via `lru_cache`, so
  the heavy artefacts (RF, FAISS index, sentence-transformer) load **once**;
  `warm_up()` primes them in the lifespan startup so the first request isn't slow.
- **Separation of concerns.** Controllers = HTTP only; services = orchestration;
  repositories = artefact loading; `mappers.py` = domain → wire format. Nothing in
  the HTTP layer knows how a model is loaded.

---

## 3. Cross-cutting concerns

**Validation** — Pydantic v2 models. `Demographics` bounds age/sex/codes; the
shared `validate_biomarkers` rejects unknown codes and non-positive values with a
clear message; `/retrieve` requires a query or a severity/flags signal. Failures
return **422** with the structured error list.

**Logging** — structured logs carry a per-request `request_id` (context var). The
`RequestContextMiddleware` assigns it, logs `METHOD path -> status (Nms)`, and
echoes it in the `X-Request-ID` response header for client/server correlation.

**Error handling** — one JSON envelope everywhere:
```json
{"error": {"type": "llm_unavailable", "message": "...", "detail": {...}}, "request_id": "…"}
```
Typed errors map to status codes: `validation_error`/`parsing_error` → 422,
`payload_too_large` → 413, `llm_unavailable` → 502, `model_unavailable` → 503,
and any unhandled exception → 500 (logged with the request_id, never leaked).
Notably, `/recommend` with no LLM running returns a clean **502**, not a crash.

**Swagger** — rich OpenAPI: title/description/version, tag groups, per-route
summaries and descriptions, request/response examples, and documented error
responses (422/502/…). The non-diagnostic disclaimer is stated in the API
description and returned by `/model-info` and `/recommend`.

**CORS** — allows the Streamlit origins (`:8501`, `:8502`) by default
(`ASSAY_CORS_ORIGINS` to override).

---

## 4. Safety carried through the API

Same guarantees as the engine, surfaced over HTTP: `/predict` and `/recommend`
never diagnose; flag-only markers appear under `signpost` (→ clinician), not
advice; `urgent_referral` / `urgent_note` flag prompt-attention cases; every
recommendation cites evidence and the response includes the `groundedness` audit
and the disclaimer.

---

## 5. Verification

- **8/8 API tests** (`tests/test_api.py`, FastAPI TestClient, offline): health &
  model-info, predict OK, validation → 422 envelope, retrieve by severity+flags,
  retrieve requires-a-signal → 422, upload JSON, upload unsupported → 422, and
  `/recommend` degrading gracefully to a typed **502** without an LLM.
- **Live server** (uvicorn): all six paths in `/openapi.json`; `/predict` →
  `serious`; `/model-info` → ruleset 1.1, RF macro-F1 0.868, RAG 32/all-MiniLM-L6-v2,
  27 biomarkers; `/recommend` → 502 `llm_unavailable` with `X-Request-ID`.
- No regressions: **63/63 tests** across all phases. The shared biomarker
  catalog/parser were lifted into `app.ingestion` (used by both the API and the
  dashboard); Streamlit still imports and runs.

**Phase 7 exit criteria met.** Remaining: Phase 11 (dissertation write-up).
