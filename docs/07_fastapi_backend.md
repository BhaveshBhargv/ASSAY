# Phase 7: FastAPI backend

**Author:** Bhavesh Bhargava, MSc Advanced Data Science
**Code:** `src/app/api/`, tested with TestClient (8 tests) and against a running server
**Run:** `python scripts/run_api.py`, then open `http://localhost:8000/docs`
(or `uvicorn app.api.main:app --app-dir src --reload`)

The API gives HTTP access to the same engine the dashboard uses. Controllers
handle HTTP, services run the engine, and repositories hold the loaded model and
index. The dashboard calls the engine directly at the moment, but it could switch
to these endpoints without any change to the engine.

---

## 1. Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/upload` | Parse a blood report (CSV/JSON/PDF) into recognised biomarkers and parse notes to review |
| `POST` | `/predict` | Rules, Random Forest and fusion: one severity plus a result for each biomarker |
| `POST` | `/recommend` | The whole pipeline: assess, retrieve, generate with the LLM, check, and return the report |
| `POST` | `/retrieve` | Search the NICE/NHS/WHO passages, by query text or by severity and flagged markers |
| `GET`  | `/model-info` | Ruleset version, model details and metrics, index details, biomarker list |
| `GET`  | `/health` | Health check |

API docs: Swagger at `/docs`, ReDoc at `/redoc`, and the schema at `/openapi.json`.

---

## 2. Structure

```
src/app/api/
  main.py                  creates the app: startup, middleware, CORS, error handlers, routers
  dependencies.py          wiring (cached repositories, services created per request)
  core/
    config.py              settings (environment variables starting with ASSAY_)
    logging.py             request id and RequestContextMiddleware
    errors.py              AppError types and the JSON error handlers
  schemas/                 Pydantic request and response models, where validation happens
    common.py  predict.py  recommend.py  retrieve.py  upload.py  model_info.py
  repositories/            hold the loaded artefacts
    rule_repository.py       RuleEngine and ruleset
    model_repository.py      Random Forest (RiskAdapter) and its metadata
    retrieval_repository.py  FAISS GuidelineRetriever
    llm_repository.py        creates LLM providers
  services/                the logic behind each endpoint
    assessment_service.py  recommendation_service.py  retrieval_service.py
    upload_service.py      model_info_service.py      mappers.py
  controllers/             one FastAPI router per resource
```

- Controllers only depend on service getters such as
  `Depends(get_assessment_service)`. Services get their repositories through the
  constructor, and the repositories are cached with `lru_cache`, so the Random
  Forest, FAISS index and embedding model are loaded once per process.
  `warm_up()` loads them at startup so the first request isn't slow.
- Controllers deal with HTTP, services run the engine, repositories load
  artefacts, and `mappers.py` turns engine objects into response models. The
  HTTP layer never needs to know how a model is loaded.

---

## 3. Validation, logging and errors

**Validation.** The request models use Pydantic v2. `Demographics` checks age,
sex and the NHANES codes. `validate_biomarkers` rejects unknown biomarker codes
and values that aren't positive numbers, with a clear message. `/retrieve` needs
either a query or a severity and/or flagged markers. Invalid requests get a
**422** with the list of errors.

**Logging.** Every request gets a `request_id`. `RequestContextMiddleware` sets
it, logs `METHOD path -> status (Nms)`, and returns it in the `X-Request-ID`
header so a client can match a response to the server logs.

**Errors.** All errors use the same JSON shape:
```json
{"error": {"type": "llm_unavailable", "message": "...", "detail": {...}}, "request_id": "…"}
```
Each error type has a status code: `validation_error` and `parsing_error` → 422,
`payload_too_large` → 413, `llm_unavailable` → 502, `model_unavailable` → 503.
Any other exception becomes a 500. It's logged with the request id, and no
internal details are returned. When no LLM is running, `/recommend` returns a
**502** rather than failing.

**Swagger.** The OpenAPI docs include a description, version, tags, a summary
and description for each route, request and response examples, and the error
responses (422, 502 and so on). The non-diagnostic disclaimer is part of the API
description and is also returned by `/model-info` and `/recommend`.

**CORS.** The Streamlit ports (`:8501`, `:8502`) are allowed by default. Set
`ASSAY_CORS_ORIGINS` to change this.

---

## 4. Safety

The API keeps the same safeguards as the engine. `/predict` and `/recommend`
never diagnose. Flag-only markers are listed under `signpost` for discussion
with a clinician rather than turned into advice. `urgent_referral` and
`urgent_note` mark results that need prompt attention. Every recommendation
cites evidence, and the response includes the `groundedness` audit and the
disclaimer.

---

## 5. Verification

- **`tests/test_api.py`**, 8 tests using TestClient, offline: health and
  model-info, a valid `/predict`, the 422 error for invalid input, `/retrieve` by
  severity and flags, the 422 when `/retrieve` gets nothing to search with, JSON
  upload, the 422 for an unsupported file type, and `/recommend` returning **502**
  without an LLM.
- **Running server** (uvicorn): all six paths appear in `/openapi.json`;
  `/predict` returns `serious` for the sample patient; `/model-info` reports
  ruleset 1.1, model macro-F1 0.868, 32 indexed passages with all-MiniLM-L6-v2,
  and 27 biomarkers; `/recommend` returns 502 `llm_unavailable` with an
  `X-Request-ID` header.
- All 63 tests pass. The biomarker catalog and report parser were moved into
  `app.ingestion` so the API and the dashboard share the same code.
