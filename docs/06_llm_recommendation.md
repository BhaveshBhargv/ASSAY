# Phase 6 — LLM Recommendation Engine (LangChain)

**Author:** Bhavesh Bhargava — MSc Advanced Data Science
**Status:** Implemented (`src/app/recommend/`), all tests passing (7/7 offline;
real RF + real retriever integration verified).
**Stack:** LangChain (provider adapters) + Pydantic (output contract), behind clean ports.

> **Role in the pipeline (Stage 5).** This is the final stage. It fuses the rule
> engine and Random Forest into one severity, retrieves cited guideline evidence,
> and asks an LLM to turn that evidence into grounded, non-diagnostic lifestyle
> advice — every recommendation carrying its own rationale and citation.

---

## 1. Approved design decisions

| Decision | Choice | Why |
|---|---|---|
| Provider | **Provider-agnostic, local Ollama default** (Anthropic adapter alongside) | Free, offline, reproducible for the dissertation; no API key/cost; swap to hosted Claude by name. |
| Framework | **LangChain** in adapters only, behind `ILLMProvider` | Real LangChain implementation for LLM calls; domain/fusion never import it → swappable + testable. |
| Fusion | **Safety-dominant `max(rule, RF)`** + `escalated_by_rf` flag | RF escalates on the hidden-risk *pattern* (the novelty slice), never de-escalates a rule finding. |
| Output | **Pydantic `RecommendationReport`** (9 sections) | Structured, validated; each advice item = `{advice, rationale, evidence[]}`. |
| Grounding | **Citation-required parser + 2nd-pass groundedness verifier** | Enforces "only retrieved evidence" and "explain why"; yields a measurable groundedness score for Phase 9. |

---

## 2. How the four hard rules are enforced

| Rule | Mechanism |
|---|---|
| **Never diagnose** | System prompt forbids disease attribution; schema has no diagnosis field; a regex guard (`guards.scan_diagnostic`) flags second-person disease claims ("you have diabetes", "you're diabetic", "diagnosed with …") while allowing neutral biomarker facts and the "not a diagnosis" disclaimer. |
| **Always explain why** | `AdviceItem.rationale` is required; the verifier **drops** any item with an empty rationale. |
| **Only retrieved evidence** | The prompt injects ONLY the top-k retrieved passages, each tagged `[E1]…[En]`; the verifier strips citations to ids that exist and **drops** items left with none. |
| **Avoid hallucination** | Deterministic JSON parse + citation requirement + low temperature (0.2) + the groundedness verifier; an optional LLM entailment recheck adds a semantic "is this supported by Ei?" pass. |

Safety text the LLM is **not** allowed to author — the non-diagnostic disclaimer,
the flag-only **clinician signpost**, and the **urgent-referral** note — is added
by the system (`bundle.py`), so it can never be omitted or reworded by the model.

---

## 3. Architecture (Clean / SOLID)

```
demographics + biomarkers
   │  RuleEngine.evaluate            → per-biomarker status (Phase 3)
   │  RiskAdapter → RiskModel        → RF severity + probabilities (Phase 4)
   ▼  fuse()                         → ONE severity + flagged/signpost split
   │  GuidelineRetriever             → top-k cited evidence (Phase 5)
   ▼  build_messages → generator     → RecommendationReport (LLM via LangChain)
   │  guards.verify                  → non-diagnostic + groundedness pass
   ▼  RecommendationBundle           → report + system safety text + audit
```

- **Dependency Inversion:** the generator depends only on `ILLMProvider`
  (`complete(system, user) -> str`). `OllamaProvider` / `AnthropicProvider` wrap
  LangChain chat models; `FakeProvider` drives offline tests. LangChain never
  leaks into the domain.
- **Single Responsibility:** fusion fuses, prompt builds prompts, generator parses,
  guards verify, bundle renders. Each file does one job.
- **RiskAdapter** rebuilds the RF feature vector with the *same* stateless Phase-2
  transforms (`add_features` + `build_feature_matrix`), so train and inference
  agree; if the panel is too sparse for the model it degrades to rules-only.

---

## 4. Module map (`src/app/recommend/`)

| File | Responsibility |
|---|---|
| `config.py` | Provider defaults, temperature, k, groundedness threshold, disclaimer text |
| `report.py` | `AdviceItem`, `RecommendationReport` (Pydantic) + JSON format instructions |
| `ports.py` | `ILLMProvider` protocol |
| `providers.py` | `OllamaProvider`, `AnthropicProvider` (LangChain), `FakeProvider` |
| `fusion.py` | `fuse()` → `FusedAssessment` (severity + flagged/signpost) |
| `prompt.py` | System/user prompt build, evidence pack `[E1]…[En]` (LangChain `ChatPromptTemplate`) |
| `generator.py` | Call provider, robust JSON extraction, schema validation |
| `guards.py` | Non-diagnostic scan + groundedness verifier (+ optional LLM recheck) |
| `bundle.py` | Compose report + safety text + audit; `to_dict()` / `render()` |
| `engine.py` | `RecommendationEngine` orchestrator + `RiskAdapter` |
| `scripts/generate_recommendation.py` | Entrypoint (`--provider`, `--no-rf`, `--json`) |
| `tests/test_recommend.py` | Fusion, JSON extraction, guards, end-to-end (offline) |

**Run (needs `ollama pull llama3.1`):**
`python scripts/generate_recommendation.py`
**Offline demo (rules-only, no model):** inject `FakeProvider` (see tests).

---

## 5. Verification

- **7/7 offline tests pass**: safety-dominant fusion, RF escalation, noisy-JSON
  extraction, schema parse, groundedness drop of ungrounded/unexplained items,
  non-diagnostic detection, full end-to-end render.
- **Integration verified**: real Random Forest + real FAISS retriever + faked LLM
  → fusion=`serious`, 6 relevant NICE/NHS passages retrieved, groundedness 1.0,
  zero diagnostic violations.
- No regressions: rule engine 19/19, RAG 5/5, ML smoke 1/1.

**Phase 6 exit criteria met.** Phase 7 (FastAPI backend) now wraps this same
engine — see `07_fastapi_backend.md`.
