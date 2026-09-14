# Phase 6: LLM recommendations

**Author:** Bhavesh Bhargava, MSc Advanced Data Science
**Code:** `src/app/recommend/`, 7 offline tests, also checked with the real
Random Forest and retriever
**Stack:** LangChain for the provider calls and Pydantic for the output format

This is the last stage. The rule engine and Random Forest results are fused into
one severity, guideline evidence is retrieved, and an LLM turns that evidence
into lifestyle advice. It must not diagnose, and every recommendation needs its
own reason and citation.

---

## 1. Design decisions

| Decision | Choice | Reason |
|---|---|---|
| Providers | **Ollama** (local, the default for the API and scripts), **OpenRouter** (used by the dashboard) and **Anthropic**, all behind `ILLMProvider` | Ollama is free, works offline and gives reproducible runs with no API key. The cloud providers give faster or better output, and switching is one setting. |
| Framework | **LangChain**, only inside the provider classes | Nothing else imports LangChain, including fusion, so providers can be swapped and tests can use a fake one. |
| Fusion | **`max(rule, RF)`** with an `escalated_by_rf` flag | The model can raise the severity but never lower a rule result. The intended benefit was catching hidden risk (the novelty slice); Phase 4 §5 and Phase 9 show how that turned out. |
| Output | **Pydantic `RecommendationReport`** with 9 advice sections | Structured and validated; every advice item has `{advice, rationale, evidence[]}`. |
| Grounding | **Citations required, then a groundedness check** | Only retrieved evidence can be cited and every item has to say why. It also gives a groundedness score for Phase 9. |

---

## 2. How the four rules are enforced

| Rule | How |
|---|---|
| **Never diagnose** | The system prompt forbids naming diseases and the output schema has no diagnosis field. A regex check (`guards.scan_diagnostic`) flags statements like "you have diabetes", "you're diabetic" or "diagnosed with …", while allowing plain statements about biomarkers and the "not a diagnosis" disclaimer. |
| **Always explain why** | `AdviceItem.rationale` is required, and the check **removes** any item with an empty rationale. |
| **Only use retrieved evidence** | The prompt contains only the retrieved passages, labelled `[E1]…[En]`. The check drops citations to ids that don't exist and **removes** items left with no citation. |
| **Avoid hallucination** | JSON parsing, required citations, a low temperature (0.2) and the groundedness check. An optional second LLM call can also check whether each item is really supported by the passage it cites. |

Some safety text is never written by the LLM: the non-diagnostic disclaimer, the
"discuss with your clinician" signpost for flag-only markers, and the urgent
referral note. The app adds these itself (`bundle.py`), so the model can't leave
them out or change them.

---

## 3. Structure

```
demographics + biomarkers
   │  RuleEngine.evaluate            → status for each biomarker (Phase 3)
   │  RiskAdapter → RiskModel        → RF severity and probabilities (Phase 4)
   ▼  fuse()                         → one severity, flagged and signpost lists
   │  GuidelineRetriever             → top-k cited passages (Phase 5)
   ▼  build_messages → generator     → RecommendationReport (LLM via LangChain)
   │  guards.verify                  → diagnosis and groundedness checks
   ▼  RecommendationBundle           → report, safety text and audit
```

- The generator only depends on `ILLMProvider` (`complete(system, user) -> str`).
  `OllamaProvider`, `OpenRouterProvider` and `AnthropicProvider` wrap LangChain
  chat models, and `FakeProvider` is used in the offline tests.
- Each file has one job: fusion, prompts, parsing, checks or rendering.
- **RiskAdapter** builds the model's features with the same functions used in
  data preparation (`add_features` and `build_feature_matrix`), so training and
  prediction match. If too few biomarkers are given for the model, it falls back
  to the rules only.

---

## 4. Modules (`src/app/recommend/`)

| File | What it does |
|---|---|
| `config.py` | Default provider and model names, temperature, k, groundedness threshold, disclaimer text |
| `report.py` | `AdviceItem` and `RecommendationReport` (Pydantic), plus the JSON format instructions |
| `ports.py` | The `ILLMProvider` protocol |
| `providers.py` | `OllamaProvider`, `OpenRouterProvider`, `AnthropicProvider` (LangChain) and `FakeProvider` |
| `fusion.py` | `fuse()` → `FusedAssessment` (severity, flagged and signpost lists) |
| `prompt.py` | Builds the system and user prompts and the `[E1]…[En]` evidence list (LangChain `ChatPromptTemplate`) |
| `generator.py` | Calls the provider, extracts the JSON and validates it |
| `guards.py` | Diagnosis check and groundedness check (and the optional LLM recheck) |
| `bundle.py` | Combines the report, safety text and audit; `to_dict()` and `render()` |
| `engine.py` | `RecommendationEngine`, which runs the whole thing, and `RiskAdapter` |
| `scripts/generate_recommendation.py` | Command-line entry point (`--provider`, `--no-rf`, `--json`) |
| `tests/test_recommend.py` | Fusion, JSON extraction, checks and an offline end-to-end run |

**Run:** `python scripts/generate_recommendation.py`, which needs
`ollama pull llama3.1`, or add `--provider` to use another provider.
**Without a model:** use `FakeProvider`, as the tests do.

---

## 5. Verification

- **7 offline tests pass:** fusion taking the higher severity, RF escalation,
  pulling JSON out of messy model output, parsing against the schema, removing
  uncited and unexplained items, detecting diagnostic statements, and a full
  end-to-end render.
- **With the real components:** the real Random Forest and FAISS retriever with a
  fake LLM gave a fused severity of `serious`, 6 relevant NICE/NHS passages,
  groundedness 1.0 and no diagnostic statements.
- The other test files still pass: rule engine 19/19, RAG 5/5, ML smoke test 1/1.

Phase 7 (the FastAPI backend) uses this same engine; see `07_fastapi_backend.md`.
