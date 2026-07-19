# Phase 3 — Clinical Rule Engine

**Author:** Bhavesh Bhargava — MSc Advanced Data Science
**Status:** Implemented (`src/app/`), 19/19 unit tests passing.
**Guideline basis:** NICE (NG28, NG136, CG181, NG49, NG203), NHS, WHO, NCEP ATP III, IDF, standard clinical reference ranges.
**Panel:** 27 blood biomarkers — 21 actionable (drive label + recommendations), 6 flag-only.
BP/BMI/waist are **out of scope** (not blood tests; unavailable from a report at inference).

> Deterministic path of the five-stage contract: **Rules → RFC → Fusion → RAG → LLM**.
> The engine holds *no* clinical constants in code — every threshold, status,
> interpretation and citation lives in `config/clinical_rules.yaml`.

---

## 1. Approved design decisions

| Decision | Choice | Rationale |
|---|---|---|
| **Severity model** | Per-biomarker `status ∈ {normal, low, high, borderline, severe}` (your 5 categories), orthogonal to a fusion `severity ∈ {normal, borderline, serious}` and an `urgent` flag | `low`/`high` capture *direction*; `borderline`/`severe` capture *magnitude*. Keeping status and fusion-severity separate lets a marker be clinically "high" yet only a borderline risk contributor (e.g. waist). |
| **Config source** | **Unified** — one `clinical_rules.yaml` drives both the engine and Phase-2 Layer-1 labelling | Rules and training labels can never drift (SOLID single source of truth). Verified by re-running data prep. |
| **Severe handling** | Any `severe` band raises `summary.urgent_referral = true` | Safety-dominant escalation (Phase 1). Configurable per band via `urgent:`. |
| **Extensibility** | New biomarker = new YAML block, **zero Python changes** | Open/Closed principle. |

---

## 2. The five statuses (per biomarker)

| Status | Meaning | Example |
|---|---|---|
| `normal` | within reference range | HbA1c 5.2% |
| `borderline` | mild deviation, either direction | HbA1c 6.0% (pre-diabetes); SBP 130 |
| `low` | below range (clinically low) | HDL < sex threshold; SBP < 90 |
| `high` | above range (clinically high) | LDL 165; HbA1c 7.1% |
| `severe` | markedly abnormal → urgent | SBP ≥ 180; triglycerides ≥ 500; HbA1c ≥ 9 |

Each maps to a fusion `severity` and an `urgent` flag **per band in config**, so the
clinical meaning is fully data-driven and auditable.

---

## 3. How the config works (`clinical_rules.yaml`)

Each biomarker declares ordered `bands`. A value falls in the first band whose
`[min, max)` contains it (min inclusive, max exclusive; omit for ±∞). `min`/`max`
may be a scalar **or** a sex map `{male, female}` — resolved at evaluation time.

```yaml
hdl_mgdl:
  name: HDL cholesterol
  unit: " mg/dL"
  reference_range: {low: {male: 40, female: 50}, high: 200}
  bands:
    - {status: severe, max: 30, severity: serious, direction: low, urgent: true,
       guideline: {source: NICE, code: CG181},
       interpretation: "{name} {value}{unit} is very low (<30); a strong ... risk factor."}
    - {status: low, min: 30, max: {male: 40, female: 50}, severity: borderline, direction: low, ...}
    - {status: normal, min: {male: 40, female: 50}, severity: normal, direction: in_range, ...}
```

**Load-time validation (fail fast):** scalar bands must be ascending, gap-free, and
cover the whole number line; unknown status/severity values are rejected. A
misconfigured YAML fails at startup, not silently at inference.

**Biomarkers covered (27, blood-only panel).** Each carries `category` (always
`blood`), `tier`, and `label_role`:
- **Panels:** cardiometabolic (HbA1c, glucose, lipids), CBC (Hb, Hct, RBC, MCV,
  MCH, MCHC, RDW, WBC, platelets), liver (ALT, AST, ALP, albumin, bilirubin),
  kidney (creatinine, BUN), electrolytes (Na, K, Cl, Ca), Vitamin D.
- **Out of scope:** blood pressure, BMI, waist — not blood tests, and unavailable
  from a blood report at inference. Weight & BP are signposted to the GP instead.
- **`tier`** = `actionable` (21 markers → drive recommendations + label) vs
  `flag_only` (6: electrolytes, WBC, platelets → reported + escalated, but NEVER
  turned into lifestyle advice; routed to a clinician signpost instead).
- **`label_role`** = `core` vs `secondary` (see §3b). Unit tests lock the
  category/tier/aggregation behaviour.

### 3b. Weighted-core label aggregation

With 25 actionable markers, a naive "any borderline → borderline" rule made the
`normal` class vanish (a lone raised RDW or Vitamin-D insufficiency flagged
everyone). The aggregation policy (config `meta`, so it's tunable) is:

- any actionable **serious** marker → **serious**;
- a **core** marker (cardiometabolic + measurements) at borderline → **borderline**;
- **secondary** markers (CBC/liver/kidney/Vit-D) → borderline only when
  `>= secondary_borderline_min` (default 2) are mildly abnormal;
- else **normal**.

`RuleEngineResult.overall_severity` and the Phase-2 Layer-1 label call this **one**
policy, so training labels and inference severity cannot drift. Flag-only markers
never affect the graded label — they only raise `urgent_referral`.

---

## 4. Architecture (Clean / SOLID)

```
config/clinical_rules.yaml         ← all clinical knowledge (data)
        │  load + validate
        ▼
src/app/rules/loader.py            → RuleSet
        │  injected into
        ▼
src/app/domain/services/rule_engine.py   (domain service; no I/O, no constants)
        │  uses
        ▼
src/app/domain/models.py + enums.py       (pure entities/value objects)
```

- **Dependency Inversion:** the engine depends on the `RuleSet` abstraction; the
  YAML/loader is injected. Swap the config, swap the clinical policy.
- **Single Responsibility:** the engine *only* classifies. It does not fuse, retrieve,
  or explain.
- **No leakage of frameworks into the domain:** `domain/` imports nothing external.

---

## 5. Standardized JSON output

`engine.evaluate(readings, context).to_dict()`:

```json
{
  "ruleset_version": "1.0",
  "context": {"age": 54, "sex": "male"},
  "biomarkers": [
    {
      "code": "hba1c_pct", "name": "HbA1c", "value": 6.8, "unit": "%",
      "status": "high", "direction": "high", "severity": "serious", "urgent": false,
      "reference_range": {"low": 4.0, "high": 5.6},
      "interpretation": "HbA1c 6.8% is in the diabetes range (>=6.5%).",
      "guideline": {"source": "NICE", "code": "NG28"}
    }
  ],
  "summary": {
    "overall_severity": "serious",
    "urgent_referral": true,
    "flagged": ["hba1c_pct", "sbp_mmhg", "..."],
    "status_counts": {"high": 4, "borderline": 4, "low": 1, "severe": 1},
    "unknown_codes": []
  }
}
```

`overall_severity` is the safety-dominant max across biomarkers — **this is the value
the Fusion stage (Phase 3→Stage 3 of the contract) consumes**, together with
`urgent_referral`.

---

## 6. Module map (`src/app/`)

| File | Responsibility |
|---|---|
| `domain/enums.py` | `Status`, `Severity`, `Direction`, severity ranking + `max_severity` |
| `domain/models.py` | `Band`, `BiomarkerRule`, `RuleSet`, `BiomarkerResult`, `RuleEngineResult`, sex-specific threshold resolution |
| `domain/services/rule_engine.py` | The engine: `classify`, `evaluate`, `layer1_severity` |
| `rules/loader.py` | Parse + validate `clinical_rules.yaml` → `RuleSet` |
| `scripts/run_rule_engine.py` | CLI/demo emitting the standardized JSON |
| `tests/test_rule_engine.py` | 12 unit tests |

---

## 7. Unification result (rules ↔ training labels)

Phase-2 Layer-1 labelling now calls `engine.layer1_severity(...)`. Re-running data
prep confirmed stability:

| Label | Original inline logic | Unified via engine |
|---|---|---|
| serious | 12,752 | 12,753 |
| borderline | 7,016 | 7,577 |
| normal | 2,030 | 1,468 |

`serious` is unchanged; ~560 records moved normal→borderline because the engine adds
clinically legitimate mild flags the old logic lacked (hypotension, hypoglycaemia,
increased-risk waist tier). This is a correctness gain, and rules/labels are now one
definition.

---

## 8. Verification

- **12/12 unit tests pass**, covering: config load + contiguity validation, each
  status band, boundary inclusivity (6.5 → diabetes, 6.49 → pre-diabetes),
  sex-specific HDL (45 mg/dL: normal for men, low for women), low-direction BP,
  missing/NaN → `None`, `evaluate` summary aggregation, unknown-code handling, and
  JSON-serialisability.
- **Demo** (`run_rule_engine.py`) produces valid standardized JSON; a severe SBP (182)
  correctly sets `urgent_referral: true`.

**Phase 3 exit criteria met — ready for Phase 4 (Random Forest model) on approval.**
