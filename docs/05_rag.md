# Phase 5 — Retrieval-Augmented Generation (RAG) Knowledge Base

**Author:** Bhavesh Bhargava — MSc Advanced Data Science
**Status:** Implemented (`src/app/rag/`), index built, all tests passing.
**Stack:** sentence-transformers (`all-MiniLM-L6-v2`) + FAISS, behind clean ports.

> **Role in the pipeline (Stage 4).** The fused severity label + flagged
> biomarkers form a query; RAG retrieves the top-k evidence-based guideline
> passages that ground the LLM's recommendations (Stage 5). Every passage carries
> a citation so no advice is ungrounded.

---

## 1. Approved design decisions

| Decision | Choice | Why |
|---|---|---|
| Corpus | **Curated cited passages + PDF parser** | Focused, legal (paraphrased summaries with attribution), reproducible; real PDFs can be dropped in too. |
| Embeddings | **all-MiniLM-L6-v2** (384-d, local) | Fast, free, offline, proven for RAG; L2-normalised → cosine via inner product. |
| Framework | **Lightweight custom** (FAISS + ports) | Minimal deps, full control, matches the SOLID ports design; no framework lock-in. |
| Index | FAISS `IndexFlatIP` (+ `IDMap2`) | Exact cosine search — ideal for a small, fixed corpus. |
| Chunking | Recursive char splitter (600/100) | Curated passages pass through whole; only long PDFs are chunked. |

---

## 2. Architecture (Clean / SOLID)

```
data/guidelines/curated_corpus.yaml   data/guidelines/pdfs/*.pdf
              │  loaders.py (document loader + PDF parser)
              ▼
        GuidelineChunk[]  (text + provenance: source, code, biomarkers, categories, severities)
              │  splitter.py (only long text)
              ▼
   IEmbedder ──► SentenceTransformerEmbedder      ← ports.py (abstractions)
              │
   IVectorStore ──► FaissVectorStore  ──►  index/guidelines.faiss + index/chunks.json
              ▲
        retriever.py  (query build → embed → search → biomarker-boost re-rank)
```

- **Dependency Inversion:** ingestion/retrieval depend on `IEmbedder` / `IVectorStore`;
  swap MiniLM→BGE or FAISS→Chroma without touching the logic.
- **Single Responsibility:** loader parses, splitter splits, embedder embeds, store
  indexes, retriever searches. Each does one job.
- **Offline vs online:** index is built once (`build_rag_index.py`); retrieval is a
  fast query-time lookup.

---

## 3. The knowledge base (`curated_corpus.yaml`)

**32 passages** across **NICE, NHS, WHO**, covering every condition the biomarker
panel flags: pre-diabetes/diabetes (NG28, PH38), lipids (CG181), fatty liver
(NG49), kidney (NG203), iron/B12/folate anaemia, vitamin D, plus general lifestyle
(activity, diet, alcohol, smoking, weight, sleep) and safety signposting.

Each passage is a **paraphrased summary with attribution** (source + guideline
code), *not* verbatim source text — legally clean and dissertation-appropriate.
Metadata per passage: `biomarkers`, `categories` (diet/physical_activity/alcohol/…),
`severities` (borderline/serious/all), enabling biomarker-aware retrieval.

**Real PDFs:** drop `SOURCE_CODE_title.pdf` files in `data/guidelines/pdfs/` and
they're parsed (pypdf), chunked, and indexed alongside the curated corpus.

---

## 4. Retrieval

`GuidelineRetriever.retrieve_for_assessment(severity, flagged, k)`:

1. **Query build** — turns the fused label + flagged biomarkers into natural
   language, e.g. *"Evidence-based lifestyle recommendations for serious
   cardiometabolic risk with: high HbA1c; low HDL cholesterol."*
2. **Embed + search** — cosine top-k over the FAISS index.
3. **Biomarker-boost re-rank** — passages tagged with the patient's flagged
   markers get a small score boost over generic matches (light hybrid retrieval).

**Live examples (built index):**

| Query | Top passages |
|---|---|
| "raised HbA1c and low HDL" | NICE NG28 (diabetes HbA1c) · PH38 (metabolic cluster) · NHS low-HDL |
| ALT-high + HbA1c-borderline (serious) | NICE NG28 · CG181 (CVD risk) · PH38 (metabolic cluster) |
| "high cholesterol diet advice" | NICE CG181 (cholesterol diet) — lipid passage top-ranked |

---

## 5. Module map (`src/app/rag/`)

| File | Responsibility |
|---|---|
| `models.py` | `GuidelineChunk`, `RetrievalResult` (+ `citation()` for provenance) |
| `ports.py` | `IEmbedder`, `IVectorStore` abstractions |
| `embedder.py` | SentenceTransformer adapter (normalised embeddings) |
| `vector_store.py` | FAISS adapter (`IndexFlatIP` + `IDMap2`, save/load) |
| `splitter.py` | Recursive character text splitter (no external dep) |
| `loaders.py` | Curated-YAML loader + **PDF parser** (pypdf) |
| `ingest.py` | Build pipeline: load → embed → index → persist |
| `retriever.py` | Query build, semantic search, biomarker-boost re-rank |
| `config.py` | Corpus / index paths |
| `scripts/build_rag_index.py` | Entrypoint (`--query` to test) |
| `tests/test_rag.py` | Splitter, loader, query, retrieval tests |

**Run:** `python scripts/build_rag_index.py --query "..."`

---

## 6. Verification

- **5/5 tests pass**: splitter bounds, corpus provenance (all NICE/NHS/WHO, all
  cited), query construction, and live retrieval (a cholesterol query returns a
  lipid passage at rank 1).
- Index: 32 chunks, 384-d, `all-MiniLM-L6-v2`, persisted to `src/app/rag/index/`.

**Phase 5 exit criteria met — ready for Phase 6 (LLM recommendation engine) on approval.**
