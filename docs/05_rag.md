# Phase 5: Guideline retrieval (RAG)

**Author:** Bhavesh Bhargava, MSc Advanced Data Science
**Code:** `src/app/rag/`, with the index built and committed
**Stack:** sentence-transformers (`all-MiniLM-L6-v2`) and FAISS

This is stage 4 of the pipeline. The fused severity and the flagged biomarkers
are turned into a query, and the best-matching guideline passages are retrieved
for the LLM to write its recommendations from (stage 5). Every passage has a
citation, so every piece of advice can point back to its source.

---

## 1. Design decisions

| Decision | Choice | Reason |
|---|---|---|
| Corpus | **Curated passages with citations, plus a PDF loader** | Short paraphrased summaries with the source named avoid copying guideline text and keep results reproducible. Real guideline PDFs can be added as well. |
| Embeddings | **all-MiniLM-L6-v2** (384 dimensions, runs locally) | Fast, free, works offline and is widely used for retrieval. The vectors are normalised, so the inner product is the cosine similarity. |
| Framework | **Small custom code on FAISS** | Few dependencies and full control. A corpus this size doesn't need a RAG framework. |
| Index | FAISS `IndexFlatIP` wrapped in `IDMap2` | Exact cosine search, which is fine for a small fixed corpus. |
| Chunking | Recursive character splitter (600 characters, 100 overlap) | The curated passages are short enough to stay whole; only long PDFs get split. |

---

## 2. Structure

```
data/guidelines/curated_corpus.yaml   data/guidelines/pdfs/*.pdf
              │  loaders.py (YAML loader and PDF reader)
              ▼
        GuidelineChunk[]  (text plus source, code, biomarkers, categories, severities)
              │  splitter.py (long text only)
              ▼
   IEmbedder ──► SentenceTransformerEmbedder      ← ports.py (interfaces)
              │
   IVectorStore ──► FaissVectorStore  ──►  index/guidelines.faiss + index/chunks.json
              ▲
        retriever.py  (build query → embed → search → boost flagged biomarkers)
```

- Ingestion and retrieval depend on the `IEmbedder` and `IVectorStore`
  interfaces, so MiniLM could be replaced with BGE, or FAISS with Chroma, without
  changing that code.
- Each file does one thing: loading, splitting, embedding, indexing or searching.
- The index is built once with `build_rag_index.py`, and retrieval at request
  time is a quick lookup.

---

## 3. The knowledge base (`curated_corpus.yaml`)

**32 passages** from **NICE, NHS and WHO**, covering what the biomarker panel can
flag: prediabetes and diabetes (NG28, PH38), lipids (CG181), fatty liver (NG49),
kidney disease (NG203), iron, B12 and folate deficiency anaemia, and vitamin D,
plus general lifestyle topics (activity, diet, alcohol, smoking, weight, sleep)
and safety signposting.

Each passage is a **paraphrased summary with its source and guideline code**,
not text copied from the original document. Each also lists `biomarkers`,
`categories` (diet, physical_activity, alcohol, …) and `severities` (borderline,
serious or all), which retrieval uses to favour passages about the patient's
flagged markers.

**Guideline PDFs:** put files named `SOURCE_CODE_title.pdf` in
`data/guidelines/pdfs/`. They're read with pypdf, split and indexed along with
the curated passages.

---

## 4. Retrieval

`GuidelineRetriever.retrieve_for_assessment(severity, flagged, k)`:

1. **Build a query** from the severity and flagged biomarkers. For a serious
   result with high HbA1c and low HDL the query is *"Lifestyle guidance for high
   HbA1c; low HDL cholesterol. Evidence-based advice for serious health risk."*
2. **Embed and search** for the top-k passages by cosine similarity.
3. **Re-rank:** passages tagged with one of the patient's flagged biomarkers get
   a small score boost over general matches.

For example, the query "high cholesterol diet advice" returns a lipid passage
first (this is one of the tests). Retrieval quality across a set of test queries
is measured in Phase 9.

---

## 5. Modules (`src/app/rag/`)

| File | What it does |
|---|---|
| `models.py` | `GuidelineChunk` and `RetrievalResult`, with `citation()` |
| `ports.py` | The `IEmbedder` and `IVectorStore` interfaces |
| `embedder.py` | SentenceTransformer embedder (normalised vectors) |
| `vector_store.py` | FAISS store (`IndexFlatIP` and `IDMap2`, save and load) |
| `splitter.py` | Recursive character text splitter, no extra dependency |
| `loaders.py` | Loads the curated YAML and reads PDFs (pypdf) |
| `ingest.py` | Builds the index: load, embed, index, save |
| `retriever.py` | Builds queries, searches and re-ranks |
| `config.py` | Corpus and index paths |
| `scripts/build_rag_index.py` | Builds the index (`--query` runs a test search) |
| `tests/test_rag.py` | Splitter, loader, query and retrieval tests |

**Run:** `python scripts/build_rag_index.py --query "..."`

---

## 6. Verification

- **5 tests pass:** splitter size limits, the corpus (every passage is from NICE,
  NHS or WHO and has a citation), query building, and retrieval (a cholesterol
  query returns a lipid passage first).
- Index: 32 chunks, 384 dimensions, `all-MiniLM-L6-v2`, saved in
  `src/app/rag/index/`.
