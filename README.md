# NLP Project — Retrieval-Augmented Generation (RAG) on Epstein Files

**Group Members:**
- Casucci Leonardo (Student ID: 2196383)
- Frigo Gianmaria (Student ID: 2196376)
- **Master Program:** Computer Engineering

---

An end-to-end, source-grounded Retrieval-Augmented Generation (RAG) system operating over heterogeneous legal disclosures and court filings (raw email datasets and scanned multi-column PDF deposition transcripts).

The repository maintains an interactive, reproducible experimental workflow in [`PART2.ipynb`](PART2.ipynb) while encapsulating reusable core logic across three decoupled Python modules:

- **`documents.py`**: Raw email ingestion, NFKC normalization, content-based email deduplication, layout-aware PDF OCR (`PPStructureV3`), unified document schema consolidation, and token-budget-aware document chunking with provenance headers.
- **`rag.py`**: Dense vector indexing (`BAAI/bge-base-en-v1.5` with FAISS inner-product), sparse lexical retrieval (TF-IDF), candidate cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L6-v2`), adjacent chunk expansion (`_add_neighbors`), prompt assembly, and local quantized LLM inference (`gemma-4-E4B-it-heretic-GGUF` via `llama-cpp-python`).
- **`evaluation.py`**: Provider-neutral automated test-set generation (equal PDF/email split), retrieval ranking evaluation (Hit@K / Recall@K), and LLM-as-a-judge answer scoring for accuracy, faithfulness, and citation validity via OpenRouter / Google GenAI.

Run Jupyter from the repository root so the notebook can import these modules:

```bash
jupyter lab PART2.ipynb
```

> **Cached run:** All `FORCE_*` flags default to `False`, so valid committed
> artifacts are loaded. Enable only the stage you want to regenerate; test-set
> or evaluation regeneration requires an OpenRouter or Gemini API key.

---

## Python dependencies

Install the core dependencies:

```bash
python -m pip install -r requirements.txt
```

`llama-cpp-python` is hardware-specific and is installed separately.

For CPU:

```bash
python -m pip install llama-cpp-python
```

For CUDA 12.4+:

```bash
python -m pip install --upgrade --force-reinstall \
  llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

### Environment Configuration

Optional API keys for LLM inference and evaluation can be configured in a `.env` file at the repository root:

```env
HF_TOKEN=your_huggingface_token
OPENROUTER_API_KEY=your_openrouter_key
GOOGLE_API_KEY=your_google_ai_key
```

---

## PDF OCR & Ingestion Pipeline

PDF documents in `artifacts/epstein_files_pdfs/` are processed with **PPStructureV3** and **PP-OCRv6** through the PyTorch/Transformers backend:
- Layout region detection groups recognized text blocks to restore natural multi-column reading order.
- Table, formula, chart, seal, and image recognition are disabled to optimize throughput.
- Each recognized page is assigned a stable provenance identifier (`pdf:filename:p{page}`).
- Email records undergo content-based deduplication while aggregating and preserving all original document IDs for full citation traceability.

### Token-Budget-Aware Chunking

Documents are split using Hugging Face tokenizers with a strict token ceiling (512 tokens). Each chunk prepends a structured metadata header (document title, type, sender, recipients, date/page, and source IDs) and computes a dynamic token budget for body text to guarantee zero token truncation during embedding computation.

---

## Retrieval, Generation & Evaluation

The system compares two retrieval strategies on the same benchmark questions:

1. **Dense Baseline**: Embeds queries with `bge-base-en-v1.5` and performs maximum inner-product search over L2-normalized vectors in a FAISS index.
2. **Hybrid Retrieve-Rerank-Expand**:
   - Retrieves Top-20 dense and Top-20 TF-IDF lexical candidates.
   - Scores the candidate union with a Cross-Encoder reranker (`ms-marco-MiniLM-L6-v2`).
   - Retains the Top-5 reranked anchor chunks and dynamically expands context by adding adjacent ($\pm 1$) chunks from the same source document.

Answers are generated locally using **Gemma 4-E4B-it** (GGUF Q4_K_M) with strict citation constraints. A 16-item balanced test set (8 PDF, 8 email) is evaluated using LLM-as-a-judge scoring:
- **Retrieval Recall@5**: Dense `60.0%` vs. Hybrid `93.3%`
- **Answer Generation Score (1–10)**: Dense `6.125` vs. Hybrid `8.562`

---

## Caching & Artifacts

All intermediate data artifacts are cached in `artifacts/` to enable fast, deterministic, and offline execution:
- `raw_dataset.parquet`: Raw downloaded email records.
- `ocr_pages.parquet`: Layout-ordered OCR extracted PDF pages.
- `chunks_df.parquet`: Token-budgeted text chunks with headers.
- `embeddings.npy`: Precomputed 768-dimensional dense embeddings.
- `micro_test_set.json`: Ground-truth benchmark Q/A pairs.
- `evaluation_results.json` & `evaluation_results_hybrid.json`: Evaluation judgments and scores.

Cache integrity is maintained via `.meta.json` sidecar files containing SHA-256 fingerprints, automatically invalidating stale caches when source data or hyperparameters change.