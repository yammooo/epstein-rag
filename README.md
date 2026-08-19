# NLP project

The notebook keeps the experimental workflow visible while reusable code is
split into three small modules:

- `documents.py`: email preparation, PDF OCR, source-aware documents, and chunking.
- `rag.py`: dense and hybrid retrieval, reranking, prompting, and local generation.
- `evaluation.py`: provider-neutral test-set generation and method comparison.

Run Jupyter from the repository root so the notebook can import these modules.

## Python dependencies

Install the common dependencies:

```bash
python -m pip install -r requirements.txt
```

`llama-cpp-python` is hardware-specific and is installed separately. For CPU:

```bash
python -m pip install llama-cpp-python
```

For CUDA 12.4:

```bash
python -m pip install --upgrade --force-reinstall 
  llama-cpp-python 
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

## PDF OCR

PDF pages are read directly with PPStructureV3 and PP-OCRv6 through the
Transformers/PyTorch backend. Layout regions restore multi-column reading
order, but only recognized text is stored; table, formula, chart, seal, and
image outputs are disabled. The first OCR run downloads the model weights.

The notebook caches raw emails, OCR pages, chunks, embeddings, and evaluation
results in `artifacts/`. Cache metadata invalidates OCR pages, chunks, and
embeddings when their relevant inputs change.
