# NLP project

The notebook keeps the experimental workflow visible while reusable code is
split into three small modules:

- `documents.py`: email preparation, PDF OCR, source-aware documents, and chunking.
- `rag.py`: embeddings, FAISS retrieval, prompt construction, and local generation.
- `evaluation.py`: Gemini test-set generation and evaluation.

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

## WSL dependencies for PDF OCR

PDF OCR also needs Tesseract and Poppler inside WSL (a Windows installation is
not enough for a WSL notebook kernel):

```bash
sudo apt update
sudo apt install -y tesseract-ocr poppler-utils
```

The notebook caches raw emails, OCR pages, chunks, embeddings, and evaluation
results in `artifacts/`. Cache metadata invalidates OCR pages, chunks, and
embeddings when their relevant inputs change.
