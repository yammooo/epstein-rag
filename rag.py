"""Embedding, retrieval, prompting, and local generation helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_embedding_model(model_id: str, device: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_id, device=device)


def compute_embeddings(chunks: pd.DataFrame, embedding_model, batch_size: int = 64) -> np.ndarray:
    embeddings = embedding_model.encode(
        chunks["text"].tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return embeddings.astype("float32")


def load_or_compute_embeddings(
    chunks: pd.DataFrame,
    embedding_model,
    cache_path: Path,
    model_id: str,
    force: bool = False,
) -> np.ndarray:
    """Load embeddings only when they match the current chunks and model."""
    fingerprint = _chunks_fingerprint(chunks, model_id)
    meta_path = cache_path.with_suffix(cache_path.suffix + ".meta.json")

    if not force and cache_path.exists() and meta_path.exists():
        with meta_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("fingerprint") == fingerprint:
            embeddings = np.load(cache_path)
            if embeddings.shape[0] == len(chunks):
                return embeddings.astype("float32")

    embeddings = compute_embeddings(chunks, embedding_model)
    np.save(cache_path, embeddings)
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump({"fingerprint": fingerprint}, handle)
    return embeddings


def build_index(embeddings: np.ndarray):
    import faiss

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype("float32"))
    return index


def retrieve(
    query: str,
    embedding_model,
    index,
    chunks: pd.DataFrame,
    k: int = 5,
) -> pd.DataFrame:
    """Return the k highest-scoring source chunks for a query."""
    if chunks.empty:
        raise ValueError("Cannot retrieve from an empty chunk collection.")

    k = min(k, len(chunks))
    query_embedding = embedding_model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(query_embedding, k)

    results = chunks.iloc[indices[0]].copy()
    results["score"] = scores[0]
    return results


def build_prompt(query: str, results: pd.DataFrame) -> str:
    """Format retrieved evidence for the Gemma instruction template."""
    context = "\n\n".join(
        f"[Segment {number}]\n{row['text']}"
        for number, (_, row) in enumerate(results.iterrows(), start=1)
    )
    instruction = (
        "You are an investigative assistant analyzing source documents from the Epstein files. "
        "Answer using ONLY the provided source segments. For each factual claim, cite the "
        "'Source document IDs' in the segment headers. If the evidence does not contain the "
        "answer, state that it is not available in the retrieved data. Do not speculate."
    )
    return (
        f"<start_of_turn>user\n{instruction}\n\n"
        f"RELEVANT SOURCE SEGMENTS:\n{context}\n\n"
        f"USER QUESTION: {query}<end_of_turn>\n<start_of_turn>model\n"
    )


def load_llm(model_id: str, filename: str, token: str | None, gpu_offload: bool):
    from llama_cpp import Llama

    return Llama.from_pretrained(
        repo_id=model_id,
        filename=filename,
        n_ctx=8192,
        n_gpu_layers=-1 if gpu_offload else 0,
        token=token,
    )


def generate_answer(llm, prompt: str, max_new_tokens: int = 1024) -> str:
    response = llm(
        prompt,
        max_tokens=max_new_tokens,
        echo=False,
        temperature=0.1,
        top_p=0.95,
        stop=["<end_of_turn>", "<|endoftext|>"],
    )
    return response["choices"][0]["text"].strip()


def _chunks_fingerprint(chunks: pd.DataFrame, model_id: str) -> str:
    payload = {
        "model_id": model_id,
        "chunks": json.loads(chunks[["chunk_id", "text"]].to_json(orient="records")),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
