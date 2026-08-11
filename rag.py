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


def load_reranker(model_id: str, device: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_id, device=device)


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


def build_tfidf_index(chunks: pd.DataFrame):
    """Build the lightweight lexical index used by hybrid retrieval."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b",
    )
    return vectorizer, vectorizer.fit_transform(chunks["text"])


def retrieve_dense(
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


def retrieve_hybrid(
    query: str,
    embedding_model,
    dense_index,
    tfidf_vectorizer,
    tfidf_matrix,
    reranker,
    chunks: pd.DataFrame,
    dense_k: int = 20,
    tfidf_k: int = 20,
    rerank_k: int = 5,
    neighbors: int = 1,
    max_chunks: int = 10,
) -> pd.DataFrame:
    """Retrieve dense and lexical candidates, rerank them, then add neighbors."""
    if chunks.empty:
        raise ValueError("Cannot retrieve from an empty chunk collection.")
    if tfidf_matrix.shape[0] != len(chunks):
        raise ValueError("TF-IDF index does not match the chunk collection.")
    if max_chunks < 1:
        raise ValueError("max_chunks must be positive.")

    dense = retrieve_dense(query, embedding_model, dense_index, chunks, k=dense_k)

    query_vector = tfidf_vectorizer.transform([query])
    tfidf_scores = (tfidf_matrix @ query_vector.T).toarray().ravel()
    tfidf_indices = np.argsort(-tfidf_scores)[: min(tfidf_k, len(chunks))]
    lexical = chunks.iloc[tfidf_indices].copy()

    candidates = pd.concat([dense, lexical], ignore_index=True).drop_duplicates("chunk_id")
    pairs = [(query, text) for text in candidates["text"]]
    candidates["score"] = np.asarray(reranker.predict(pairs, batch_size=16)).reshape(-1)

    anchor_count = min(rerank_k, max_chunks, len(candidates))
    anchors = candidates.nlargest(anchor_count, "score").copy()
    anchors["retrieval_role"] = "anchor"
    return _add_neighbors(anchors, chunks, neighbors, max_chunks)


def _add_neighbors(
    anchors: pd.DataFrame,
    chunks: pd.DataFrame,
    neighbors: int,
    max_chunks: int,
) -> pd.DataFrame:
    """Append nearby chunks without dropping any reranked anchor."""
    anchors = anchors.head(max_chunks).copy()
    if neighbors < 1 or len(anchors) >= max_chunks:
        return anchors.reset_index(drop=True)

    positions = {
        (row["document_index"], row["chunk_index"]): index
        for index, row in chunks.iterrows()
    }
    selected = [anchors]
    seen = set(anchors["chunk_id"])

    for _, anchor in anchors.iterrows():
        for distance in range(1, neighbors + 1):
            for chunk_index in (
                anchor["chunk_index"] - distance,
                anchor["chunk_index"] + distance,
            ):
                index = positions.get((anchor["document_index"], chunk_index))
                if index is None or chunks.at[index, "chunk_id"] in seen:
                    continue

                neighbor = chunks.loc[[index]].copy()
                neighbor["score"] = np.nan
                neighbor["retrieval_role"] = "neighbor"
                selected.append(neighbor)
                seen.add(chunks.at[index, "chunk_id"])
                if len(seen) >= max_chunks:
                    return pd.concat(selected, ignore_index=True)

    return pd.concat(selected, ignore_index=True)

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
