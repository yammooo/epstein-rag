"""Embedding computation, dense & hybrid retrieval, reranking, prompting, and LLM inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_embedding_model(model_id: str, device: str):
    """Initialize and return a SentenceTransformer dense embedding model.

    Args:
        model_id: Hugging Face model repository identifier (e.g. 'BAAI/bge-base-en-v1.5').
        device: Hardware device for model execution ('cuda', 'cpu', 'mps').

    Returns:
        SentenceTransformer: Initialized sentence transformer model instance.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_id, device=device)


def load_reranker(model_id: str, device: str):
    """Initialize and return a CrossEncoder model for candidate reranking.

    Args:
        model_id: Hugging Face model repository identifier (e.g. 'cross-encoder/ms-marco-MiniLM-L6-v2').
        device: Hardware device for model execution ('cuda', 'cpu', 'mps').

    Returns:
        CrossEncoder: Initialized cross-encoder reranker instance.
    """
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_id, device=device)


def compute_embeddings(chunks: pd.DataFrame, embedding_model, batch_size: int = 64) -> np.ndarray:
    """Generate normalized dense embeddings for document text chunks.

    Args:
        chunks: DataFrame containing text chunks in a 'text' column.
        embedding_model: SentenceTransformer instance.
        batch_size: Inference batch size (default 64).

    Returns:
        np.ndarray: 2D float32 array of shape (n_chunks, embedding_dim) with L2-normalized vectors.
    """
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
    """Load cached chunk embeddings or compute and save them if cache is missing or stale.

    Validates cache integrity using a SHA-256 fingerprint hash based on chunk text contents,
    chunk IDs, and model ID.

    Args:
        chunks: DataFrame of document text chunks.
        embedding_model: SentenceTransformer instance.
        cache_path: File path to '.npy' binary cache location.
        model_id: Model repository ID string for cache validation fingerprinting.
        force: If True, bypass cache and recompute embeddings.

    Returns:
        np.ndarray: 2D float32 array of dense embeddings.
    """
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
    """Build a FAISS flat inner-product index over L2-normalized embeddings for cosine similarity.

    Args:
        embeddings: 2D float32 array of normalized vector embeddings.

    Returns:
        faiss.IndexFlatIP: Populated FAISS inner-product vector index.
    """
    import faiss

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype("float32"))
    return index


def build_tfidf_index(chunks: pd.DataFrame):
    """Construct a TF-IDF vectorizer and sparse document-term matrix for lexical search.

    Args:
        chunks: DataFrame containing text chunks in a 'text' column.

    Returns:
        tuple[TfidfVectorizer, scipy.sparse.csr_matrix]: Fitted TfidfVectorizer instance
            and sparse TF-IDF document-term feature matrix.
    """
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
    """Retrieve the top-k highest scoring chunks using dense vector inner-product search.

    Args:
        query: User input search query string.
        embedding_model: SentenceTransformer model instance.
        index: FAISS IndexFlatIP vector index.
        chunks: Master DataFrame containing chunk records.
        k: Maximum number of nearest neighbor chunks to return (default 5).

    Returns:
        pd.DataFrame: Top-k matching chunk records with inner-product similarity in 'score' column.

    Raises:
        ValueError: If the chunk collection is empty.
    """
    if chunks.empty:
        raise ValueError("Cannot retrieve from an empty chunk collection.")
    if k <= 0:
        return chunks.iloc[[]].copy()

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
    """Perform hybrid retrieval combining dense and lexical search, cross-encoder reranking, and neighbor expansion.

    Retrieves top candidates from both dense (FAISS) and lexical (TF-IDF) indices,
    deduplicates candidates, scores query-document pairs with a CrossEncoder reranker,
    selects top reranked anchor chunks, and expands surrounding document context via neighbor chunks.

    Args:
        query: User input search query string.
        embedding_model: SentenceTransformer model instance.
        dense_index: FAISS vector index.
        tfidf_vectorizer: Fitted scikit-learn TfidfVectorizer instance.
        tfidf_matrix: Sparse TF-IDF document-term matrix.
        reranker: CrossEncoder reranker model instance.
        chunks: Master DataFrame containing chunk records.
        dense_k: Number of dense candidates to retrieve (default 20).
        tfidf_k: Number of lexical candidates to retrieve (default 20).
        rerank_k: Number of top cross-encoder reranked anchors to retain (default 5).
        neighbors: Number of adjacent context chunks to append per anchor (default 1).
        max_chunks: Maximum total chunk count capacity limit (default 10).

    Returns:
        pd.DataFrame: Combined DataFrame of anchor and neighbor chunks annotated with 'retrieval_role'
            ('anchor' or 'neighbor') and 'score'.

    Raises:
        ValueError: If chunks collection is empty, TF-IDF matrix dimension mismatches chunks, or max_chunks < 1.
    """
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
    """Expand retrieved context by adding adjacent document chunks around anchor chunks.

    Appends preceding and succeeding chunks (chunk_index - d, chunk_index + d) from the same
    source document while preserving anchor ordering and respecting max_chunks limit.

    Args:
        anchors: DataFrame of top reranked anchor chunks.
        chunks: Master DataFrame of all document chunks.
        neighbors: Radial distance of adjacent chunks to append.
        max_chunks: Hard limit on total total returned chunks.

    Returns:
        pd.DataFrame: Concatenated DataFrame of anchor and neighbor chunks.
    """
    anchors = anchors.head(max_chunks).copy()
    if neighbors < 1 or len(anchors) >= max_chunks:
        return anchors.reset_index(drop=True)

    positions = dict(
        zip(
            zip(chunks["document_index"], chunks["chunk_index"]),
            chunks.index,
        )
    )
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
    """Format retrieved context chunks and user question into a Gemma instruction-tuned prompt.

    Args:
        query: User input question string.
        results: DataFrame of retrieved chunks containing text and provenance headers.

    Returns:
        str: Fully formatted prompt string with system instructions, labeled source segments,
            and model response turn tags.
    """
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
    """Load a GGUF format quantized language model via llama-cpp-python.

    Args:
        model_id: Hugging Face model repository ID (e.g. 'google/gemma-2-9b-it-GGUF').
        filename: GGUF filename (e.g. 'gemma-2-9b-it-Q4_K_M.gguf').
        token: Optional Hugging Face access token for private repositories.
        gpu_offload: If True, offload all model layers to GPU (-1); otherwise CPU (0).

    Returns:
        Llama: Initialized Llama CPP model instance.
    """
    from llama_cpp import Llama

    return Llama.from_pretrained(
        repo_id=model_id,
        filename=filename,
        n_ctx=8192,
        n_gpu_layers=-1 if gpu_offload else 0,
        token=token,
    )


def generate_answer(llm, prompt: str, max_new_tokens: int = 1024) -> str:
    """Execute LLM text generation on a formatted prompt and return the stripped answer string.

    Args:
        llm: Initialized Llama model instance.
        prompt: Formatted prompt string.
        max_new_tokens: Token generation budget ceiling (default 1024).

    Returns:
        str: Generated answer text stripped of leading/trailing whitespace.
    """
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
    """Compute a SHA-256 fingerprint hash for embedding cache validation.

    Args:
        chunks: DataFrame containing 'chunk_id' and 'text' columns.
        model_id: Embedding model ID string.

    Returns:
        str: 64-character SHA-256 hex digest string.
    """
    payload = {
        "model_id": model_id,
        "chunks": json.loads(chunks[["chunk_id", "text"]].to_json(orient="records")),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

