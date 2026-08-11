import unittest

import numpy as np
import pandas as pd

from rag import build_tfidf_index, retrieve_hybrid


class FakeEmbeddingModel:
    def encode(self, texts, normalize_embeddings=True):
        return np.array([[1.0, 0.0]], dtype="float32")


class FakeDenseIndex:
    def search(self, query_embedding, k):
        indices = np.array([[2, 3]])[:, :k]
        scores = np.array([[0.9, 0.8]], dtype="float32")[:, :k]
        return scores, indices


class FakeReranker:
    def predict(self, pairs, batch_size=16):
        return np.array(
            [10.0 if "Hotel Raphael" in text else 1.0 for _, text in pairs]
        )


class HybridRetrievalTest(unittest.TestCase):
    def test_reranks_lexical_candidate_and_adds_neighbor(self):
        chunks = pd.DataFrame(
            [
                {
                    "chunk_id": "hotel_0",
                    "document_index": 0,
                    "chunk_index": 0,
                    "text": "Meeting moved to Hotel Raphael because of security difficulties.",
                },
                {
                    "chunk_id": "hotel_1",
                    "document_index": 0,
                    "chunk_index": 1,
                    "text": "The meeting was scheduled for 8:30 p.m. on Tuesday.",
                },
                {
                    "chunk_id": "other_0",
                    "document_index": 1,
                    "chunk_index": 0,
                    "text": "A different meeting involving Barak.",
                },
                {
                    "chunk_id": "other_1",
                    "document_index": 2,
                    "chunk_index": 0,
                    "text": "Unrelated material.",
                },
            ]
        )
        vectorizer, matrix = build_tfidf_index(chunks)

        results = retrieve_hybrid(
            "Why was the meeting moved to Hotel Raphael, and at what time?",
            FakeEmbeddingModel(),
            FakeDenseIndex(),
            vectorizer,
            matrix,
            FakeReranker(),
            chunks,
            dense_k=2,
            tfidf_k=2,
            rerank_k=1,
            neighbors=1,
            max_chunks=2,
        )

        self.assertEqual(results["chunk_id"].tolist(), ["hotel_0", "hotel_1"])
        self.assertEqual(results["retrieval_role"].tolist(), ["anchor", "neighbor"])


if __name__ == "__main__":
    unittest.main()
