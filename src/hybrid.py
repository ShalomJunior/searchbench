from typing import Optional
from src.bm25 import BM25Engine
from src.dense import DenseEngine

DocID = str | int


class WeightedHybridEngine:
    def __init__(
        self, bm25_engine: BM25Engine, dense_engine: DenseEngine, alpha: float = 0.5
    ) -> None:
        self.bm25 = bm25_engine
        self.dense = dense_engine
        self.alpha = alpha

    def _normalize_scores(
        self, results: list[tuple[DocID, float]]
    ) -> dict[DocID, float]:
        """
        TODO: Implement Min-Max normalization.
        1. Find the min and max scores in the results list.
        2. Handle the edge case where max == min (to avoid division by zero).
        3. Return a dictionary mapping DocID -> normalized score [0.0, 1.0].
        """
        if not results:
            return {}
        mn = min(score for _, score in results)
        mx = max(score for _, score in results)
        if mx == mn:
            return {doc_id: 0.0 for doc_id, _ in results}
        return {doc_id: (score - mn) / (mx - mn) for doc_id, score in results}

    def search(
        self, query: str, corpus: dict[DocID, str], top_k: int = 100
    ) -> list[tuple[DocID, float]]:
        """
        TODO: Implement Score Fusion.
        1. Run self.bm25.search() and self.dense.search().
        2. Normalize both sets of results using self._normalize_scores().
        3. Create a dictionary to hold the combined scores for every unique DocID found.
        4. Apply the convex combination formula using self.alpha.
           - Remember: If a document was only found by one engine, its score from the other engine is 0.0.
        5. Sort the dictionary by final score in descending order.
        6. Return the top_k (DocID, score) tuples.
        """
        bm25_results = self.bm25.search(query, corpus, top_k)
        dense_results = self.dense.search(query, corpus, top_k)

        normalized_bm25 = self._normalize_scores(bm25_results)
        normalized_dense = self._normalize_scores(dense_results)

        combined_scores = {}
        for doc_id in set(normalized_bm25.keys()) | set(normalized_dense.keys()):
            bm25_score = normalized_bm25.get(doc_id, 0.0)
            dense_score = normalized_dense.get(doc_id, 0.0)
            combined_scores[doc_id] = (
                self.alpha * bm25_score + (1 - self.alpha) * dense_score
            )

        sorted_results = sorted(
            combined_scores.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_results[:top_k]


class RRFHybridEngine:
    def __init__(
        self, bm25_engine: BM25Engine, dense_engine: DenseEngine, k: int = 60
    ) -> None:
        self.bm25 = bm25_engine
        self.dense = dense_engine
        self.k = k

    def search(
        self, query: str, corpus: dict[DocID, str], top_k: int = 100
    ) -> list[tuple[DocID, float]]:
        """
        TODO: Implement Reciprocal Rank Fusion.
        1. Run self.bm25.search() and self.dense.search().
        2. Initialize a dictionary mapping DocID -> RRF score (default to 0.0).
        3. Loop through the BM25 results. For each document, add its RRF score:
           1.0 / (self.k + rank). Note: rank is 1-indexed (1, 2, 3...).
        4. Loop through the Dense results and ADD to the same dictionary.
        5. Sort the dictionary by final RRF score in descending order.
        6. Return the top_k (DocID, score) tuples.
        """
        bm25_results = self.bm25.search(query, corpus, top_k)
        dense_results = self.dense.search(query, corpus, top_k)
        rrf_scores = {}
        for rank, (doc_id, _) in enumerate(bm25_results, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (self.k + rank)
        for rank, (doc_id, _) in enumerate(dense_results, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (self.k + rank)
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]
