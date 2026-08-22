import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import Optional, Any

DocID = str | int


class DenseEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        print(f"Loading Bi-Encoder model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.index: Optional[Any] = None
        self.doc_ids: list[DocID] = []

    def fit(self, corpus: dict[DocID, str]) -> None:
        doc_texts: list[str] = list(corpus.values())
        self.doc_ids = list(corpus.keys())

        # 1. Encode with normalization (crucial for Cosine Similarity) and cast to float32
        vectors: np.ndarray = self.model.encode(doc_texts, normalize_embeddings=True).astype(np.float32)

        # 2. Initialize and populate the index using Inner Product (IP) for Cosine Similarity
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def search(self, query: str, corpus: dict[DocID, str] = None, top_k: int = 100) -> list[tuple[DocID, float]]:
        if not self.index:
            return []

        # Encode query with normalization
        query_vector: np.ndarray = self.model.encode([query], normalize_embeddings=True).astype(np.float32)

        distances, faiss_indices = self.index.search(query_vector, top_k)

        # Filter out FAISS -1 padding
        results: list[tuple[DocID, float]] = [
            (self.doc_ids[i], float(distance))
            for i, distance in zip(faiss_indices[0], distances[0])
            if i != -1
        ]

        return results
