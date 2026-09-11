from sentence_transformers import CrossEncoder

DocID = str | int

class CrossEncoderReRanker:
    """
    A neural re-ranker that scores (query, document) pairs simultaneously 
    using full cross-attention. This serves as Stage 2 in a multi-stage
    retrieval pipeline.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        """
        Initializes the CrossEncoder model.
        The default model is highly optimized for fast document ranking on MS MARCO.
        """
        print(f"Loading Cross-Encoder model: {model_name}...")
        self.model = CrossEncoder(model_name)

    def rerank(
        self, 
        query: str, 
        candidates: list[tuple[DocID, float]], 
        corpus: dict[DocID, str], 
        top_k: int = 10
    ) -> list[tuple[DocID, float]]:
        """
        Re-ranks a list of candidate documents using the cross-encoder.

        Args:
            query: The raw search query string.
            candidates: List of (DocID, initial_score) retrieved by the Stage-1 engine (e.g., Top 100).
            corpus: Document dictionary mapping DocID to full document text.
            top_k: Number of re-ranked documents to return (default: 10).

        Returns:
            List of (DocID, cross_encoder_score) sorted in descending order of relevance.
        """
        if not candidates:
            return []

        # 1. Prepare the input pairs: a list of [query, document_text]
        model_inputs = []
        doc_ids = []
        
        for doc_id, _ in candidates:
            doc_text = corpus.get(doc_id, "")
            model_inputs.append([query, doc_text])
            doc_ids.append(doc_id)

        # 2. Predict relevance scores for all pairs at once (batched execution)
        cross_scores = self.model.predict(model_inputs)

        # 3. Pair the new scores back with their respective DocIDs
        reranked_results = [
            (doc_ids[i], float(cross_scores[i])) 
            for i in range(len(doc_ids))
        ]

        # 4. Sort by the new cross-encoder score in descending order
        reranked_results.sort(key=lambda item: item[1], reverse=True)

        return reranked_results[:top_k]
