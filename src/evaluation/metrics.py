import math

DocID = str | int


def precision_at_k(retrieved_ids: list[DocID], relevant_ids: list[DocID], k: int = 10) -> float:
    """Calculate Precision@K."""
    inter = list(set(retrieved_ids[:k]) & set(relevant_ids))
    return len(inter) / k


def recall_at_k(retrieved_ids: list[DocID], relevant_ids: list[DocID], k: int = 10) -> float:
    """Calculate Recall@K."""
    if not relevant_ids:
        return 0.0
    inter = list(set(retrieved_ids[:k]) & set(relevant_ids))
    return len(inter) / len(relevant_ids)


def mrr(retrieved_ids: list[DocID], relevant_ids: list[DocID], k: int | None = None) -> float:
    """Calculate Reciprocal Rank for a single query."""
    rel = set(relevant_ids)
    if k is not None:
        retrieved_ids = retrieved_ids[:k]
    j = -1
    while j + 1 < len(retrieved_ids):
        j += 1
        if retrieved_ids[j] in rel:
            return 1.0 / (j + 1)
    return 0.0


def dcg(retrieved_ids: list[DocID], relevant_scores_dict: dict[DocID, int], k: int = 10) -> float:
    if not retrieved_ids:
        return 0.0
    ids = retrieved_ids[:k]
    return sum(
        [
            (2 ** relevant_scores_dict.get(id, 0.0) - 1) / math.log2(index + 2)
            for index, id in enumerate(ids)
        ]
    )


def ndcg_at_k(retrieved_ids: list[DocID], relevant_scores_dict: dict[DocID, int], k: int = 10) -> float:
    """
    Calculate NDCG@K.
    relevant_scores_dict maps doc_id to a graded relevance score (e.g., {doc_id: 3}).
    """
    ideal_ids = sorted(
        relevant_scores_dict.keys(), key=lambda x: relevant_scores_dict[x], reverse=True
    )[:k]
    dcg_val = dcg(retrieved_ids, relevant_scores_dict, k)
    idcg_val = dcg(ideal_ids, relevant_scores_dict, k)

    return dcg_val / idcg_val if idcg_val > 0 else 0.0
