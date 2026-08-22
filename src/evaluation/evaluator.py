import time
from typing import Any
from src.evaluation.metrics import ndcg_at_k, mrr, recall_at_k

def evaluate_engine(
    engine: Any,
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    corpus: dict[str, str],
    top_k: int = 100,
) -> None:
    """Generic evaluation loop for any search engine."""
    ndcg_list = []
    mrr_list = []
    recall_10_list = []
    recall_100_list = []

    print("\nRunning Evaluation...")
    start_eval = time.time()
    
    for q_id, query in queries.items():
        # Safety guard: skip queries with no ground truth
        if q_id not in qrels:
            continue

        results = engine.search(query, corpus=corpus, top_k=top_k)
        scores = qrels[q_id]

        retrieved_ids = [doc_id for doc_id, _ in results]
        relevant_ids = [doc_id for doc_id, score in scores.items() if score > 0]

        ndcg_list.append(ndcg_at_k(retrieved_ids, scores, k=10)) # type: ignore
        mrr_list.append(mrr(retrieved_ids, relevant_ids, k=10)) # type: ignore
        recall_10_list.append(recall_at_k(retrieved_ids, relevant_ids, k=10)) # type: ignore
        recall_100_list.append(recall_at_k(retrieved_ids, relevant_ids, k=100)) # type: ignore

    print(f"Evaluation completed in {time.time() - start_eval:.4f} seconds.\n")

    if queries:
        print(f"Average NDCG@10:    {sum(ndcg_list) / len(ndcg_list):.4f}")
        print(f"Average MRR@10:     {sum(mrr_list) / len(mrr_list):.4f}")
        print(f"Average Recall@10:  {sum(recall_10_list) / len(recall_10_list):.4f}")
        print(f"Average Recall@100: {sum(recall_100_list) / len(recall_100_list):.4f}")
