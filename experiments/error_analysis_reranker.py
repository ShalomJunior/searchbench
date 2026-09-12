import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bm25 import BM25Engine
from src.dense import DenseEngine
from src.hybrid import RRFHybridEngine
from src.reranker import CrossEncoderReRanker
from src.evaluation.metrics import ndcg_at_k
from src.evaluation.data import load_beir_dataset, format_beir_corpus

def main() -> None:
    print("=== Day 26: Re-ranking Error Analysis ===")
    
    beir_corpus, queries, qrels = load_beir_dataset("scifact")
    flat_corpus = format_beir_corpus(beir_corpus)

    print("Loading models and building indices...")
    bm25 = BM25Engine()
    bm25.fit(flat_corpus)

    dense = DenseEngine(model_name="BAAI/bge-small-en-v1.5")
    dense.fit(flat_corpus)

    hybrid = RRFHybridEngine(bm25_engine=bm25, dense_engine=dense, k=60)
    reranker = CrossEncoderReRanker()

    cases = []

    for q_id, query in queries.items():
        if q_id not in qrels:
            continue

        scores = qrels[q_id]
        relevant_doc_ids = {doc_id for doc_id, s in scores.items() if s > 0}
        if not relevant_doc_ids:
            continue

        # 1. First-stage candidate generation (Top 25)
        candidates = hybrid.search(query, flat_corpus, top_k=25)
        hybrid_top10 = [doc_id for doc_id, _ in candidates[:10]]

        # 2. Second-stage neural re-ranking (Top 10 from the 100)
        reranked = reranker.rerank(query, candidates, flat_corpus, top_k=10)
        reranked_top10 = [doc_id for doc_id, _ in reranked]

        # Check if ranking changed within the top 10
        if hybrid_top10 != reranked_top10:
            ndcg_before = ndcg_at_k(hybrid_top10, scores, k=10)
            ndcg_after = ndcg_at_k(reranked_top10, scores, k=10)
            delta = ndcg_after - ndcg_before

            cases.append({
                "query_id": q_id,
                "query": query,
                "ndcg_delta": delta,
                "ndcg_before": ndcg_before,
                "ndcg_after": ndcg_after,
                "relevant_docs": list(relevant_doc_ids),
                "before_top3": hybrid_top10[:3],
                "after_top3": reranked_top10[:3],
            })

    # Sort queries by absolute change to inspect the most dramatic rank shifts
    cases.sort(key=lambda x: abs(x["ndcg_delta"]), reverse=True)
    selected_cases = cases[:20]

    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "rerank_error_analysis_20.json")
    with open(out_path, "w") as f:
        json.dump(selected_cases, f, indent=2)

    print(f"\nSaved 20 key differential cases to {out_path}\n")
    print(f"{'Query ID':<10} | {'NDCG Delta':<12} | {'Before Top-1':<14} | {'After Top-1':<14}")
    print("-" * 55)
    for c in selected_cases[:5]:
        print(f"{c['query_id']:<10} | {c['ndcg_delta']:<+12.4f} | {str(c['before_top3'][0]):<14} | {str(c['after_top3'][0]):<14}")

if __name__ == "__main__":
    main()
