import os
import sys
import json
import time

# Ensure we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bm25 import BM25Engine
from src.dense import DenseEngine
from src.evaluation.metrics import ndcg_at_k
from src.evaluation.data import load_beir_dataset, format_beir_corpus


def main() -> None:
    print("=== Day 12: Retrieval Error Analysis ===")

    # 1. Load Data (Using our refactored centralized data loader!)
    beir_corpus, queries, qrels = load_beir_dataset("scifact")
    flat_corpus = format_beir_corpus(beir_corpus)

    # 2. Initialize Engines
    print("\nBuilding BM25 Index...")
    bm25 = BM25Engine()
    bm25.fit(flat_corpus)

    print("\nBuilding Dense Index (Encoding corpus)...")
    dense = DenseEngine(model_name="all-MiniLM-L6-v2")
    dense.fit(flat_corpus)

    # 3. Categorization Buckets
    bm25_wins = []
    dense_wins = []
    both_fail = []

    print("\nEvaluating queries to find edge cases...")

    start_eval = time.time()
    for q_id, query in queries.items():
        if q_id not in qrels:
            continue

        scores = qrels[q_id]

        # Get Top 10 for both
        bm25_res = bm25.search(query, corpus=flat_corpus, top_k=10)
        dense_res = dense.search(query, corpus=flat_corpus, top_k=10)

        bm25_ids = [doc_id for doc_id, _ in bm25_res]
        dense_ids = [doc_id for doc_id, _ in dense_res]

        # Calculate individual NDCG@10
        bm25_ndcg = ndcg_at_k(bm25_ids, scores, k=10)
        dense_ndcg = ndcg_at_k(dense_ids, scores, k=10)

        # Find the absolute best ground truth document to display
        best_doc_id = max(scores, key=scores.get)

        analysis_payload = {
            "query_id": q_id,
            "query": query,
            "ground_truth_doc": flat_corpus.get(best_doc_id, "N/A"),
            "bm25_top_result": (
                flat_corpus.get(bm25_ids[0], "N/A") if bm25_ids else "N/A"
            ),
            "dense_top_result": (
                flat_corpus.get(dense_ids[0], "N/A") if dense_ids else "N/A"
            ),
            "bm25_ndcg": round(bm25_ndcg, 4),
            "dense_ndcg": round(dense_ndcg, 4),
        }

        # Categorize
        # Refinement: Using < 0.1 for failure instead of strictly 0.0.
        # Sometimes an engine finds a weak document at rank 10 (NDCG=0.03). This is still practically a failure.
        if bm25_ndcg > 0.5 and dense_ndcg < 0.1:
            bm25_wins.append(analysis_payload)
        elif dense_ndcg > 0.5 and bm25_ndcg < 0.1:
            dense_wins.append(analysis_payload)
        elif bm25_ndcg < 0.1 and dense_ndcg < 0.1:
            both_fail.append(analysis_payload)

    print(f"Evaluation completed in {time.time() - start_eval:.4f} seconds.\n")

    # 4. Export Report
    # We dump 10 of each to a JSON file for manual review
    report = {
        "bm25_wins": bm25_wins[:10],
        "dense_wins": dense_wins[:10],
        "both_fail": both_fail[:10],
    }

    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "error_analysis_report.json",
    )
    with open(out_path, "w") as f:
        json.dump(report, f, indent=4)

    print(f"Analysis complete! Found:")
    print(f"- {len(bm25_wins)} pure BM25 wins")
    print(f"- {len(dense_wins)} pure Dense wins")
    print(f"- {len(both_fail)} mutual failures")
    print(f"\nExported 10 examples of each to: {out_path}")


if __name__ == "__main__":
    main()
