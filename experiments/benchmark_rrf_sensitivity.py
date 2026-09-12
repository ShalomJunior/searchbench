import os
import sys
import csv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bm25 import BM25Engine
from src.dense import DenseEngine
from src.hybrid import RRFHybridEngine
from src.evaluation.metrics import ndcg_at_k, mrr, recall_at_k
from src.evaluation.data import load_beir_dataset, format_beir_corpus

def main() -> None:
    print("=== RRF Sensitivity Analysis (k Sweep) ===")
    
    # 1. Load Data
    beir_corpus, queries, qrels = load_beir_dataset("scifact")
    flat_corpus = format_beir_corpus(beir_corpus)

    print("\nFitting BM25 Engine...")
    bm25 = BM25Engine()
    bm25.fit(flat_corpus)

    # Use MiniLM for faster iteration during parameter sweeping
    print("Fitting Dense Engine (Encoding Corpus)...")
    dense = DenseEngine(model_name="all-MiniLM-L6-v2")
    dense.fit(flat_corpus)

    k_values = [1, 5, 10, 20, 30, 40, 60, 80, 100, 150, 200]
    results_data = []

    print(f"\n{'k':<5} | {'NDCG@10':<10} | {'MRR@10':<10} | {'Recall@100':<12}")
    print("-" * 46)

    for k in k_values:
        rrf = RRFHybridEngine(bm25, dense, k=k)
        
        ndcg_list, mrr_list, recall_list = [], [], []

        for q_id, query in queries.items():
            if q_id not in qrels:
                continue
            
            results = rrf.search(query, flat_corpus, top_k=100)
            scores = qrels[q_id]

            retrieved_ids = [doc_id for doc_id, _ in results]
            relevant_ids = [doc_id for doc_id, score in scores.items() if score > 0]

            ndcg_list.append(ndcg_at_k(retrieved_ids, scores, k=10))
            mrr_list.append(mrr(retrieved_ids, relevant_ids))
            recall_list.append(recall_at_k(retrieved_ids, relevant_ids, k=100))

        avg_ndcg = sum(ndcg_list) / len(ndcg_list) if ndcg_list else 0.0
        avg_mrr = sum(mrr_list) / len(mrr_list) if mrr_list else 0.0
        avg_recall = sum(recall_list) / len(recall_list) if recall_list else 0.0

        print(f"{k:<5} | {avg_ndcg:<10.4f} | {avg_mrr:<10.4f} | {avg_recall:<12.4f}")
        results_data.append([k, avg_ndcg, avg_mrr, avg_recall])

    # 3. Export to CSV for plotting
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "rrf_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "NDCG@10", "MRR@10", "Recall@100"])
        writer.writerows(results_data)
        
    print(f"\nExported results to {csv_path}")

if __name__ == "__main__":
    main()
