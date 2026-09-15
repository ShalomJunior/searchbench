import os
import sys
import time
import csv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bm25 import BM25Engine
from src.dense import DenseEngine
from src.hybrid import RRFHybridEngine
from src.reranker import CrossEncoderReRanker
from src.evaluation.metrics import ndcg_at_k
from src.evaluation.data import load_beir_dataset, format_beir_corpus


def main() -> None:
    print("=== Reranking Depth Latency Experiment ===")

    beir_corpus, queries, qrels = load_beir_dataset("scifact")
    flat_corpus = format_beir_corpus(beir_corpus)

    print("\nInitializing models and building indices...")
    bm25 = BM25Engine()
    bm25.fit(flat_corpus)

    dense = DenseEngine(model_name="BAAI/bge-small-en-v1.5")
    dense.fit(flat_corpus)

    # Fusion Engine
    hybrid = RRFHybridEngine(bm25_engine=bm25, dense_engine=dense, k=60)
    reranker = CrossEncoderReRanker()

    depths = [10, 25, 50, 100, 200]
    results_data = []

    print(f"\n{'Rerank Depth':<15} | {'NDCG@10':<10} | {'Latency (ms)':<12}")
    print("-" * 43)

    for depth in depths:
        ndcg_list = []
        total_latency = 0.0
        valid_queries = 0

        for q_id, query in queries.items():
            if q_id not in qrels:
                continue

            scores = qrels[q_id]

            start_time = time.perf_counter()

            # 1. First-stage retrieval up to 'depth'
            candidates = hybrid.search(query, flat_corpus, top_k=depth)

            # 2. Second-stage re-ranking
            final_results = reranker.rerank(query, candidates, flat_corpus, top_k=10)

            total_latency += time.perf_counter() - start_time

            retrieved_ids = [doc_id for doc_id, _ in final_results]
            ndcg_list.append(ndcg_at_k(retrieved_ids, scores, k=10))
            valid_queries += 1

        avg_ndcg = sum(ndcg_list) / len(ndcg_list) if ndcg_list else 0.0
        avg_latency_ms = (
            (total_latency / valid_queries) * 1000 if valid_queries else 0.0
        )

        print(f"Top {depth:<11} | {avg_ndcg:<10.4f} | {avg_latency_ms:<12.2f}")
        results_data.append([depth, avg_ndcg, avg_latency_ms])

    # Export to CSV for plotting
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "rerank_depth_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Rerank_Depth", "NDCG@10", "Latency_ms"])
        writer.writerows(results_data)

    print(f"\nExported results to {csv_path}")


if __name__ == "__main__":
    main()
