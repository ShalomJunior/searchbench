import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dense import DenseEngine
from src.evaluation.metrics import ndcg_at_k
from src.evaluation.data import load_beir_dataset, format_beir_corpus


def main() -> None:
    print("=== Day 13: Embedding Model Comparison ===")

    # 1. Load Data (Using our centralized data module)
    beir_corpus, queries, qrels = load_beir_dataset("scifact")
    flat_corpus = format_beir_corpus(beir_corpus)

    models_to_test = [
        "all-MiniLM-L6-v2",  # Our baseline
        "BAAI/bge-small-en-v1.5",  # Current SOTA for small models
        "all-mpnet-base-v2",  # Larger, heavier model
    ]

    print(
        f"{'Model':<25} | {'NDCG@10':<9} | {'Encode (s)':<10} | {'Latency (ms)':<12} | {'Index (MB)':<10}"
    )
    print("-" * 75)

    for model_name in models_to_test:
        engine = DenseEngine(model_name=model_name)

        # 1. Measure Corpus Encoding Time
        start_encode = time.time()
        engine.fit(flat_corpus)
        encode_time = time.time() - start_encode

        # 2. Calculate Index Size (num_vectors * dimensions * 4 bytes for float32)
        dimensions = engine.index.d
        num_vectors = engine.index.ntotal
        index_size_mb = (num_vectors * dimensions * 4) / (1024 * 1024)

        # 3. Measure Quality (NDCG@10) and Query Latency
        ndcg_scores = []
        total_search_time = 0.0
        valid_queries = 0

        for q_id, query in queries.items():
            if q_id not in qrels:
                continue

            start_search = time.time()
            results = engine.search(query, corpus=flat_corpus, top_k=10)
            total_search_time += time.time() - start_search

            retrieved_ids = [doc_id for doc_id, _ in results]
            ndcg_scores.append(ndcg_at_k(retrieved_ids, qrels[q_id], k=10))
            valid_queries += 1

        avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if valid_queries > 0 else 0.0
        avg_latency_ms = (
            (total_search_time / valid_queries) * 1000 if valid_queries > 0 else 0.0
        )

        print(
            f"{model_name:<25} | {avg_ndcg:<9.4f} | {encode_time:<10.2f} | {avg_latency_ms:<12.2f} | {index_size_mb:<10.2f}"
        )

    print("-" * 75)
    print(
        "\nBenchmark Complete! This highlights the critical Quality vs Latency vs Memory trade-off."
    )


if __name__ == "__main__":
    main()
