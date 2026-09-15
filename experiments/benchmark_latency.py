import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bm25 import BM25Engine
from src.dense import DenseEngine
from src.hybrid import RRFHybridEngine
from src.evaluation.data import load_beir_dataset, format_beir_corpus


def main() -> None:
    print("=== Latency Measurement Benchmark ===")

    # 1. Load Data
    beir_corpus, queries, qrels = load_beir_dataset("scifact")
    flat_corpus = format_beir_corpus(beir_corpus)

    print("\nFitting BM25 Engine...")
    bm25 = BM25Engine()
    bm25.fit(flat_corpus)

    print("Fitting Dense Engine (all-MiniLM-L6-v2)...")
    dense = DenseEngine(model_name="all-MiniLM-L6-v2")
    dense.fit(flat_corpus)

    rrf = RRFHybridEngine(bm25, dense, k=60)

    bm25_times = []
    dense_times = []
    rrf_total_times = []

    print("\nMeasuring search latency over 300 queries...")
    for q_id, query in queries.items():
        if q_id not in qrels:
            continue

        # Measure BM25
        t0 = time.perf_counter()
        bm25.search(query, flat_corpus, top_k=100)
        bm25_times.append(time.perf_counter() - t0)

        # Measure Dense
        t0 = time.perf_counter()
        dense.search(query, flat_corpus, top_k=100)
        dense_times.append(time.perf_counter() - t0)

        # Measure Total RRF Pipeline (which runs both internally + fuses)
        t0 = time.perf_counter()
        rrf.search(query, flat_corpus, top_k=100)
        rrf_total_times.append(time.perf_counter() - t0)

    avg_bm25 = (sum(bm25_times) / len(bm25_times)) * 1000  # Convert to ms
    avg_dense = (sum(dense_times) / len(dense_times)) * 1000
    avg_total = (sum(rrf_total_times) / len(rrf_total_times)) * 1000
    avg_fusion = avg_total - avg_bm25 - avg_dense

    print("\nAverage Latency Per Query (Milliseconds):")
    print("-" * 45)
    print(f"BM25 Latency        : {avg_bm25:>6.2f} ms")
    print(f"Dense Latency       : {avg_dense:>6.2f} ms")
    print(f"RRF Fusion Overhead : {avg_fusion:>6.2f} ms")
    print("-" * 45)
    print(f"Total Pipeline      : {avg_total:>6.2f} ms")

    print("\nNote: The total pipeline runs sequentially right now.")
    print("In production, BM25 and Dense are executed asynchronously in parallel,")
    print(
        f"which would bring the total latency closer to max(BM25, Dense) + Fusion = {max(avg_bm25, avg_dense) + avg_fusion:.2f} ms."
    )


if __name__ == "__main__":
    main()
