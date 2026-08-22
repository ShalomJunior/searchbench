import os
import sys
import time

# Ensure we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bm25 import BM25Engine
from src.evaluation.data import load_beir_dataset, format_beir_corpus
from src.evaluation.evaluator import evaluate_engine

def main() -> None:
    print("=== BM25 Retrieval Benchmark (SciFact) ===")

    # 1. Load and format the data
    beir_corpus, queries, qrels = load_beir_dataset("scifact")
    flat_corpus = format_beir_corpus(beir_corpus)

    print(f"\nLoaded {len(flat_corpus)} documents and {len(queries)} queries.")

    # 2. Initialize and Fit BM25
    engine = BM25Engine(k1=1.5, b=0.75)
    print("Building BM25 Index...")
    start_time = time.time()
    engine.fit(flat_corpus) # type: ignore
    print(f"Index built in {time.time() - start_time:.4f} seconds.")

    # 3. Evaluate
    evaluate_engine(engine, queries, qrels, flat_corpus)

if __name__ == "__main__":
    main()
