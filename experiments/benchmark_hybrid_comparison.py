import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bm25 import BM25Engine
from src.dense import DenseEngine
from src.hybrid import WeightedHybridEngine, RRFHybridEngine
from src.evaluation.metrics import ndcg_at_k, mrr, recall_at_k
from src.evaluation.data import load_beir_dataset, format_beir_corpus

def evaluate_engine(engine, queries, qrels, corpus, name):
    ndcg_list, mrr_list, recall_list = [], [], []
    for q_id, query in queries.items():
        if q_id not in qrels:
            continue
        results = engine.search(query, corpus, top_k=100)
        scores = qrels[q_id]

        retrieved_ids = [doc_id for doc_id, _ in results]
        relevant_ids = [doc_id for doc_id, s in scores.items() if s > 0]

        ndcg_list.append(ndcg_at_k(retrieved_ids, scores, k=10))
        mrr_list.append(mrr(retrieved_ids, relevant_ids))
        recall_list.append(recall_at_k(retrieved_ids, relevant_ids, k=100))

    avg_ndcg = sum(ndcg_list) / len(ndcg_list) if ndcg_list else 0.0
    avg_mrr = sum(mrr_list) / len(mrr_list) if mrr_list else 0.0
    avg_recall = sum(recall_list) / len(recall_list) if recall_list else 0.0

    print(f"{name:<25} | {avg_ndcg:<10.4f} | {avg_mrr:<10.4f} | {avg_recall:<12.4f}")

def main() -> None:
    print("=== Ultimate Search Architecture Comparison ===")
    
    # 1. Load Data
    beir_corpus, queries, qrels = load_beir_dataset("scifact")
    flat_corpus = format_beir_corpus(beir_corpus)

    print("\nFitting BM25 Engine (Only needs to be done once)...")
    bm25 = BM25Engine()
    bm25.fit(flat_corpus)

    models_to_test = ["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]

    for model_name in models_to_test:
        print(f"\n{'='*65}")
        print(f"Testing with Dense Model: {model_name}")
        print(f"{'='*65}")
        
        dense = DenseEngine(model_name=model_name)
        dense.fit(flat_corpus)

        weighted = WeightedHybridEngine(bm25, dense, alpha=0.25)
        rrf = RRFHybridEngine(bm25, dense, k=60)

        print(f"\n{'System':<25} | {'NDCG@10':<10} | {'MRR':<10} | {'Recall@100':<12}")
        print("-" * 65)
        
        evaluate_engine(bm25, queries, qrels, flat_corpus, "BM25 (Baseline)")
        evaluate_engine(dense, queries, qrels, flat_corpus, f"Dense ({model_name.split('/')[-1]})")
        evaluate_engine(weighted, queries, qrels, flat_corpus, "Weighted Hybrid (a=0.25)")
        evaluate_engine(rrf, queries, qrels, flat_corpus, "RRF Hybrid (k=60)")

if __name__ == "__main__":
    main()
