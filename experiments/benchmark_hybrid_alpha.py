import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bm25 import BM25Engine
from src.dense import DenseEngine
from src.hybrid import WeightedHybridEngine
from src.evaluation.metrics import ndcg_at_k, mrr, recall_at_k
from src.evaluation.data import load_beir_dataset, format_beir_corpus

def main() -> None:
    print("=== Weighted Hybrid Alpha Sweep ===")
    
    # 1. Load Data (Using centralized data module)
    beir_corpus, queries, qrels = load_beir_dataset("scifact")
    flat_corpus = format_beir_corpus(beir_corpus)

    # 2. Initialize and fit base engines once
    print("Fitting BM25 Engine...")
    bm25 = BM25Engine()
    bm25.fit(flat_corpus)

    print("Fitting Dense Engine (Encoding Corpus)...")
    dense = DenseEngine(model_name="BAAI/bge-small-en-v1.5")
    dense.fit(flat_corpus)

    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    print(f"\n{'Alpha':<8} | {'NDCG@10':<10} | {'MRR':<10} | {'Recall@100':<12}")
    print("-" * 46)

    for alpha in alphas:
        hybrid = WeightedHybridEngine(bm25_engine=bm25, dense_engine=dense, alpha=alpha)
        
        ndcg_list, mrr_list, recall_list = [], [], []

        for q_id, query in queries.items():
            if q_id not in qrels:
                continue
            
            results = hybrid.search(query, flat_corpus, top_k=100)
            scores = qrels[q_id]

            retrieved_ids = [doc_id for doc_id, _ in results]
            relevant_ids = [doc_id for doc_id, score in scores.items() if score > 0]

            ndcg_list.append(ndcg_at_k(retrieved_ids, scores, k=10))
            mrr_list.append(mrr(retrieved_ids, relevant_ids))
            recall_list.append(recall_at_k(retrieved_ids, relevant_ids, k=100))

        avg_ndcg = sum(ndcg_list) / len(ndcg_list) if ndcg_list else 0.0
        avg_mrr = sum(mrr_list) / len(mrr_list) if mrr_list else 0.0
        avg_recall = sum(recall_list) / len(recall_list) if recall_list else 0.0

        print(f"{alpha:<8.2f} | {avg_ndcg:<10.4f} | {avg_mrr:<10.4f} | {avg_recall:<12.4f}")

if __name__ == "__main__":
    main()
