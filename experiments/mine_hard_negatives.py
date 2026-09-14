import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bm25 import BM25Engine
from src.dense import DenseEngine
from src.hybrid import RRFHybridEngine
from src.evaluation.data import load_beir_dataset, format_beir_corpus

def main() -> None:
    print("=== Day 30: Hard Negative Mining ===")
    
    # We load the 'train' split here to generate training data
    beir_corpus, queries, qrels = load_beir_dataset("scifact", split="train")
    flat_corpus = format_beir_corpus(beir_corpus)

    print("Fitting hybrid retrieval engines...")
    bm25 = BM25Engine()
    bm25.fit(flat_corpus)

    dense = DenseEngine(model_name="BAAI/bge-small-en-v1.5")
    dense.fit(flat_corpus)
    
    hybrid = RRFHybridEngine(bm25_engine=bm25, dense_engine=dense, k=60)

    training_triplets = []
    
    print("Mining hard negatives for training queries...")
    for q_id, query in queries.items():
        if q_id not in qrels:
            continue
            
        scores = qrels[q_id]
        positive_ids = [doc_id for doc_id, s in scores.items() if s > 0]
        
        if not positive_ids:
            continue
            
        # 1. Retrieve Top 100 Candidates (we retrieve 100 here to get deep hard negatives)
        candidates = hybrid.search(query, flat_corpus, top_k=100)
        
        # 2. Filter out known positives to find negatives
        hard_negatives = [doc_id for doc_id, _ in candidates if doc_id not in positive_ids]
        
        # 3. Store Triplets (query, positive, hard_negative)
        # We sample up to 5 hard negatives per positive to balance the dataset
        for pos_id in positive_ids:
            sampled_negatives = hard_negatives[:5]
            for neg_id in sampled_negatives:
                training_triplets.append({
                    "query": query,
                    "positive": flat_corpus[pos_id],
                    "negative": flat_corpus[neg_id]
                })

    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "hard_negatives.json")
    with open(out_path, "w") as f:
        json.dump(training_triplets, f, indent=2)
        
    print(f"Successfully mined {len(training_triplets)} triplets and saved to {out_path}")

if __name__ == "__main__":
    main()
