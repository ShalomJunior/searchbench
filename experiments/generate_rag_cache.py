import os
import sys
import json
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bm25 import BM25Engine
from src.dense import DenseEngine
from src.hybrid import RRFHybridEngine
from src.reranker import CrossEncoderReRanker
from src.evaluation.data import load_beir_dataset, format_beir_corpus

def main():
    parser = argparse.ArgumentParser(description="Generate RAG Retrieval Cache")
    parser.add_argument("--dataset", type=str, default="scifact")
    parser.add_argument("--limit", type=int, default=50, help="Number of queries to cache for RAG evaluation")
    args = parser.parse_args()

    print(f"=== Generating RAG Retrieval Cache ({args.dataset.upper()}) ===")
    
    beir_corpus, queries, qrels = load_beir_dataset(args.dataset)
    flat_corpus = format_beir_corpus(beir_corpus)

    # Limit the number of queries for the manual evaluation set
    queries_list = list(queries.items())
    if args.limit and len(queries_list) > args.limit:
        queries_list = queries_list[:args.limit]
    
    print("\nLoading models and building indices...")
    bm25 = BM25Engine()
    bm25.fit(flat_corpus)

    dense = DenseEngine(model_name="all-MiniLM-L6-v2")
    dense.fit(flat_corpus)

    hybrid = RRFHybridEngine(bm25_engine=bm25, dense_engine=dense, k=60)
    reranker = CrossEncoderReRanker()

    print(f"\nRunning retrieval for {len(queries_list)} queries...")
    
    cache_data = []
    
    for i, (q_id, query) in enumerate(queries_list):
        # 1. Retrieve top 100 candidates with Hybrid RRF
        candidates = hybrid.search(query, flat_corpus, top_k=100)
        
        # 2. Rerank to get top 10
        top_10 = reranker.rerank(query, candidates, flat_corpus, top_k=10)
        
        # 3. Format documents for the cache
        docs_cache = []
        for rank, (doc_id, score) in enumerate(top_10):
            docs_cache.append({
                "doc_id": doc_id,
                "text": flat_corpus[doc_id],
                "score": float(score),
                "rank": rank + 1
            })
            
        cache_data.append({
            "query_id": q_id,
            "query": query,
            "documents": docs_cache
        })
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(queries_list)} queries...")

    # Save to data/rag_retrieval_cache.json
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"rag_retrieval_cache_{args.dataset}.json")
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, indent=2, ensure_ascii=False)
        
    print(f"\nSuccess! Cached {len(queries_list)} queries and their top-10 documents to {out_path}")

if __name__ == "__main__":
    main()
