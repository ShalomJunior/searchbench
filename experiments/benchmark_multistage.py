import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from src.bm25 import BM25Engine
from src.elastic import ElasticBM25Engine
from src.dense import DenseEngine
from src.hybrid import RRFHybridEngine
from src.reranker import CrossEncoderReRanker
from src.evaluation.metrics import ndcg_at_k, mrr, recall_at_k
from src.evaluation.data import load_beir_dataset, format_beir_corpus


def evaluate_pipeline(name: str, queries: dict, qrels: dict, search_func) -> None:
    ndcg_list = []
    mrr_list = []
    recall_list = []
    total_latency = 0.0
    valid_queries = 0

    for q_id, query in queries.items():
        if q_id not in qrels:
            continue

        scores = qrels[q_id]
        relevant_ids = [doc_id for doc_id, score in scores.items() if score > 0]

        if not relevant_ids:
            continue

        start_time = time.perf_counter()
        results = search_func(query)
        total_latency += time.perf_counter() - start_time

        retrieved_ids = [doc_id for doc_id, _ in results]

        ndcg_list.append(ndcg_at_k(retrieved_ids, scores, k=10))
        mrr_list.append(mrr(retrieved_ids, relevant_ids))
        recall_list.append(recall_at_k(retrieved_ids, relevant_ids, k=100))
        valid_queries += 1

    avg_ndcg = sum(ndcg_list) / len(ndcg_list) if ndcg_list else 0.0
    avg_mrr = sum(mrr_list) / len(mrr_list) if mrr_list else 0.0
    avg_recall = sum(recall_list) / len(recall_list) if recall_list else 0.0
    avg_latency_ms = (total_latency / valid_queries) * 1000 if valid_queries else 0.0

    print(
        f"{name:<27} | {avg_ndcg:<10.4f} | {avg_mrr:<10.4f} | {avg_recall:<10.4f} | {avg_latency_ms:<12.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Stage Architecture Benchmark")
    parser.add_argument(
        "--dataset",
        type=str,
        default="scifact",
        help="BEIR dataset to evaluate on (e.g., scifact, fiqa)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of queries to evaluate (prevents 15h timeouts on large datasets)",
    )
    parser.add_argument(
        "--use-elastic",
        action="store_true",
        help="Use Elasticsearch backend instead of pure Python BM25",
    )
    args = parser.parse_args()

    print(f"=== Multi-Stage Architecture Benchmark ({args.dataset.upper()}) ===")

    beir_corpus, queries, qrels = load_beir_dataset(args.dataset)

    if args.limit:
        print(f"Limiting evaluation to the first {args.limit} queries...")
        queries = dict(list(queries.items())[: args.limit])
    flat_corpus = format_beir_corpus(beir_corpus)

    print("\nLoading models and building indices (this may take a few minutes)...")

    if args.use_elastic:
        print(
            "Using ElasticBM25Engine (Ensure Elasticsearch is running on localhost:9200)"
        )
        bm25 = ElasticBM25Engine()
    else:
        print("Using Python BM25Engine (Warning: Slow on large datasets)")
        bm25 = BM25Engine()

    bm25.fit(flat_corpus)

    # We use BGE for the dense baseline to push maximum quality
    dense = DenseEngine(model_name="BAAI/bge-small-en-v1.5")
    dense.fit(flat_corpus)

    # We use our existing RRFHybridEngine instead of HybridRetriever
    hybrid = RRFHybridEngine(bm25_engine=bm25, dense_engine=dense, k=60)
    reranker = CrossEncoderReRanker()

    finetuned_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "results",
        "fine_tuned_cross_encoder",
    )
    finetuned_reranker = (
        CrossEncoderReRanker(model_name=finetuned_path)
        if os.path.exists(finetuned_path)
        else None
    )

    print(
        f"\n{'System Architecture':<27} | {'NDCG@10':<10} | {'MRR@10':<10} | {'Recall@100':<10} | {'Latency (ms)':<12}"
    )
    print("-" * 85)

    # 1. BM25 Baseline
    evaluate_pipeline(
        "1. BM25", queries, qrels, lambda q: bm25.search(q, flat_corpus, top_k=100)
    )

    # 2. Dense Baseline
    evaluate_pipeline(
        "2. Dense (BGE)",
        queries,
        qrels,
        lambda q: dense.search(q, flat_corpus, top_k=100),
    )

    # 3. Hybrid (First-Stage)
    evaluate_pipeline(
        "3. Hybrid (RRF)",
        queries,
        qrels,
        lambda q: hybrid.search(q, flat_corpus, top_k=100),
    )

    # 4. Hybrid + Cross-Encoder
    def hybrid_plus_reranker(q):
        # Fetch top 100 candidates rapidly
        candidates = hybrid.search(q, flat_corpus, top_k=100)
        # Rerank to get the definitive top 100 for evaluation
        return reranker.rerank(q, candidates, flat_corpus, top_k=100)

    evaluate_pipeline("4. Hybrid + Base Reranker", queries, qrels, hybrid_plus_reranker)

    # 5. Hybrid + Fine-Tuned Cross-Encoder
    if finetuned_reranker:

        def hybrid_plus_finetuned(q):
            candidates = hybrid.search(q, flat_corpus, top_k=100)
            return finetuned_reranker.rerank(q, candidates, flat_corpus, top_k=100)

        evaluate_pipeline(
            "5. Hybrid + FT Reranker", queries, qrels, hybrid_plus_finetuned
        )
    else:
        print(
            f"{'5. Hybrid + FT Reranker':<27} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<12} (Model not found)"
        )


if __name__ == "__main__":
    main()
