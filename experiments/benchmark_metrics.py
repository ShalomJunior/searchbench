import os
import sys

# Ensure we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bm25 import BM25Engine
from src.evaluation.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k
from src.tfidf import build_index as tfidf_build_index
from src.tfidf import search as tfidf_search


def main() -> None:
    print("=== TF-IDF vs BM25: Metrics Benchmark ===")

    # A robust 9-document corpus to properly evaluate Top-K ranking
    corpus = {
        # Doc 1: The true best result. Reasonable length, natural term frequency.
        1: "machine learning is a subset of artificial intelligence. this comprehensive guide explains how machine learning models work, how to train them, and why machine learning is transforming data science.",
        # Doc 2: The "Ratio Trap". TF-IDF will give this a perfect 1.0 TF score because it's 100% keyword.
        2: "machine learning",
        # Doc 3: The "Spam Trap". Keyword stuffing.
        3: "machine learning " * 15,
        # Doc 4 & 7: Massive noise documents to increase the corpus average document length (avgdl).
        4: "data science and data analytics are completely different fields. " * 10,
        7: "the history of artificial intelligence from the 1950s to today. " * 10,
        # Doc 5: A decent, but short alternative.
        5: "a quick introduction to machine learning techniques.",
        # Doc 6, 8, 9: Partial matches and pure noise.
        6: "deep learning and neural networks are amazing.",
        8: "learning python is fun. learning pandas is useful.",
        9: "we use a heavy machine in the factory.",
    }

    # Ground truth (3 = Perfect, 2 = Good, 1 = Barely acceptable, 0 = Spam/Irrelevant)
    query = "machine learning"
    ground_truth_scores = {
        1: 3,  # The definitive, detailed guide
        5: 2,  # A good, concise intro
        2: 1,  # Too short to be a great resource, but technically relevant
        3: 0,  # Keyword spam (should be punished)
    }

    # Any document with a score > 0 is considered "relevant" for binary metrics
    relevant_ids = [
        doc_id for doc_id, score in ground_truth_scores.items() if score > 0
    ]

    print("\n[Ground Truth]")
    print(f"Query: '{query}'")
    print(f"Relevant Docs: {relevant_ids}")

    # Initialize engines
    bm25 = BM25Engine(k1=1.5, b=0.75)

    print("\nBuilding Indexes...")
    tfidf_index = tfidf_build_index(corpus)
    bm25.fit(corpus)

    # Search
    print("\n[Executing Searches]")
    tfidf_results = tfidf_search(query, corpus, tfidf_index)
    bm25_results = bm25.search(query, corpus)

    # Extract just the retrieved IDs in ranked order
    tfidf_retrieved = [doc_id for doc_id, _ in tfidf_results]
    bm25_retrieved = [doc_id for doc_id, _ in bm25_results]

    print(f"TF-IDF Ranked Output: {tfidf_retrieved}")
    print(f"BM25 Ranked Output:   {bm25_retrieved}")

    # Evaluate
    k = 3  # We only care about the top 3 results for this benchmark
    print(f"\n=== Evaluation Metrics (@K={k}) ===")

    # TF-IDF Scores
    tfidf_p = precision_at_k(tfidf_retrieved, relevant_ids, k)
    tfidf_r = recall_at_k(tfidf_retrieved, relevant_ids, k)
    tfidf_mrr = mrr(tfidf_retrieved, relevant_ids)
    tfidf_ndcg = ndcg_at_k(tfidf_retrieved, ground_truth_scores, k)

    # BM25 Scores
    bm25_p = precision_at_k(bm25_retrieved, relevant_ids, k)
    bm25_r = recall_at_k(bm25_retrieved, relevant_ids, k)
    bm25_mrr = mrr(bm25_retrieved, relevant_ids)
    bm25_ndcg = ndcg_at_k(bm25_retrieved, ground_truth_scores, k)

    # Print side-by-side
    print(f"{'Metric':<15} | {'TF-IDF':<10} | {'BM25':<10}")
    print("-" * 40)
    print(f"{'Precision@' + str(k):<15} | {tfidf_p:<10.4f} | {bm25_p:<10.4f}")
    print(f"{'Recall@' + str(k):<15} | {tfidf_r:<10.4f} | {bm25_r:<10.4f}")
    print(f"{'MRR':<15} | {tfidf_mrr:<10.4f} | {bm25_mrr:<10.4f}")
    print(f"{'NDCG@' + str(k):<15} | {tfidf_ndcg:<10.4f} | {bm25_ndcg:<10.4f}")

    print("\nConclusion:")
    if bm25_ndcg > tfidf_ndcg:
        print(
            "BM25 outperformed TF-IDF! It successfully penalized the keyword-stuffed document (Doc 3)."
        )
    else:
        print("TF-IDF scored higher (or equal). Check the ranking logic!")


if __name__ == "__main__":
    main()
