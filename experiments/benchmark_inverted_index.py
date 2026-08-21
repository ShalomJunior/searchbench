import time
import random
from collections import defaultdict

def generate_synthetic_corpus(num_docs: int = 100000) -> dict[int, str]:
    """Generates a large dictionary of random documents."""
    print(f"Generating corpus of {num_docs} documents...")
    vocab = [
        "machine", "learning", "python", "data", "science", "algorithm", "cloud", "computing",
        "artificial", "intelligence", "neural", "network", "deep", "reinforcement", "supervised",
        "unsupervised", "model", "training", "evaluation", "metrics", "accuracy", "precision",
        "recall", "f1", "tensor", "gradient", "descent", "optimization", "loss", "function"
    ]
    corpus = {}
    for i in range(num_docs):
        # Create random documents of 10-20 words each
        corpus[i] = " ".join(random.choices(vocab, k=random.randint(10, 20)))
    return corpus

def build_index(corpus: dict[int, str]) -> defaultdict[str, list[int]]:
    """Builds the inverted index from the corpus."""
    print("Building inverted index...")
    index = defaultdict(list)
    for doc_id, text in corpus.items():
        for word in set(text.split()):
            index[word].append(doc_id)
    return index

def naive_search(query: str, corpus: dict[int, str]) -> list[int]:
    """O(N) linear scan across all documents."""
    results = []
    query_terms = query.split()
    for doc_id, text in corpus.items():
        doc_tokens = text.split()
        if any(term in doc_tokens for term in query_terms):
            results.append(doc_id)
    return results

def indexed_search(query: str, index: defaultdict[str, list[int]]) -> list[int]:
    """O(1) dictionary lookups to find candidate documents."""
    query_terms = query.split()
    candidate_ids = set()
    for term in query_terms:
        if term in index:
            candidate_ids.update(index[term])
    return list(candidate_ids)

def run_benchmark() -> None:
    """Executes the benchmark and profiles the runtime."""
    # 1. Setup Data
    corpus = generate_synthetic_corpus(100000)
    
    # 2. Pre-computation phase
    start_time = time.time()
    index = build_index(corpus)
    print(f"Index built in {time.time() - start_time:.4f} seconds\n")
    
    query = "machine learning"
    
    # 3. Profile Naive Search
    print("--- Running Naive Search ---")
    start_time = time.time()
    naive_results = naive_search(query, corpus)
    naive_time = time.time() - start_time
    print(f"Found {len(naive_results)} candidates in {naive_time:.4f} seconds\n")
    
    # 4. Profile Indexed Search
    print("--- Running Indexed Search ---")
    start_time = time.time()
    indexed_results = indexed_search(query, index)
    indexed_time = time.time() - start_time
    print(f"Found {len(indexed_results)} candidates in {indexed_time:.4f} seconds\n")
    
    # 5. Calculate Speedup
    # Adding a tiny epsilon (1e-9) to prevent division by zero
    speedup = naive_time / (indexed_time + 1e-9)
    print(f"Speedup: The inverted index is {speedup:.2f}x faster.")


def main() -> None:
    run_benchmark()


if __name__ == "__main__":
    main()