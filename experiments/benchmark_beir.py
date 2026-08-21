import os
import sys
import time

from beir import util
from beir.datasets.data_loader import GenericDataLoader

# Ensure we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bm25 import BM25Engine
from src.evaluation.metrics import ndcg_at_k, mrr, recall_at_k

def load_beir_dataset(dataset_name: str = "scifact") -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, dict[str, int]]]:
    """Downloads and loads a BEIR dataset."""
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(project_root, "datasets")

    dataset_path = os.path.join(out_dir, dataset_name)
    
    if not os.path.exists(dataset_path):
        print(f"Downloading and unzipping {dataset_name}...")
        data_path = util.download_and_unzip(url, out_dir)
    else:
        print(f"Dataset {dataset_name} already exists locally. Skipping download.")
        data_path = dataset_path

    print("Loading corpus, queries, and qrels...")
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    return corpus, queries, qrels


def main() -> None:
    # 1. Load the data
    beir_corpus, queries, qrels = load_beir_dataset("scifact")

    # 2. Format the corpus for our BM25Engine
    # BEIR provides dicts like: {"title": "...", "text": "..."}
    # We need to flatten this into a single string per doc_id
    flat_corpus = {}
    for doc_id, doc_data in beir_corpus.items():
        flat_corpus[doc_id] = doc_data.get("title", "") + " " + doc_data.get("text", "")

    print(f"\nLoaded {len(flat_corpus)} documents and {len(queries)} queries.")

    # 3. Initialize and Fit BM25
    engine = BM25Engine(k1=1.5, b=0.75)
    print("Building BM25 Index...")
    start_time = time.time()

    engine.fit(flat_corpus)

    print(f"Index built in {time.time() - start_time:.4f} seconds.")

    # 4. Evaluation Loop
    ndcg_list = []
    mrr_10_list = []
    recall_10_list = []
    recall_100_list = []

    for q_id, query in queries.items():
        # Safety guard: skip queries with no ground truth
        if q_id not in qrels:
            continue
        results = engine.search(query, flat_corpus)
        scores = qrels[q_id]

        retrieved_ids = [doc_id for doc_id, _ in results]
        relevant_ids = [doc_id for doc_id, score in scores.items() if score > 0]

        ndcg_list.append(ndcg_at_k(retrieved_ids, scores, k=10))
        mrr_10_list.append(mrr(retrieved_ids, relevant_ids, k=10))
        recall_10_list.append(recall_at_k(retrieved_ids, relevant_ids, k=10))
        recall_100_list.append(recall_at_k(retrieved_ids, relevant_ids, k=100))

    if ndcg_list:
        print(f"Average NDCG@10:    {sum(ndcg_list) / len(ndcg_list):.4f}")
        print(f"Average MRR@10:     {sum(mrr_10_list) / len(mrr_10_list):.4f}")
        print(f"Average Recall@10:  {sum(recall_10_list) / len(recall_10_list):.4f}")
        print(f"Average Recall@100: {sum(recall_100_list) / len(recall_100_list):.4f}")


if __name__ == "__main__":
    main()
