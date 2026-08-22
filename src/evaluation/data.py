import os
from beir import util
from beir.datasets.data_loader import GenericDataLoader

def load_beir_dataset(
    dataset_name: str = "scifact",
) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, dict[str, int]]]:
    """Downloads and loads a BEIR dataset."""
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

def format_beir_corpus(beir_corpus: dict[str, dict[str, str]]) -> dict[str, str]:
    """Flattens BEIR corpus dictionaries into single strings per doc_id."""
    flat_corpus = {}
    for doc_id, doc_data in beir_corpus.items():
        flat_corpus[doc_id] = doc_data.get("title", "") + " " + doc_data.get("text", "")
    return flat_corpus
