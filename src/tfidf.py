from collections import defaultdict
from math import log

DocID = str | int

# Toy Corpus
documents: dict[DocID, str] = {
    1: "machine learning is fascinating",
    2: "artificial intelligence and machine learning",
    3: "the history of artificial intelligence",
    4: "learning python for data science",
}


def build_index(corpus: dict[DocID, str]) -> defaultdict[str, list[DocID]]:
    """
    Should return an inverted index and any other metadata (like document lengths)
    needed for scoring.
    """
    index: defaultdict[str, list[DocID]] = defaultdict(list)
    for id, text in corpus.items():
        for word in set(text.split()):
            index[word].append(id)
    return index


def tf(term: str, doc_tokens: list[str]) -> float:
    """Calculate term frequency."""
    if not doc_tokens:
        return 0.0
    return doc_tokens.count(term) / len(doc_tokens)


def idf(term: str, corpus: dict[DocID, str], index: defaultdict[str, list[DocID]]) -> float:
    """Calculate inverse document frequency."""
    if not term in index:
        return 0.0
    return log(len(corpus) / len(index[term]))


def tfidf(term: str, doc_tokens: list[str], corpus: dict[DocID, str], index: defaultdict[str, list[DocID]]) -> float:
    """Calculate the full TF-IDF score."""
    return tf(term, doc_tokens) * idf(term, corpus, index)


def search(query: str, corpus: dict[DocID, str], index: defaultdict[str, list[DocID]]) -> list[tuple[DocID, float]]:
    """
    Process the query, calculate cosine similarity between the query
    and all documents, and return a ranked list of document IDs.
    """
    query_tokens = query.split()
    candidates = set()
    for term in query_tokens:
        if term in index:
            candidates.update(index[term])
    if not candidates:
        return []
    q_emb = {
        term: tfidf(term, query_tokens, corpus, index) for term in set(query_tokens)
    }
    q_norm = sum(x**2 for x in q_emb.values()) ** 0.5
    if q_norm == 0:
        return []

    scores = {}
    for id in candidates:
        doc_tokens = corpus[id].split()
        dot = 0.0
        for term, value in q_emb.items():
            if term in doc_tokens:
                d_term_score = tfidf(term, doc_tokens, corpus, index)
                dot += value * d_term_score

        # NOTE FOR PRODUCTION:
        # Calculating the d_norm inside the search query loop is currently your heaviest remaining
        # operation because it computes TF-IDF for every word in a document at query time.
        # In a real-world engine, the build_index function calculates the d_norm for every document
        # once upfront and stores it in memory (e.g., in a dictionary mapping doc_id to norm),
        # dropping the query execution time to near zero.
        doc_unique_words = set(doc_tokens)
        d_norm = (
            sum(
                tfidf(word, doc_tokens, corpus, index) ** 2 for word in doc_unique_words
            )
            ** 0.5
        )
        if d_norm == 0:
            scores[id] = 0.0
        else:
            scores[id] = dot / (d_norm * q_norm)

    # 5. Rank and return
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked


def main():
    print("Building inverted index...")
    index = build_index(documents)
    print(f"Index built! Vocabulary size: {len(index)}\n")

    queries = ["machine learning", "python data", "artificial intelligence"]

    for q in queries:
        print(f"--- Searching for: '{q}' ---")
        results = search(q, documents, index)

        if not results:
            print("No matching documents found.\n")
        else:
            for id, score in results:
                print(f"[{score:.4f}] Doc {id}: {documents[id]}")
            print()


if __name__ == "__main__":
    main()
