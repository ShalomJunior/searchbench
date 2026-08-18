import math
from collections import defaultdict


class BM25Engine:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.index = defaultdict(list)
        self.doc_lengths = {}
        self.avgdl = 0.0
        self.total_docs = 0
        self.idf_cache = {}

    def fit(self, corpus):
        """
        TODO: Implement the indexing phase.
        1. Populate self.index with the inverted index.
        2. Populate self.doc_lengths with the length of each document.
        3. Calculate self.avgdl (average document length).
        4. Set self.total_docs.
        """
        for doc_id, doc in corpus.items():
            words = doc.split()
            self.doc_lengths[doc_id] = len(words)
            self.avgdl += len(words)
            for text in set(words):
                self.index[text].append(doc_id)
        self.total_docs = len(corpus)
        self.avgdl /= self.total_docs

    def compute_idf(self, term):
        """
        TODO: Implement the IDF calculation for a single term.
        Use the standard formula: math.log( (N - df + 0.5) / (df + 0.5) + 1 )
        where N is total_docs and df is the number of documents containing the term.
        """
        if term not in self.idf_cache:
            N = self.total_docs
            df = len(self.index[term]) if term in self.index else 0
            self.idf_cache[term] = math.log(1 + (N - df + 0.5) / (df + 0.5))
        return self.idf_cache[term]

    def score(self, query_tokens, doc_id, doc_tokens):
        """
        TODO: Implement the BM25 mathematical formula provided above.
        Return the final float score for this document against the query.
        """
        score = 0.0
        doc_len = self.doc_lengths[doc_id]
        for term in query_tokens:
            tf = doc_tokens.count(term)
            if tf > 0:
                idf = self.compute_idf(term)
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                score += idf * (num / den)
        return score

    def search(self, query, corpus):
        """
        Executes a search query against the corpus using BM25 ranking.
        Uses the inverted index for O(1) candidate filtering.
        """
        query_tokens = query.split()

        candidates = set()
        for term in query_tokens:
            if term in self.index:
                candidates.update(self.index[term])

        if not candidates:
            return []

        scores = {}
        for doc_id in candidates:
            doc_tokens = corpus[doc_id].split()
            scores[doc_id] = self.score(query_tokens, doc_id, doc_tokens)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked


def parameter_experiment():
    print("\n=== BM25 Parameter Experiment ===")
    toy_corpus = {
        1: "machine learning",
        2: "machine learning machine learning machine learning machine learning machine learning",
        3: "machine learning and a bunch of other irrelevant words that make this document very long",
    }
    query = "machine learning"

    print("\n--- Testing Length Normalization (b) ---")
    print("k1=1.5 (fixed). 'b' controls how much document length penalizes the score.")
    print("Doc 1: short (1 hit) vs Doc 3: long (1 hit)")
    for b in [0.0, 0.5, 0.75, 1.0]:
        engine = BM25Engine(k1=1.5, b=b)
        engine.fit(toy_corpus)
        res1 = engine.score(query.split(), 1, toy_corpus[1].split())
        res3 = engine.score(query.split(), 3, toy_corpus[3].split())
        print(f"b={b:<4} | Doc 1 (short): {res1:.4f} | Doc 3 (long): {res3:.4f}")

    print("\n--- Testing Term Frequency Saturation (k1) ---")
    print("b=0.75 (fixed). 'k1' controls how much extra hits increase the score.")
    print("Doc 1: 1 hit vs Doc 2: 5 hits")
    for k1 in [0.1, 1.5, 3.0, 10.0]:
        engine = BM25Engine(k1=k1, b=0.75)
        engine.fit(toy_corpus)
        res1 = engine.score(query.split(), 1, toy_corpus[1].split())
        res2 = engine.score(query.split(), 2, toy_corpus[2].split())
        print(f"k1={k1:<4} | Doc 1 (1 hit): {res1:.4f} | Doc 2 (5 hits): {res2:.4f}")


def main():
    parameter_experiment()


if __name__ == "__main__":
    main()
