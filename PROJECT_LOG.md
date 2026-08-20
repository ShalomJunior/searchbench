# Project Log

## What is Information Retrieval?

At its core, Information Retrieval (IR) is the science of bridging the gap between a user's unstated intent and a massive corpus of unstructured data. It is not just finding words in a text; it is about surfacing the most valuable information as efficiently as possible.

**Query**: The explicit, often imperfect, expression of a user's information need.

**Document**: The fundamental unit of information I am searching through—whether that is a Wikipedia article, a legal contract, or a single parsed paragraph.

**Relevance**: The ultimate metric of success. It defines how accurately a retrieved document satisfies the true intent behind the query.

**Retrieval**: The coarse-grained, highly optimized filter. It is the process of rapidly reducing a corpus of millions of documents down to a candidate pool of a few hundred (the "Top-K") without burning excessive compute.

**Ranking**: The fine-grained, computationally expensive sorting mechanism. It takes the candidate pool from the retrieval phase and orders it so that the most relevant documents appear at the absolute top.

**Evaluation**: The empirical proof of the system's quality. It involves using strict mathematical metrics to prove that the retrieval and ranking algorithms actually improve the user experience.

## 1. The Mathematics of TF-IDF

### Term Frequency (TF)

This answers the question: How often does the search term appear in this specific document? If a document mentions "algorithm" 10 times, it is likely more relevant than a document that mentions it once.

$$TF(t, d) = \frac{\text{count of term } t \text{ in document } d}{\text{total terms in document } d}$$

### Inverse Document Frequency (IDF)

This answers the question: How rare is this term across the entire dataset? Words like "the" or "is" will have a high TF but are useless for search. Rare words like "backpropagation" carry high information value. I use a logarithm to heavily penalize common words and boost rare ones.

$$IDF(t) = \log\left(\frac{N}{df_t}\right)$$

_(Where $N$ is the total number of documents, and $df_t$ is the number of documents containing the term $t$)_

### TF-IDF Score

I simply multiply them together. A term gets a high score if it appears frequently in a specific document but rarely across the whole corpus.

$$TF\text{-}IDF(t, d) = TF(t, d) \times IDF(t)$$

### Cosine Similarity

To search, I treat the query and every document as mathematical vectors in a high-dimensional space (where every unique word in the corpus is a dimension). I then measure the angle between the query vector ($\mathbf{q}$) and the document vector ($\mathbf{d}$).

$$\text{Cosine Similarity} = \frac{\mathbf{q} \cdot \mathbf{d}}{\Vert{}\mathbf{q}\Vert{} \Vert{}\mathbf{d}\Vert{}}$$

## 2. Benchmark: Naive vs Inverted Index

To prove why an inverted index is strictly necessary for web-scale retrieval, I benchmarked an $O(N)$ linear scan against an $O(1)$ inverted index lookup over a synthetic corpus of 100,000 documents.

### Algorithmic Complexity

- **Naive Search**: $O(|V_{query}| \times \sum_{i=1}^{N} |D_i|)$. For every query term, the engine must scan the entire length of every single document in the corpus.
- **Inverted Index Precompute**: $O(\sum_{i=1}^{N} |D_i|)$. The index must read every document once to build the mapping of `term -> posting list`. This is an expensive, one-time offline cost.
- **Indexed Search**: $O(|V_{query}| + |Candidates|)$. The engine does $O(1)$ dictionary lookups to find the candidate documents, meaning the search time only depends on the size of the query and the number of matching documents, _not_ the total size of the corpus.

### Results (100,000 documents)

With an augmented vocabulary of 30 words, the inverted index successfully filtered out non-relevant documents and delivered a massive speedup.

- **Naive Search Time**: ~0.2120 seconds
- **Indexed Search Time**: ~0.0080 seconds
- **Speedup**: ~26.49x faster

**Scaling Implications**:
While my toy benchmark showed a ~26x speedup, in a real-world system with millions of documents and a massive vocabulary, a specific query term only hits a tiny fraction of the corpus. Because the naive search time grows linearly with the entire corpus size $O(N)$ while the indexed search scales only with the number of candidate matches $O(|Candidates|)$, the true speedup multiplier can easily reach $10^4$ or $10^5$ (10,000x to 100,000x faster).

## 3. The Mathematics of BM25

BM25 improves upon standard TF-IDF by introducing two critical parameters: `b` (length normalization) and `k1` (term frequency saturation).

$$\text{Score}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{TF(t, d) \cdot (k_1 + 1)}{TF(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{\vert{}d\vert{}}{\text{avgdl}}\right)}$$

### Length Normalization (`b`)
The `b` parameter controls how much document length penalizes the score (typically set to 0.75).
- If `b = 0.0`, length normalization is completely turned off. A 10,000-word document gets the exact same score as a 2-word document if they both contain the query term exactly once.
- As `b` approaches `1.0`, long documents are heavily penalized for having the query term "diluted" among thousands of irrelevant words, while extremely short documents are heavily rewarded.

### Term Frequency Saturation (`k1`)
The `k1` parameter controls how much extra occurrences of a term increase the score (typically set between 1.2 and 2.0).
- If `k1` is very low (e.g., `0.1`), finding a keyword 5 times gives almost zero extra benefit over finding it just once. The term frequency reward flatlines instantly.
- If `k1` is very high (e.g., `10.0`), the score climbs significantly and almost linearly for repeated words, which makes the algorithm highly susceptible to keyword stuffing.

### Experimental Proof
I ran an empirical parameter experiment to observe these constraints in action. For a query of "machine learning":

**1. Length Normalization (k1=1.5, varying b)**
*Doc 1: "machine learning" (short, 1 hit)*
*Doc 3: "machine learning [10 irrelevant words]" (long, 1 hit)*
- `b=0.00` | Doc 1: 0.2671 | Doc 3: 0.2671
- `b=0.50` | Doc 1: 0.3483 | Doc 3: 0.2226
- `b=0.75` | Doc 1: 0.4109 | Doc 3: 0.2054
- `b=1.00` | Doc 1: 0.5007 | Doc 3: 0.1908

**2. Term Frequency Saturation (b=0.75, varying k1)**
*Doc 1: 1 hit vs Doc 2: 5 hits*
- `k1=0.1` | Doc 1: 0.2820 | Doc 2: 0.2875 *(flatlines instantly)*
- `k1=1.5` | Doc 1: 0.4109 | Doc 2: 0.5039
- `k1=3.0` | Doc 1: 0.4748 | Doc 2: 0.6474
- `k1=10.` | Doc 1: 0.5686 | Doc 2: 0.9277 *(keyword stuffing)*

## 4. Information Retrieval (IR) Metrics

To properly benchmark and evaluate my search engines, I use the following standard industry metrics:

### Precision@K
Measures the proportion of retrieved documents in the top $K$ that are actually relevant.
$$\text{Precision@K} = \frac{| \text{Relevant} \cap \text{Retrieved}_{@K} |}{K}$$

### Recall@K
Measures the proportion of all truly relevant documents that were successfully retrieved in the top $K$.
$$\text{Recall@K} = \frac{| \text{Relevant} \cap \text{Retrieved}_{@K} |}{|\text{Relevant}|}$$

### Reciprocal Rank (RR) & Mean Reciprocal Rank (MRR)
For a *single query*, I calculate the Reciprocal Rank (RR) by looking at how far down the ranked list the *first* relevant document appears. If the first relevant document is at rank $j$, the RR is $\frac{1}{j}$.
$$\text{RR} = \frac{1}{j}$$

**Mean Reciprocal Rank (MRR)** is simply the average of the RR across an entire dataset of multiple queries ($|Q|$):
$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \text{RR}_i$$

### Normalized Discounted Cumulative Gain (NDCG@K)
Measures the ranking quality by taking into account the *graded relevance* of documents (e.g., highly relevant=3, somewhat relevant=1) and penalizing relevant documents that appear lower in the list using a logarithmic discount. This uses the industry-standard exponential formulation.
$$\text{DCG@K} = \sum_{i=1}^{K} \frac{2^{rel_i} - 1}{\log_2(i + 1)}$$
$$\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$
*(Where IDCG is the Ideal DCG, obtained by sorting all documents by their true relevance score).*

## 5. Experiment 001: BM25 Lexical Baseline

### Overview
The goal of this experiment is to establish a strong, non-neural lexical baseline using a raw BM25 algorithm. This baseline will be used to measure the efficacy of future Dense Retrieval, Hybrid Fusion, and Neural Reranking implementations.

### Setup
- **Dataset**: BEIR - SciFact (5,183 scientific documents, 300 queries)
- **Method**: Raw BM25 (No tokenizer, no stopword removal, no stemming)
- **Parameters**: 
  - `k1` (Term frequency saturation) = `1.5`
  - `b` (Length normalization) = `0.75`

### Metrics
| Metric | Score |
|--------|-------|
| **Recall@10** | `0.6446` |
| **Recall@100** | `0.7894` |
| **MRR@10** | `0.5105` |
| **NDCG@10** | `0.5379` |

### Observations
1. **High MRR**: An MRR@10 of `0.5105` indicates that, on average, the very first relevant scientific document is placed at Rank 2. For a purely lexical search engine operating on complex scientific text, this is exceptionally high.
2. **Solid Recall**: Retrieving nearly 79% of all relevant documents in the top 100 results (`Recall@100 = 0.7894`) proves that BM25 is an incredibly strong candidate generator. This means a downstream Neural Reranker will have an excellent pool of candidates to pull from.
3. **Execution Speed**: Indexing over 5,000 documents in ~0.65 seconds proves the underlying data structures are highly optimized for a Python-native implementation.

### Failures
1. **Vocabulary Mismatch**: Because the BM25 engine relies on exact keyword matching, it completely fails when a query uses synonyms (e.g., "AI" vs "Artificial Intelligence").
2. **No Semantic Understanding**: The engine does not understand the *context* of the scientific abstracts. It simply looks for term frequency, which can lead to "keyword soup" documents ranking artificially high if they happen to contain the query terms out of context.
