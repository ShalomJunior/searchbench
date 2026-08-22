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
_Doc 1: "machine learning" (short, 1 hit)_
_Doc 3: "machine learning [10 irrelevant words]" (long, 1 hit)_

- `b=0.00` | Doc 1: 0.2671 | Doc 3: 0.2671
- `b=0.50` | Doc 1: 0.3483 | Doc 3: 0.2226
- `b=0.75` | Doc 1: 0.4109 | Doc 3: 0.2054
- `b=1.00` | Doc 1: 0.5007 | Doc 3: 0.1908

**2. Term Frequency Saturation (b=0.75, varying k1)**
_Doc 1: 1 hit vs Doc 2: 5 hits_

- `k1=0.1` | Doc 1: 0.2820 | Doc 2: 0.2875 _(flatlines instantly)_
- `k1=1.5` | Doc 1: 0.4109 | Doc 2: 0.5039
- `k1=3.0` | Doc 1: 0.4748 | Doc 2: 0.6474
- `k1=10.` | Doc 1: 0.5686 | Doc 2: 0.9277 _(keyword stuffing)_

## 4. Information Retrieval (IR) Metrics

To properly benchmark and evaluate my search engines, I use the following standard industry metrics:

### Precision@K

Measures the proportion of retrieved documents in the top $K$ that are actually relevant.
$$\text{Precision@K} = \frac{| \text{Relevant} \cap \text{Retrieved}_{@K} |}{K}$$

### Recall@K

Measures the proportion of all truly relevant documents that were successfully retrieved in the top $K$.
$$\text{Recall@K} = \frac{| \text{Relevant} \cap \text{Retrieved}_{@K} |}{|\text{Relevant}|}$$

### Reciprocal Rank (RR) & Mean Reciprocal Rank (MRR)

For a _single query_, I calculate the Reciprocal Rank (RR) by looking at how far down the ranked list the _first_ relevant document appears. If the first relevant document is at rank $j$, the RR is $\frac{1}{j}$.
$$\text{RR} = \frac{1}{j}$$

**Mean Reciprocal Rank (MRR)** is simply the average of the RR across an entire dataset of multiple queries ($|Q|$):
$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \text{RR}_i$$

### Normalized Discounted Cumulative Gain (NDCG@K)

Measures the ranking quality by taking into account the _graded relevance_ of documents (e.g., highly relevant=3, somewhat relevant=1) and penalizing relevant documents that appear lower in the list using a logarithmic discount. This uses the industry-standard exponential formulation.
$$\text{DCG@K} = \sum_{i=1}^{K} \frac{2^{rel_i} - 1}{\log_2(i + 1)}$$
$$\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$
_(Where IDCG is the Ideal DCG, obtained by sorting all documents by their true relevance score)._

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

| Metric         | Score    |
| -------------- | -------- |
| **Recall@10**  | `0.6446` |
| **Recall@100** | `0.7894` |
| **MRR@10**     | `0.5105` |
| **NDCG@10**    | `0.5379` |

### Observations

1. **High MRR**: An MRR@10 of `0.5105` indicates that, on average, the very first relevant scientific document is placed at Rank 2. For a purely lexical search engine operating on complex scientific text, this is exceptionally high.
2. **Solid Recall**: Retrieving nearly 79% of all relevant documents in the top 100 results (`Recall@100 = 0.7894`) proves that BM25 is an incredibly strong candidate generator. This means a downstream Neural Reranker will have an excellent pool of candidates to pull from.
3. **Execution Speed**: Indexing over 5,000 documents in ~0.65 seconds proves the underlying data structures are highly optimized for a Python-native implementation.

### Failures

1. **Vocabulary Mismatch**: Because the BM25 engine relies on exact keyword matching, it completely fails when a query uses synonyms (e.g., "AI" vs "Artificial Intelligence").
2. **No Semantic Understanding**: The engine does not understand the _context_ of the scientific abstracts. It simply looks for term frequency, which can lead to "keyword soup" documents ranking artificially high if they happen to contain the query terms out of context.

## 6. The Theory of Dense Retrieval

BM25 is a sparse, lexical engine: it only works if the exact keywords overlap. If a user searches for "automobile" and the document only says "car", BM25 will score it a flat zero.

Dense Retrieval fundamentally solves this by mapping text into a mathematical space where distance represents _semantic meaning_.

### Sentence Embeddings & Bi-Encoders

Instead of counting words, we pass our text through a Transformer neural network (like BERT or MiniLM). The network reads the entire string, understands the context, and outputs a single **dense vector** (an array of e.g., 384 floating-point numbers).

In a **Bi-Encoder** architecture, the query and the document are processed completely independently of each other.
This independent processing is the architectural secret that makes semantic search possible at scale: we can pre-compute the embeddings for all 5,000 (or 5 million) documents _offline_ and cache them in memory. At search time, we only have to run the neural network once for the user's query.

### Cosine Similarity

Once the query and documents are transformed into vectors in the same 384-dimensional space, we can measure how closely related they are by calculating the angle between them using **Cosine Similarity**:

$$\text{Cosine Similarity} = \frac{\mathbf{q} \cdot \mathbf{d}}{\Vert{}\mathbf{q}\Vert{} \Vert{}\mathbf{d}\Vert{}}$$

If we $L_2$-normalize the vectors beforehand, their magnitudes ($\Vert{}\mathbf{q}\Vert{}$ and $\Vert{}\mathbf{d}\Vert{}$) become $1$. This mathematically simplifies the cosine similarity down to a blazing-fast **Inner Product** (Dot Product):

$$\text{Inner Product} = \mathbf{q} \cdot \mathbf{d} = \sum_{i=1}^{n} q_i d_i$$

### Vector Search (FAISS)

Even with the math simplified to a dot product, calculating the distance between the query vector and _every single document vector in a 100-million document corpus_ at query time is computationally impossible.

We solve this using **Approximate Nearest Neighbors (ANN)** libraries like Facebook AI Similarity Search (**FAISS**). FAISS organizes the high-dimensional space into clusters (like Voronoi cells). Instead of comparing the query to every document, FAISS figures out which cluster the query vector lands in, and only computes the dot product against the documents inside that specific neighborhood, bringing the search time down from $O(N)$ to $O(\log N)$.

## 7. The Theory of Approximate Nearest Neighbors (ANN)

Dense retrieval calculates the Cosine Similarity (Inner Product) between a query vector and document vectors. However, comparing a single query vector against 100 million document vectors at query time (Exact Search) is computationally impossible for a low-latency web application. 

This introduces the need for **Approximate Nearest Neighbors (ANN)**, where we intentionally sacrifice a tiny fraction of Recall (accuracy) to gain a massive speedup in Latency.

### Exact Search vs. Approximate Search
- **Exact Nearest Neighbor (Flat Index)**: Computes the distance between the query and *every single document*. 
  - *Pros*: Guarantees finding the absolute best matches (100% Recall).
  - *Cons*: Scales linearly $O(N)$. At millions of documents, latency becomes unacceptable.
- **Approximate Nearest Neighbor (ANN)**: Uses clever data structures to only compare the query against a small "neighborhood" of highly likely candidates.
  - *Pros*: Scales logarithmically $O(\log N)$ or sub-linearly. Enables millisecond latency on billions of documents.
  - *Cons*: Might miss the true best match if it falls outside the probed neighborhood.

### Core ANN Algorithms

#### 1. Inverted File Index (IVF)
IVF solves the scaling problem by clustering the vector space.
- **How it works**: During indexing, IVF runs K-Means clustering to partition the vector space into $V$ clusters (Voronoi cells). Each cluster has a centroid.
- **Search**: Instead of scanning all documents, the query is compared only to the $V$ centroids. Once the closest centroid is found, the system only computes exact distances for the documents *inside that specific cluster*.
- **Tradeoff**: You can increase Recall by probing multiple nearby clusters (increasing `nprobe`), but this directly increases Latency.

#### 2. Hierarchical Navigable Small World (HNSW)
HNSW solves the scaling problem using a multi-layered graph.
- **How it works**: HNSW builds a skip-list-like graph structure. The top layer has very few, long-distance connections (highways). As you move down the layers, the graph becomes denser with local connections (city streets).
- **Search**: A query enters the top layer, rapidly jumping across long distances to find the general neighborhood, then drops down to lower layers to fine-tune the search among local neighbors.
- **Tradeoff**: HNSW provides incredibly fast search latency and extremely high Recall, but building the graph during indexing is very slow and consumes a massive amount of RAM compared to IVF.

### The Recall vs. Latency Tradeoff
In system design, ANN forces a strict engineering tradeoff:
- If you optimize strictly for **Recall**, you probe more clusters (IVF) or search deeper in the graph (HNSW), which drives up **Latency**.
- If you optimize strictly for **Latency**, you probe fewer clusters, but you risk missing the true nearest neighbors, dropping your **Recall**.
- **Memory** is the hidden third variable: HNSW is fast and accurate but requires expensive, memory-heavy servers to hold the graph.

## 8. Experiment 002: Dense Retrieval Baseline

### Overview

The goal of this experiment is to establish a semantic search baseline using a neural bi-encoder model (`all-MiniLM-L6-v2`) and FAISS. Dense retrieval maps documents and queries into a continuous vector space where distance represents semantic similarity, overcoming BM25's vocabulary mismatch limitation.

### Setup

- **Dataset**: BEIR - SciFact (5,183 scientific documents, 300 queries)
- **Method**: Bi-Encoder Semantic Search
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Index**: `faiss.IndexFlatIP` (Cosine Similarity)

### Metrics

| Metric         | Score    | vs BM25 |
| -------------- | -------- | ------- |
| **Recall@100** | `0.9250` | +0.1356 |
| **MRR@10**     | `0.6110` | +0.1005 |
| **NDCG@10**    | `0.6451` | +0.1072 |

### Observations

1. **Massive Quality Increase**: Dense retrieval drastically outperformed raw BM25 across every single metric. Recall@100 jumped from 78.9% to an incredible 92.5%, proving that semantic search is significantly better at finding relevant scientific documents even when exact keywords are missing.
2. **Computational Expense**: Encoding the 5,000 document corpus on a CPU took nearly 9 minutes, compared to BM25's 0.65 seconds. However, once the index was built, searching all 300 queries using FAISS took only ~10 seconds. This highlights the architectural necessity of pre-computing embeddings offline.
3. **The Semantic Advantage**: The model successfully bridged the vocabulary gap. A query searching for "neural networks" successfully retrieved documents discussing "deep learning" because the model understood they exist in the same semantic space.
