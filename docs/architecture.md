# SearchBench: System Architecture

## Overview

SearchBench is a multi-stage, neural-first information retrieval pipeline. It is designed to evaluate the quality and latency trade-offs between classical lexical search, dense semantic search, hybrid rank fusion, and cross-encoder re-ranking.

## Pipeline Architecture

```text
                      [ User Query ]
                            │
                            ▼
                  ┌───────────────────┐
                  │  Query Processing │
                  └─────────┬─────────┘
                            │
             ┌──────────────┴──────────────┐
             │                             │
             ▼                             ▼
    ┌─────────────────┐           ┌─────────────────┐
    │  Elasticsearch  │           │  Dense Engine   │
    │  (Lexical BM25) │           │ (Bi-Encoder ANN)│
    └────────┬────────┘           └────────┬────────┘
             │                             │
             └──────────────┬──────────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │    RRF Fusion     │
                  │   (Rank Merging)  │
                  └─────────┬─────────┘
                            │
                            ▼
                   [ Top 100 Candidates ]
                            │
                            ▼
                  ┌───────────────────┐
                  │   Cross-Encoder   │
                  │   (Re-ranking)    │
                  └─────────┬─────────┘
                            │
                            ▼
                    [ Top 10 Results ]
                            │
                            ▼
                  ┌───────────────────┐
                  │    Evaluation     │
                  │ (NDCG, MRR, Latency)│
                  └───────────────────┘
```

## Component Breakdown

### 1. First-Stage Retrieval (Candidate Generation)

- **Lexical Retrieval (`ElasticBM25Engine`)**: Handles exact-match keyword searching and entity extraction. Initially built using Python dictionaries, this component was upgraded to **Elasticsearch 8.12.0** to leverage C++/Java inverted indices, reducing lexical query latency from 7,000+ ms down to ~24 ms on production-scale corpora.
- **Semantic Retrieval (`DenseEngine`)**: Uses a modern bi-encoder (`BAAI/bge-small-en-v1.5`) to map queries and documents into a dense vector space, capturing conceptual intent and synonymy. Search is accelerated via vector similarity algorithms (FAISS-like abstractions).

### 2. Hybrid Fusion

- **Reciprocal Rank Fusion (`RRFHybridEngine`)**: Merges the ranked lists from the lexical and dense engines using their rank positions ($k=60$) rather than raw scores. This smooths out differences in score distributions and provides a highly robust candidate pool.

### 3. Second-Stage Re-ranking (Top 100 Candidates)

- **Neural Re-ranker (`CrossEncoderReRanker`)**: Passes the query and the **Top 100** candidate documents simultaneously through a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`). This allows for full cross-attention between every query token and document token, resolving complex semantic nuances.
- **Domain-Specific Fine-Tuning**: Because the base cross-encoder was trained on MS-MARCO (Bing search logs), it suffers from performance degradation when ranking highly specialized scientific text. To fix this, SearchBench integrates a pipeline for **Hard Negative Mining** and trains the cross-encoder via **InfoNCE Loss** on target domains, creating highly specialized "Fine-Tuned" models (e.g., optimized for SciFact).

_Engineering Note:_ Initial CPU testing indicated an optimal reranking depth of 25 due to hardware limitations (taking ~2.7s). However, moving the pipeline to a GPU accelerator (T4) allowed for reranking depth to be increased to **100 candidates**, completing the forward passes in under 500 ms while maximizing retrieval potential.
