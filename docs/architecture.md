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
    │   BM25 Engine   │           │  Dense Engine   │
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
                    [ Top 25 Candidates ]
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

- **Lexical Retrieval (`BM25Engine`)**: Handles exact-match keyword searching and entity extraction, ensuring we don't miss documents that perfectly match the search terms.
- **Semantic Retrieval (`DenseEngine`)**: Uses a modern bi-encoder (`BAAI/bge-small-en-v1.5`) to map queries and documents into a dense vector space, capturing conceptual intent and synonymy.

### 2. Hybrid Fusion

- **Reciprocal Rank Fusion (`RRFHybridEngine`)**: Merges the ranked lists from the lexical and dense engines using their rank positions ($k=60$) rather than raw scores. This smooths out differences in score distributions and provides a highly robust candidate pool.

### 3. Second-Stage Re-ranking (Optimal Depth = 25)

- **Neural Re-ranker (`CrossEncoderReRanker`)**: Passes the query and the **Top 25** candidate documents simultaneously through a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`). This allows for full cross-attention between every query token and document token, resolving complex semantic nuances (such as detecting negations like "no involvement").
- _Engineering Note:_ Extensive latency vs. quality benchmarking on SciFact proved that re-ranking exactly 25 candidates provides the absolute maximum NDCG@10 (0.6975) while keeping latency around ~2.7s per query. Passing 100 candidates significantly degrades both quality (domain confusion) and speed (11+ seconds).
