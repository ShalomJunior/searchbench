# SearchBench 🔍

SearchBench is a multi-stage, neural-first Information Retrieval (IR) pipeline built in Python from scratch. It is designed to evaluate and benchmark the quality and latency trade-offs between classical lexical search, dense semantic search, hybrid rank fusion, and cross-encoder re-ranking on scientific text.

## 🏗️ Architecture

The system implements a state-of-the-art **Retrieve & Re-rank** architecture:

1. **First-Stage Retrieval (High Recall):**
   - **Lexical Engine (BM25):** Handles exact-match keyword searching and entity extraction.
   - **Semantic Engine (Dense):** Uses `BAAI/bge-small-en-v1.5` to capture conceptual intent and synonymy via Approximate Nearest Neighbors (FAISS).
   - **Fusion (RRF):** Reciprocal Rank Fusion merges the lexical and dense lists into a highly robust candidate pool.

2. **Second-Stage Re-ranking (High Precision):**
   - **Cross-Encoder:** Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` to perform deep token-level cross-attention on the top candidates, resolving complex semantic nuances (like negations) that bi-encoders miss.

*(See [docs/architecture.md](docs/architecture.md) for a detailed breakdown and system diagram).*

## 📊 Key Findings

SearchBench was evaluated on the **BEIR SciFact dataset** (5,183 scientific documents, 300 complex queries). 

### 1. Multi-Stage Performance
The integration of semantic models significantly outperforms classical text search:

| System Architecture | NDCG@10 | MRR@10 |
| --- | --- | --- |
| 1. BM25 Baseline | 0.5379 | 0.5105 |
| 2. Dense Baseline (BGE) | 0.7200 | 0.6845 |
| 3. Hybrid Fusion (RRF) | 0.6641 | 0.6234 |
| **4. Hybrid + Re-ranker** | **0.6975** | **0.6720** |

### 2. The Re-ranking "Sweet Spot"
A major engineering challenge in neural search is the latency of Cross-Encoders. Our rigorous depth benchmarking revealed a critical **Quality vs. Latency tradeoff**:
- Reranking the **Top 100** candidates took **~11 seconds** per query on CPU, and actually *degraded* the NDCG due to out-of-domain vocabulary hallucination.
- The absolute optimal sweet spot is reranking exactly the **Top 25** candidates, yielding the maximum NDCG (0.6975) while maintaining an acceptable CPU latency (~2.7 seconds).

*(See [PROJECT_LOG.md](PROJECT_LOG.md) for the complete 4-week engineering diary, deep-dive metric analyses, and qualitative error reports).*

## 🚀 Quickstart

### Installation
Clone the repository and install the required dependencies (Python 3.10+ recommended):

```bash
git clone https://github.com/YOUR_USERNAME/searchbench.git
cd searchbench
pip install -r requirements.txt
```

### Running the Benchmarks
All experiments are localized in the `experiments/` directory. Resulting artifacts (CSVs, JSONs, plots) will be generated in the `results/` folder.

To reproduce the end-to-end multi-stage pipeline evaluation:
```bash
python experiments/benchmark_multistage.py
```

To reproduce the optimal reranking depth analysis:
```bash
python experiments/benchmark_rerank_depth.py
```

## 📖 Documentation
- **[Project Log](PROJECT_LOG.md):** The comprehensive chronological diary of the project's evolution, design choices, and metric analyses.
- **[Architecture](docs/architecture.md):** The system design overview.
