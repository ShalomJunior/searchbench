# SearchBench 🔍

SearchBench is a multi-stage, neural-first Information Retrieval (IR) pipeline built in Python. It is designed to evaluate and benchmark the quality and latency trade-offs between classical lexical search, dense semantic search, hybrid rank fusion, and cross-encoder re-ranking across diverse specialized domains (Science, Finance, Medicine, Social Debates).

## 🏗️ Architecture

The system implements a state-of-the-art **Retrieve & Re-rank** architecture:

1. **First-Stage Retrieval (High Recall):**
   - **Lexical Engine (Elasticsearch BM25):** Handles exact-match keyword searching and entity extraction. Upgraded from native Python dictionaries to a C++/Java inverted index to eliminate massive latency bottlenecks.
   - **Semantic Engine (Dense):** Uses `BAAI/bge-small-en-v1.5` to capture conceptual intent and synonymy via Approximate Nearest Neighbors (FAISS).
   - **Fusion (RRF):** Reciprocal Rank Fusion merges the lexical and dense lists into a highly robust candidate pool.

2. **Second-Stage Re-ranking (High Precision):**
   - **Cross-Encoder:** Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` (both Base and Custom Fine-Tuned versions) to perform deep token-level cross-attention on the top candidates, resolving complex semantic nuances.

3. **Generation (RAG):**
   - **LLM Synthesizer:** Feeds the top-10 verified documents into an LLM (e.g., `Qwen2.5-7B-Instruct`) to generate a direct answer.
   - **Zero-Trust Citations:** Enforces strict formatting where every generated claim must be mathematically linked to a specific source document via `[doc_id]` citations.

_(See [docs/architecture.md](docs/architecture.md) for a detailed breakdown and system diagram)._

## 📊 Key Engineering Findings

SearchBench was evaluated across multiple BEIR datasets: **SciFact** (Science), **FIQA** (Finance), **TREC-COVID** (Medicine), and **ArguAna** (Social Debates).

### 1. The Production Lexical Bottleneck

While Dense Retrieval (FAISS) and Cross-Encoder reranking (GPU) executed in milliseconds, the initial pure Python BM25 baseline became a massive bottleneck (taking **~7,000 ms** per query on TREC-COVID's 171k documents). After integrating a production-grade **Elasticsearch** backend, latency plummeted to **~24 ms**, conclusively proving why production systems rely on dedicated search infrastructure.

### 2. The Power of Fine-Tuning (In-Domain)

By mining Hard Negatives and applying the **InfoNCE Loss**, the Cross-Encoder was fine-tuned on the SciFact dataset. The Fine-Tuned Reranker (**0.7303 NDCG@10**) shattered the Base model baseline (**0.6888 NDCG@10**), proving the absolute necessity of domain-specific contrastive learning for specialized verticals.

### 3. Catastrophic Forgetting (Out-of-Domain)

When testing the SciFact Fine-Tuned model on unknown domains, it suffered severe **Catastrophic Forgetting**:

- **FIQA (Finance):** Base (0.3696) vs. FT (0.3399) — **Degradation**
- **TREC-COVID (Medicine):** Base (0.7387) vs. FT (0.6959) — **Degradation**
- **ArguAna (Social):** Base (0.3092) vs. FT (0.2780) — **Degradation**

The neural network "forgot" how to evaluate general texts, overfitting entirely to the scientific domain.

### 4. RAG Scaling Laws & Hallucination

When augmenting the pipeline with generation, model size heavily dictated grounding adherence. A local 0.5B model completely ignored the strict `[doc_id]` citation prompt (citing correctly in only 4/50 queries) and frequently hallucinated or broke mid-sentence. Upgrading to a 7B model on Kaggle GPUs (via 4-bit quantization) fixed formatting compliance (**40/50** correct citations) and rectified severe logical errors (e.g. negations, proportions), though manual verification of abstentions remains necessary.

_(See [PROJECT_LOG.md](PROJECT_LOG.md) for the complete engineering diary, deep-dive metric analyses, and generated visualization plots)._

## 🚀 Quickstart

### Installation

Clone the repository and install the dependencies (Python 3.10+):

```bash
git clone https://github.com/ShalomJunior/searchbench.git
cd searchbench
pip install -r requirements.txt
```

### Launching Elasticsearch

SearchBench uses Elasticsearch for lightning-fast lexical retrieval.

- **Local (Docker):** `docker-compose up -d`
- **Kaggle:** `!bash notebooks/setup_elasticsearch_kaggle.sh`

### Running the Benchmarks

Evaluate the pipeline dynamically on any BEIR dataset (e.g., `scifact`, `trec-covid`, `fiqa`, `arguana`):

```bash
python experiments/benchmark_multistage.py --dataset trec-covid --use-elastic
```

### RAG Generation (Kaggle / Cloud GPU)

Generate cited answers using an LLM on the retrieved documents. Ensure your environment has a GPU and `bitsandbytes` installed for 4-bit quantization.

```bash
# 1. Pre-compute the retrieval cache
python experiments/generate_rag_cache.py --dataset scifact

# 2. Run the LLM generator
python experiments/run_rag_generation.py --model "Qwen/Qwen2.5-7B-Instruct" --dataset scifact --quantize
```

## 📖 Documentation

- **[Project Log](PROJECT_LOG.md):** The comprehensive chronological diary of the project's evolution, design choices, and metric analyses.
- **[Architecture](docs/architecture.md):** The system design overview.
