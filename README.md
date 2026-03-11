# DevRev Search — Semantic Search over DevRev Knowledge Base

Semantic search system for the [DevRev Search](https://huggingface.co/datasets/devrev/search) dataset. Embeds ~65K knowledge base articles using either OpenAI `text-embedding-3-small` or Ollama `qwen3-embedding:0.6b`, indexes them with FAISS, and retrieves relevant documents for test queries.

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/<your-username>/devrev-search.git
cd devrev-search
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Choose Embedding Provider

The notebook supports two providers via `EMBEDDING_PROVIDER` in Section 5:
- `openai` (default)
- `ollama` (local open-source model)

#### Option A: OpenAI

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

#### Option B: Ollama (local)

```bash
# Install Ollama first: https://ollama.com/download
ollama pull qwen3-embedding:0.6b
```

Then set `EMBEDDING_PROVIDER = "ollama"` in the notebook config cell.

### 3. Run the Notebook

Open `devrev_search.ipynb` in Jupyter and run cells sequentially:

```bash
jupyter notebook devrev_search.ipynb
```

## Project Structure

```
devrev-search/
├── devrev_search.ipynb      # Main notebook: embed, index, search, evaluate
├── evaluate.py              # IR metrics: NDCG@k, MRR, Recall@k, Precision@k, MAP
├── compare_models.py        # Multi-model benchmark script (Ollama models)
├── download_datasets.py     # Standalone script to download datasets as parquet
├── requirements.txt         # Python dependencies
├── test_queries_results.json # Search results for test queries
└── README.md
```

## What the Notebook Does

| Section | Description                                                                           |
| ------- | ------------------------------------------------------------------------------------- |
| **1–4** | Load & explore the 3 dataset splits (annotated queries, test queries, knowledge base) |
| **5**   | Generate embeddings (OpenAI or Ollama) and build a FAISS index                         |
| **6**   | Interactive search — query the knowledge base                                         |
| **7**   | Run evaluation on all test queries and save results in annotated-queries format       |
| **8**   | Evaluate retrieval quality against annotated queries (NDCG, MRR, Recall, Precision)  |
| **9**   | Load a previously saved index (skip re-embedding)                                     |

## Dataset

The [`devrev/search`](https://huggingface.co/datasets/devrev/search) dataset from Hugging Face contains:

- **`knowledge_base`** — ~65K article chunks from DevRev support docs
- **`annotated_queries`** — Queries paired with golden retrievals (train)
- **`test_queries`** — Held-out queries for evaluation

## Output Format

Results are saved in the same format as `annotated_queries`:

```json
{
  "query_id": "a97f93d2-...",
  "query": "end customer organization name not appearing...",
  "retrievals": [
    {
      "id": "ART-1234_KNOWLEDGE_NODE-5",
      "text": "...",
      "title": "..."
    }
  ]
}
```

## Multi-Model Comparison

Compare multiple Ollama embedding models on the benchmark:

```bash
# Run with default models (qwen3-embedding:0.6b, nomic-embed-text, all-minilm)
python compare_models.py

# Pick specific models
python compare_models.py --models nomic-embed-text mxbai-embed-large snowflake-arctic-embed

# List all supported models
python compare_models.py --list-models
```

Results are saved to `benchmark_results/` including:
- Per-model retrieval results (JSON)
- Cached embeddings (`.npy`, reused on re-runs)
- `leaderboard.csv` with metrics for all models

### Supported Ollama Models

| Model | Dimensions | Size | Notes |
|-------|-----------|------|-------|
| `qwen3-embedding:0.6b` | 1024 | 0.6B | Default in repo |
| `nomic-embed-text` | 768 | 274M | Popular, strong for size |
| `mxbai-embed-large` | 1024 | 670M | High MTEB scores |
| `snowflake-arctic-embed` | 1024 | 335M | Top retrieval performer |
| `all-minilm` | 384 | 46M | Ultra-fast baseline |
| `bge-m3` | 1024 | 1.2G | Multilingual, enterprise-grade |

## Evaluation Metrics

The `evaluate.py` module computes standard IR metrics against the 291 annotated queries:

- **NDCG@k** - Normalized Discounted Cumulative Gain
- **MRR** - Mean Reciprocal Rank
- **MAP** - Mean Average Precision
- **Recall@k** - Fraction of relevant docs found in top-k
- **Precision@k** - Fraction of top-k results that are relevant

## Cost Estimate

If using OpenAI (`text-embedding-3-small`), embedding ~65K documents costs approximately **$0.50–$1.00** (at $0.02 per 1M tokens).  
If using Ollama (`qwen3-embedding:0.6b`), there is no API cost (runs locally).

## License

MIT
