# Submission README: `devrev_search_with_v2_ranking.ipynb`

## Overview

This submission is based on `devrev_search_with_v2_ranking.ipynb`. It implements a hybrid retrieval pipeline for the `devrev/search` benchmark using:

- Sparse retrieval with BM25
- Dense retrieval with `BAAI/bge-base-en-v1.5`
- Reciprocal Rank Fusion (RRF) to combine sparse and dense candidates
- Cross-encoder reranking with `BAAI/bge-reranker-v2-m3`

The goal is to improve retrieval quality over the baseline dense-only notebook by combining lexical matching, semantic retrieval, and a final learned reranking stage.

## What The Notebook Does

The notebook runs the full submission pipeline end to end:

1. Loads the three dataset splits from Hugging Face:
   - `annotated_queries`
   - `test_queries`
   - `knowledge_base`
2. Builds an in-memory corpus from the knowledge base.
3. Tokenizes the corpus and creates a BM25 index.
4. Encodes the corpus with `sentence-transformers` and caches dense embeddings to `bge_embeddings.npy`.
5. Builds or reloads a FAISS dense index from `bge_faiss.index`.
6. Retrieves candidates from:
   - BM25
   - dense BGE embeddings
7. Fuses both ranked lists using RRF.
8. Reranks the fused candidates with `BAAI/bge-reranker-v2-m3`.
9. Writes submission outputs to:
   - `test_queries_results_v2.json`
   - `test_queries_results_v2.parquet`
10. Saves reusable retrieval state to `corpus_state_v2.pkl`.

## Key Output Files

- `test_queries_results_v2.json`: submission-ready retrieval output
- `test_queries_results_v2.parquet`: tabular version of the same output
- `bge_embeddings.npy`: cached dense document embeddings
- `bge_faiss.index`: cached FAISS dense index
- `corpus_state_v2.pkl`: cached corpus ID/title/text mappings

## How This Differs From `devrev_search.ipynb`

`devrev_search.ipynb` is the earlier baseline pipeline. The submission notebook differs in several important ways:

### 1. Retrieval strategy

Baseline:
- Dense retrieval only
- One embedding backend at a time (`OpenAI` or `Ollama`)
- Direct nearest-neighbor search over a FAISS index

Submission notebook:
- Hybrid retrieval
- BM25 for lexical matching
- BGE dense embeddings for semantic retrieval
- RRF to merge sparse and dense candidate lists
- Cross-encoder reranking for the final top results

### 2. Models used

Baseline:
- `text-embedding-3-small` through OpenAI, or
- `qwen3-embedding:0.6b` through Ollama

Submission notebook:
- `BAAI/bge-base-en-v1.5` for dense retrieval
- `BAAI/bge-reranker-v2-m3` for reranking

### 3. Indexing setup

Baseline:
- Builds normalized embeddings
- Uses a flat FAISS inner-product index
- Stores artifacts under `faiss_index/` plus `embeddings.npy`

Submission notebook:
- Builds cached BGE embeddings in `bge_embeddings.npy`
- Uses a FAISS IVF index in `bge_faiss.index`
- Saves corpus metadata separately in `corpus_state_v2.pkl`

### 4. Ranking quality

Baseline:
- Final ranking is the FAISS dense retrieval order

Submission notebook:
- Final ranking is produced after:
  - BM25 retrieval
  - dense retrieval
  - RRF fusion
  - cross-encoder reranking

This gives the submission notebook a stronger ranking stack, especially for queries where exact keyword overlap and semantic similarity both matter.

### 5. Output filenames

Baseline:
- `test_queries_results.json`
- `test_queries_results.parquet`

Submission notebook:
- `test_queries_results_v2.json`
- `test_queries_results_v2.parquet`

This keeps the submission artifacts separate from the baseline outputs.

## Running The Submission Notebook

Create and activate the environment, then install the pinned dependencies:

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Launch Jupyter:

```bash
jupyter lab
```

Open `devrev_search_with_v2_ranking.ipynb` and run the cells in order.

## Submission Format

The generated JSON follows the expected benchmark structure:

```json
{
  "query_id": "example-query-id",
  "query": "example query text",
  "retrievals": [
    {
      "id": "document-id",
      "text": "document text",
      "title": "document title"
    }
  ]
}
```

## Notes

- The notebook is designed to reuse cached embeddings and the FAISS index on repeated runs.
- The first full run is heavier because it computes document embeddings and builds the retrieval artifacts.
- The main improvement over the earlier notebook is the ranking pipeline, not just the embedding model choice.
