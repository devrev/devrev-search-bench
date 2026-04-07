# =============================================================================
# config.py — central configuration for DevRev hybrid search
# =============================================================================
from __future__ import annotations
import os
from pathlib import Path

# Set via env-var:  export HF_TOKEN="hf_..."
# Or pass directly to the pipeline via --hf-token CLI arg.
HF_TOKEN: str | None = os.getenv("HF_TOKEN")

# Embedder: Qwen3-Embedding-0.6B
#   • sentence-transformers compatible
#   • 1024-d, 32k context, instruction-aware
#   • Use prompt_name="query" for queries; plain encode for documents
EMBED_MODEL_ID   = "Qwen/Qwen3-Embedding-0.6B"
EMBED_DIM        = 1024
EMBED_BATCH_SIZE = 32       # lower to 8-16 if GPU < 8 GB VRAM
EMBED_MAX_LENGTH = 512      # practical cap; model supports 32k but slow on CPU

# Reranker: Qwen3-Reranker-0.6B (matching family, instruction-aware)
RERANKER_MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
RERANKER_MAX_LEN  = 512

QDRANT_PATH       = "qdrant_storage"
QDRANT_COLLECTION = "devrev_kb"

BM25_CACHE_PATH   = "checkpoints/bm25_index.pkl"

DENSE_TOP_K   = 50    # candidates pulled from Qdrant per query
SPARSE_TOP_K  = 50    # candidates pulled from BM25 per query
RRF_K         = 60    # RRF constant (standard; higher → less top-rank emphasis)
RERANK_POOL   = 100   # top-N from RRF sent to cross-encoder
FINAL_TOP_K   = 10    # results returned per query

HF_DATASET_ID  = "devrev/search"
DATA_DIR       = Path("data")
RESULTS_DIR    = Path("results")

EVAL_K = 10   # Recall@K and Precision@K

CHECKPOINT_FILE = Path("checkpoints/pipeline_state.json")

# Ordered phases — each is marked done atomically after it completes.
# Re-running the pipeline skips any phase already marked done.
PHASES = [
    "datasets_downloaded",   # HF → local parquet
    "embeddings_computed",   # corpus → .npy cache
    "qdrant_indexed",        # .npy → Qdrant collection
    "bm25_indexed",          # corpus → BM25 pickle
    "test_results_saved",    # test queries → results JSON
]

LOG_DIR   = Path("logs")
LOG_LEVEL = "DEBUG"   # DEBUG | INFO | WARNING
