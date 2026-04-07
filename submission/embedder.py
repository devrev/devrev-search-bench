# =============================================================================
# embedder.py — Qwen3-Embedding-0.6B via sentence-transformers
# =============================================================================
"""
Wraps Qwen/Qwen3-Embedding-0.6B using the sentence-transformers library.

Key Qwen3 embedding specifics (from model card):
  • Documents  → plain encode, no prefix.
  • Queries    → use prompt_name="query" (stored in model.prompts).
                 Internally this prepends:
                 'Instruct: <task>\\nQuery: '
  • Embeddings are L2-normalised → cosine sim = inner product.
  • left-padding is required (set in tokenizer_kwargs).

Public API
----------
    from embedder import Embedder

    emb = Embedder()                        # loads model once
    doc_vecs = emb.encode_documents(texts)  # np.ndarray (N, 1024)
    q_vec    = emb.encode_query(query)      # np.ndarray (1024,)
    q_vecs   = emb.encode_queries(queries)  # np.ndarray (N, 1024)

    # Corpus embedding with checkpoint-aware caching:
    vecs = emb.compute_corpus_embeddings(documents, ck)
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

from checkpoint import Checkpoint
from config import (
    EMBED_MODEL_ID,
    EMBED_DIM,
    EMBED_BATCH_SIZE,
    EMBED_MAX_LENGTH,
    HF_TOKEN,
)
from logger import get_logger

log = get_logger(__name__)

_EMBED_CACHE = Path("checkpoints/corpus_embeddings.npy")


class Embedder:
    """
    Thin wrapper around SentenceTransformer for Qwen3-Embedding-0.6B.

    The model is loaded lazily on first use and cached as a class-level
    singleton so multiple callers in the same process share one copy.
    """

    _instance: SentenceTransformer | None = None

    def __init__(self) -> None:
        self._model = self._load_model()

    @classmethod
    def _load_model(cls) -> SentenceTransformer:
        if cls._instance is None:
            log.info("Loading embedding model: %s …", EMBED_MODEL_ID)
            cls._instance = SentenceTransformer(
                EMBED_MODEL_ID,
                tokenizer_kwargs={"padding_side": "left"},  # required by Qwen3
                trust_remote_code=True,
                token=HF_TOKEN,
            )
            cls._instance.max_seq_length = EMBED_MAX_LENGTH
            log.info(
                "Embedding model loaded  dim=%d  max_seq_len=%d",
                EMBED_DIM,
                EMBED_MAX_LENGTH,
            )
        return cls._instance

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of document texts (no instruction prefix).
        Returns float32 ndarray of shape (N, EMBED_DIM), L2-normalised.
        """
        log.debug(
            "encode_documents: %d texts, batch_size=%d", len(texts), EMBED_BATCH_SIZE
        )
        vecs = self._model.encode(
            texts,
            batch_size=EMBED_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        arr = np.array(vecs, dtype=np.float32)
        log.debug("encode_documents result shape: %s", arr.shape)
        return arr

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        """
        Embed a list of queries using Qwen3's built-in 'query' prompt.
        Returns float32 ndarray of shape (N, EMBED_DIM), L2-normalised.
        """
        log.debug("encode_queries: %d queries", len(queries))
        vecs = self._model.encode(
            queries,
            batch_size=EMBED_BATCH_SIZE,
            prompt_name="query",  # Qwen3 instruction-aware prompt
            normalize_embeddings=True,
            show_progress_bar=len(queries) > 10,
            convert_to_numpy=True,
        )
        arr = np.array(vecs, dtype=np.float32)
        log.debug("encode_queries result shape: %s", arr.shape)
        return arr

    def encode_query(self, query: str) -> np.ndarray:
        """Single-query convenience wrapper. Returns shape (EMBED_DIM,)."""
        return self.encode_queries([query])[0]

    def compute_corpus_embeddings(
        self,
        documents: list[str],
        ck: Checkpoint,
    ) -> np.ndarray:
        """
        Embed all corpus documents, caching the result to disk.

        If the checkpoint phase 'embeddings_computed' is already done and the
        .npy cache exists, the file is loaded directly — no GPU work needed.

        Parameters
        ----------
        documents : list[str]
            Title-prepended document texts (``"<title>\\n\\n<text>"``).
        ck : Checkpoint
            Shared pipeline checkpoint.

        Returns
        -------
        np.ndarray of shape (N, EMBED_DIM), float32
        """
        if ck.done("embeddings_computed") and _EMBED_CACHE.exists():
            log.info("Phase 'embeddings_computed' done — loading from %s", _EMBED_CACHE)
            embs = np.load(_EMBED_CACHE)
            log.info("Loaded corpus embeddings: shape=%s", embs.shape)
            return embs

        log.info("=== Phase: embeddings_computed ===")
        log.info("Embedding %d documents with %s …", len(documents), EMBED_MODEL_ID)

        embs = self.encode_documents(documents)

        _EMBED_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.save(_EMBED_CACHE, embs)
        log.info("Corpus embeddings saved → %s  shape=%s", _EMBED_CACHE, embs.shape)

        ck.mark_done(
            "embeddings_computed",
            embed_model=EMBED_MODEL_ID,
            embed_shape=list(embs.shape),
        )
        return embs
