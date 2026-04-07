# =============================================================================
# retriever.py — hybrid retrieval: Dense + BM25 → RRF → cross-encoder rerank
# =============================================================================
"""
Orchestrates the full retrieval pipeline:

    query
      │
      ├─► Qdrant (dense, Qwen3-Embedding)   top DENSE_TOP_K
      ├─► BM25Okapi (sparse)                top SPARSE_TOP_K
      │
      └─► RRF fusion → unified ranked pool
              │
              └─► Qwen3-Reranker-0.6B       top FINAL_TOP_K

Public API
----------
    from retriever import HybridRetriever

    retriever = HybridRetriever(vs, bm25, embedder, corpus_meta, reranker)
    results   = retriever.search(query)

    # results → list of dicts (length FINAL_TOP_K), each:
    # {
    #   "rank": int,
    #   "doc_id": str, "title": str, "text": str,
    #   "rrf_score": float,
    #   "rerank_score": float,   # present only if use_reranker=True
    # }
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from bm25_index import BM25Index
from config import DENSE_TOP_K, SPARSE_TOP_K, RRF_K, RERANK_POOL, FINAL_TOP_K
from embedder import Embedder
from logger import get_logger
from reranker import Reranker
from vector_store import VectorStore

log = get_logger(__name__)


# =============================================================================
# RRF — pure function, no external deps
# =============================================================================
def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """
    Merge multiple ranked lists of document IDs using Reciprocal Rank Fusion.

    Formula
    -------
        score(d) = Σ_i  1 / (k + rank_i(d))

    where rank_i(d) is the 1-based position of document d in list i.
    Documents absent from a list contribute 0 for that list.

    Parameters
    ----------
    ranked_lists : list of ranked doc-ID lists (best first)
    k            : RRF constant (default 60 per the original paper)

    Returns
    -------
    list of (doc_id, rrf_score) sorted descending by score.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] += 1.0 / (k + rank)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    log.debug(
        "RRF: %d lists → %d unique docs  top=%s (%.5f)",
        len(ranked_lists),
        len(fused),
        fused[0][0] if fused else "—",
        fused[0][1] if fused else 0.0,
    )
    return fused


# =============================================================================
# HybridRetriever
# =============================================================================
class HybridRetriever:
    """
    Combines dense (Qdrant) and sparse (BM25) retrieval with RRF fusion
    and optional cross-encoder reranking.

    Parameters
    ----------
    vector_store  : VectorStore   — initialised Qdrant wrapper
    bm25          : BM25Index     — built BM25 index
    embedder      : Embedder      — Qwen3-Embedding-0.6B wrapper
    corpus_meta   : list[dict]    — per-document metadata dicts
                                    must contain "doc_id", "title", "text"
    reranker      : Reranker|None — Qwen3-Reranker-0.6B; pass None to skip
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25: BM25Index,
        embedder: Embedder,
        corpus_meta: list[dict],
        reranker: Reranker | None = None,
    ) -> None:
        self._vs = vector_store
        self._bm25 = bm25
        self._embedder = embedder
        self._reranker = reranker
        # Build a fast doc_id → metadata lookup
        self._meta: dict[str, dict] = {m["doc_id"]: m for m in corpus_meta}
        log.info(
            "HybridRetriever ready  corpus=%d  reranker=%s",
            len(self._meta),
            "enabled" if reranker else "disabled",
        )

    def _dense_search(self, query: str) -> list[str]:
        """Return ordered list of doc_ids from Qdrant dense search."""
        q_vec = self._embedder.encode_query(query)
        hits = self._vs.search(q_vec, top_k=DENSE_TOP_K)
        doc_ids = [h["doc_id"] for h in hits]
        log.debug("Dense: %d candidates", len(doc_ids))
        return doc_ids

    def _sparse_search(self, query: str) -> list[str]:
        """Return ordered list of doc_ids from BM25 sparse search."""
        hits = self._bm25.search(query, top_k=SPARSE_TOP_K)
        doc_ids = [h["doc_id"] for h in hits]
        log.debug("Sparse: %d candidates", len(doc_ids))
        return doc_ids

    def _lookup(self, doc_id: str) -> dict:
        """Return metadata for a doc_id, or a stub dict if not found."""
        meta = self._meta.get(doc_id)
        if meta is None:
            log.warning("doc_id '%s' not found in corpus_meta — using stub", doc_id)
            return {"doc_id": doc_id, "title": "", "text": ""}
        return meta

    def search(
        self,
        query: str,
        dense_k: int = DENSE_TOP_K,
        sparse_k: int = SPARSE_TOP_K,
        rerank_pool: int = RERANK_POOL,
        final_top_k: int = FINAL_TOP_K,
        use_reranker: bool = True,
    ) -> list[dict]:
        """
        Run the full hybrid retrieval pipeline for *query*.

        Steps
        -----
        1. Dense search (Qdrant / Qwen3-Embedding)
        2. Sparse search (BM25-Okapi)
        3. RRF fusion
        4. Cross-encoder reranking (if use_reranker=True)

        Parameters
        ----------
        query        : str  — search query
        dense_k      : int  — dense candidate pool size
        sparse_k     : int  — sparse candidate pool size
        rerank_pool  : int  — how many fused candidates the reranker sees
        final_top_k  : int  — number of results returned
        use_reranker : bool — set False for ablation / speed

        Returns
        -------
        list[dict] — length ≤ final_top_k, each dict:
        {
            "rank"        : int,
            "doc_id"      : str,
            "title"       : str,
            "text"        : str,
            "rrf_score"   : float,
            "rerank_score": float,   # 0.0 when use_reranker=False
        }
        """
        log.info("search | query=%.80s …", query)

        # 1 + 2 — retrieve candidates
        dense_ids = self._dense_search(query)
        sparse_ids = self._sparse_search(query)

        # 3 — RRF fusion
        fused = reciprocal_rank_fusion([dense_ids, sparse_ids], k=RRF_K)
        rrf_score_map = {doc_id: score for doc_id, score in fused}
        pool_ids = [doc_id for doc_id, _ in fused]

        # 4 — optional reranking
        if use_reranker and self._reranker is not None:
            candidates = [
                {**self._lookup(did), "rrf_score": rrf_score_map[did]}
                for did in pool_ids[:rerank_pool]
            ]
            ranked = self._reranker.rerank(query, candidates, top_k=final_top_k)
        else:
            ranked = [
                {
                    **self._lookup(did),
                    "rrf_score": rrf_score_map[did],
                    "rerank_score": 0.0,
                }
                for did in pool_ids[:final_top_k]
            ]

        # Attach rank numbers
        results = []
        for rank, doc in enumerate(ranked[:final_top_k], start=1):
            results.append(
                {
                    "rank": rank,
                    "doc_id": doc.get("doc_id", ""),
                    "title": doc.get("title", ""),
                    "text": doc.get("text", ""),
                    "rrf_score": doc.get("rrf_score", 0.0),
                    "rerank_score": doc.get("rerank_score", 0.0),
                }
            )

        log.info(
            "search done | returned %d results  top_rrf=%.5f  top_rerank=%.4f",
            len(results),
            results[0]["rrf_score"] if results else 0.0,
            results[0]["rerank_score"] if results else 0.0,
        )
        return results
