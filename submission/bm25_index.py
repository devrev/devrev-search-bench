# =============================================================================
# bm25_index.py — BM25-Okapi sparse retriever (rank_bm25)
# =============================================================================
"""
Builds a BM25-Okapi index over the corpus and caches it as a pickle so
tokenisation is not repeated on subsequent runs.

Public API
----------
    from bm25_index import BM25Index

    idx = BM25Index()
    idx.build(documents, doc_ids, ck)
    results = idx.search(query, top_k=50)
    # results → list of {"doc_id": str, "score": float, "corpus_idx": int}
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from checkpoint import Checkpoint
from config import BM25_CACHE_PATH, SPARSE_TOP_K
from logger import get_logger

log = get_logger(__name__)

_CACHE = Path(BM25_CACHE_PATH)


def _tokenise(text: str) -> list[str]:
    """Whitespace tokenisation (fast; good enough for BM25 on English text)."""
    return text.lower().split()


class BM25Index:
    """
    Wrapper around BM25Okapi that adds checkpoint-aware build/load and a
    consistent search interface.
    """

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._doc_ids: list[str] = []

    def build(
        self,
        documents: list[str],
        doc_ids: list[str],
        ck: Checkpoint,
    ) -> None:
        """
        Tokenise all documents and build the BM25 index.

        If the checkpoint phase 'bm25_indexed' is already done and the pickle
        cache exists, the index is loaded from disk without re-tokenising.

        Parameters
        ----------
        documents : list[str]  — title-prepended document texts
        doc_ids   : list[str]  — original document IDs (same order as documents)
        ck        : Checkpoint — shared pipeline checkpoint
        """
        self._doc_ids = doc_ids

        if ck.done("bm25_indexed") and _CACHE.exists():
            log.info("Phase 'bm25_indexed' done — loading BM25 from %s", _CACHE)
            self._load()
            return

        log.info("=== Phase: bm25_indexed ===")
        log.info("Tokenising %d documents for BM25 …", len(documents))

        tokenised = [_tokenise(doc) for doc in tqdm(documents, desc="BM25 tokenise")]

        avg_len = np.mean([len(t) for t in tokenised])
        log.debug("Avg token count per document: %.1f", avg_len)

        log.info("Building BM25Okapi index …")
        self._bm25 = BM25Okapi(tokenised)

        self._save()
        ck.mark_done("bm25_indexed", bm25_corpus_size=len(documents))
        log.info("BM25 index built and cached → %s", _CACHE)

    def _save(self) -> None:
        _CACHE.parent.mkdir(parents=True, exist_ok=True)
        payload = {"bm25": self._bm25, "doc_ids": self._doc_ids}
        with open(_CACHE, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        log.debug("BM25 index saved to %s", _CACHE)

    def _load(self) -> None:
        with open(_CACHE, "rb") as fh:
            payload = pickle.load(fh)
        self._bm25 = payload["bm25"]
        self._doc_ids = payload["doc_ids"]
        log.info(
            "BM25 index loaded  corpus_size=%d",
            len(self._doc_ids),
        )

    def search(self, query: str, top_k: int = SPARSE_TOP_K) -> list[dict]:
        """
        Score all documents against *query* and return the top-k.

        Parameters
        ----------
        query : str — raw query string (tokenised internally)
        top_k : int — number of candidates to return

        Returns
        -------
        list of dicts:
            {"doc_id": str, "score": float, "corpus_idx": int}
        Sorted by score descending.
        """
        if self._bm25 is None:
            raise RuntimeError("BM25Index.build() must be called before search()")

        tokens = _tokenise(query)
        log.debug("BM25 search: query_tokens=%s  top_k=%d", tokens, top_k)

        scores = self._bm25.get_scores(tokens)  # shape (N,)
        top_indices = np.argsort(scores)[::-1][:top_k].tolist()

        results = [
            {
                "doc_id": self._doc_ids[idx],
                "score": float(scores[idx]),
                "corpus_idx": idx,
            }
            for idx in top_indices
        ]
        log.debug(
            "BM25 returned %d hits (top score=%.4f)",
            len(results),
            results[0]["score"] if results else 0.0,
        )
        return results
