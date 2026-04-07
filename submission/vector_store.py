# =============================================================================
# vector_store.py — Qdrant local vector store
# =============================================================================
"""
Uses qdrant-client in local mode: all data is stored on disk under
QDRANT_PATH.  No server process or Docker is needed.

Public API
----------
    from vector_store import VectorStore

    vs = VectorStore()                  # connects / creates collection
    vs.index_corpus(doc_ids, embeddings, payloads, ck)
    results = vs.search(query_vec, top_k=50)
    # results → list of {"id": str, "score": float, "payload": dict}
"""

from __future__ import annotations

import numpy as np
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    ScoredPoint,
)

from checkpoint import Checkpoint
from config import (
    QDRANT_PATH,
    QDRANT_COLLECTION,
    EMBED_DIM,
    DENSE_TOP_K,
)
from logger import get_logger

log = get_logger(__name__)

_UPSERT_BATCH = 256  # points per Qdrant upsert call


class VectorStore:
    """
    Thin wrapper around a local QdrantClient collection.

    The collection uses cosine distance (equivalent to inner product on
    L2-normalised vectors, which is what the Qwen3 embedder produces).
    """

    def __init__(self) -> None:
        log.info("Connecting to local Qdrant at path=%s …", QDRANT_PATH)
        self._client = QdrantClient(path=QDRANT_PATH)
        log.debug("QdrantClient initialised")

    def _collection_exists(self) -> bool:
        existing = [c.name for c in self._client.get_collections().collections]
        return QDRANT_COLLECTION in existing

    def _create_collection(self) -> None:
        log.info(
            "Creating Qdrant collection '%s'  dim=%d …", QDRANT_COLLECTION, EMBED_DIM
        )
        self._client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=EMBED_DIM,
                distance=Distance.COSINE,
            ),
        )
        log.info("Collection '%s' created", QDRANT_COLLECTION)

    def _drop_collection(self) -> None:
        log.warning("Dropping Qdrant collection '%s' for re-index …", QDRANT_COLLECTION)
        self._client.delete_collection(QDRANT_COLLECTION)

    def index_corpus(
        self,
        doc_ids: list[str],
        embeddings: np.ndarray,
        payloads: list[dict],
        ck: Checkpoint,
        force: bool = False,
    ) -> None:
        """
        Upsert all corpus embeddings into Qdrant in batches.

        Skips entirely if the checkpoint phase 'qdrant_indexed' is already
        done (and force=False).

        Parameters
        ----------
        doc_ids    : list[str]  — original document IDs (used as payload field)
        embeddings : np.ndarray — shape (N, EMBED_DIM), float32, L2-normalised
        payloads   : list[dict] — per-point metadata (title, text, …)
        ck         : Checkpoint — shared pipeline checkpoint
        force      : bool       — drop & re-index even if already done
        """
        if ck.done("qdrant_indexed") and not force:
            log.info("Phase 'qdrant_indexed' already done — skipping Qdrant index")
            return

        if force and self._collection_exists():
            self._drop_collection()

        if not self._collection_exists():
            self._create_collection()

        n = len(doc_ids)
        log.info("=== Phase: qdrant_indexed ===")
        log.info("Upserting %d points into '%s' …", n, QDRANT_COLLECTION)

        points: list[PointStruct] = []
        for i in tqdm(range(n), desc="Building Qdrant points"):
            points.append(
                PointStruct(
                    id=i,  # integer ID (Qdrant requirement)
                    vector=embeddings[i].tolist(),
                    payload={**payloads[i], "doc_id": doc_ids[i]},
                )
            )

        # Upsert in batches so OOM is unlikely even with large corpora
        total_batches = (n + _UPSERT_BATCH - 1) // _UPSERT_BATCH
        log.info("Upserting in %d batches of %d …", total_batches, _UPSERT_BATCH)

        for batch_start in tqdm(range(0, n, _UPSERT_BATCH), desc="Qdrant upsert"):
            batch = points[batch_start : batch_start + _UPSERT_BATCH]
            self._client.upsert(collection_name=QDRANT_COLLECTION, points=batch)
            log.debug("Upserted batch [%d:%d]", batch_start, batch_start + len(batch))

        info = self._client.get_collection(QDRANT_COLLECTION)
        log.info(
            "Qdrant index complete: %d vectors stored in '%s'",
            info.points_count,
            QDRANT_COLLECTION,
        )

        ck.mark_done("qdrant_indexed", qdrant_points=info.points_count)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = DENSE_TOP_K,
    ) -> list[dict]:
        """
        Run an ANN search against the Qdrant collection.

        Parameters
        ----------
        query_vector : np.ndarray — shape (EMBED_DIM,), L2-normalised float32
        top_k        : int        — number of candidates to return

        Returns
        -------
        list of dicts:
            {"doc_id": str, "score": float, "payload": dict, "point_id": int}
        """
        log.debug(
            "Qdrant search: top_k=%d  vec_norm=%.4f",
            top_k,
            float(np.linalg.norm(query_vector)),
        )

        # qdrant-client ≥1.9 removed .search() in favour of .query_points()
        response = self._client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector.tolist(),
            limit=top_k,
            with_payload=True,
        )
        hits: list[ScoredPoint] = response.points

        results = [
            {
                "doc_id": hit.payload.get("doc_id", ""),
                "score": hit.score,
                "payload": hit.payload,
                "point_id": hit.id,
            }
            for hit in hits
        ]
        log.debug(
            "Qdrant returned %d hits (top score=%.4f)",
            len(results),
            results[0]["score"] if results else 0.0,
        )
        return results
