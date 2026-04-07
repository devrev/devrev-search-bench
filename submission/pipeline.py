# =============================================================================
# pipeline.py — full end-to-end pipeline with checkpoint-aware phases
# =============================================================================
"""
Runs the complete DevRev hybrid search pipeline in order:

  Phase 1: datasets_downloaded   — download & cache HF splits
  Phase 2: embeddings_computed   — embed corpus with Qwen3-Embedding-0.6B
  Phase 3: qdrant_indexed        — upsert embeddings into Qdrant
  Phase 4: bm25_indexed          — build & cache BM25 index
  Phase 5: test_results_saved    — run hybrid search on test queries & save

Each phase is guarded by a checkpoint.  Re-running the script after a crash
automatically skips completed phases and resumes from where it failed.

Usage
-----
    # Full run (skips completed phases automatically):
    python pipeline.py

    # Force re-run from a specific phase onwards:
    python pipeline.py --reset-from embeddings_computed

    # Run evaluation after pipeline:
    python pipeline.py --evaluate

    # Quick smoke-test on 50 annotated queries:
    python pipeline.py --evaluate --eval-sample 50

    # Provide HF token via CLI (alternative to HF_TOKEN env-var):
    python pipeline.py --hf-token hf_...
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd
from tqdm import tqdm

from bm25_index import BM25Index
from checkpoint import Checkpoint
from config import (
    FINAL_TOP_K,
    PHASES,
    RESULTS_DIR,
    HF_TOKEN,
)
from dataset_loader import load_datasets
from embedder import Embedder
from evaluate import (
    evaluate_retriever,
    print_report,
    run_ablation,
    save_metrics,
)
from logger import get_logger
from reranker import Reranker
from retriever import HybridRetriever
from vector_store import VectorStore

log = get_logger(__name__)


# =============================================================================
# Corpus helpers
# =============================================================================
def build_corpus_texts(kb_df: pd.DataFrame) -> tuple[list[str], list[str], list[dict]]:
    """
    Convert the knowledge-base DataFrame into three parallel lists:

    Returns
    -------
    documents  : list[str]   — title-prepended full text for embedding
    doc_ids    : list[str]   — original doc IDs
    corpus_meta: list[dict]  — {"doc_id", "title", "text"} per document
    """
    log.info("Building corpus from %d KB rows …", len(kb_df))
    documents: list[str] = []
    doc_ids: list[str] = []
    corpus_meta: list[dict] = []

    for row in kb_df.itertuples(index=False):
        doc_id = str(row.id)
        title = str(row.title)
        text = str(row.text)

        documents.append(f"{title}\n\n{text}")
        doc_ids.append(doc_id)
        corpus_meta.append({"doc_id": doc_id, "title": title, "text": text})

    log.info("Corpus ready: %d documents", len(documents))
    return documents, doc_ids, corpus_meta


# =============================================================================
# Per-phase runners
# =============================================================================
def phase_index(
    ck: Checkpoint,
    documents: list[str],
    doc_ids: list[str],
    corpus_meta: list[dict],
    embedder: Embedder,
    vs: VectorStore,
    bm25: BM25Index,
) -> None:
    """Run embedding + Qdrant + BM25 phases (skips done ones)."""
    # Phase 2: embeddings
    embeddings = embedder.compute_corpus_embeddings(documents, ck)

    # Phase 3: Qdrant
    payloads = [{"title": m["title"], "text": m["text"]} for m in corpus_meta]
    vs.index_corpus(doc_ids, embeddings, payloads, ck)

    # Phase 4: BM25
    bm25.build(documents, doc_ids, ck)


def phase_test_results(
    ck: Checkpoint,
    retriever: HybridRetriever,
    test_df: pd.DataFrame,
) -> None:
    """Run hybrid search over test_queries and save results JSON."""
    if ck.done("test_results_saved"):
        log.info("Phase 'test_results_saved' already done — skipping")
        return

    log.info("=== Phase: test_results_saved ===")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    test_results = []
    for row in tqdm(
        test_df.itertuples(index=False), total=len(test_df), desc="Test queries"
    ):
        results = retriever.search(str(row.query), use_reranker=True)
        test_results.append(
            {
                "query_id": str(row.query_id),
                "query": str(row.query),
                "retrievals": [
                    {"id": r["doc_id"], "title": r["title"], "text": r["text"]}
                    for r in results
                ],
            }
        )
        log.debug("query_id=%s  retrieved=%d", row.query_id, len(results))

    output_path = RESULTS_DIR / "test_queries_results.json"
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(test_results, fh, indent=2, ensure_ascii=False)

    log.info("Test results saved → %s  (%d queries)", output_path, len(test_results))
    ck.mark_done("test_results_saved", test_queries_processed=len(test_results))


# =============================================================================
# Main
# =============================================================================
def build_retriever(
    vs: VectorStore,
    bm25: BM25Index,
    embedder: Embedder,
    corpus_meta: list[dict],
    use_reranker: bool = True,
) -> HybridRetriever:
    reranker = Reranker() if use_reranker else None
    return HybridRetriever(vs, bm25, embedder, corpus_meta, reranker=reranker)


def main(args: argparse.Namespace) -> None:
    if args.hf_token:
        import config as _cfg

        _cfg.HF_TOKEN = args.hf_token
        log.info("HF token provided via CLI")

    log.info("========== DevRev Hybrid Search Pipeline ==========")

    ck = Checkpoint()

    # Optional: reset from a given phase
    if args.reset_from:
        ck.reset_from(args.reset_from)

    ck.summary()

    dfs = load_datasets(ck)
    documents, doc_ids, corpus_meta = build_corpus_texts(dfs["kb"])

    embedder = Embedder()
    vs = VectorStore()
    bm25 = BM25Index()

    phase_index(ck, documents, doc_ids, corpus_meta, embedder, vs, bm25)

    retriever = build_retriever(
        vs,
        bm25,
        embedder,
        corpus_meta,
        use_reranker=not args.no_reranker,
    )

    phase_test_results(ck, retriever, dfs["test"])

    if args.evaluate:
        log.info("Running evaluation on annotated queries …")

        if args.ablation:
            metrics_list = run_ablation(
                retriever,
                dfs["annotated"],
                k=args.eval_k,
                sample=args.eval_sample,
            )
        else:
            m = evaluate_retriever(
                retriever,
                dfs["annotated"],
                k=args.eval_k,
                use_reranker=not args.no_reranker,
                sample=args.eval_sample,
                tag="Hybrid RRF + Reranker" if not args.no_reranker else "Hybrid RRF",
            )
            metrics_list = [m]

        print_report(metrics_list)
        save_metrics(metrics_list)

    ck.summary()
    log.info("Pipeline complete ✓")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DevRev hybrid search pipeline (Qwen3 + Qdrant + RRF)"
    )
    p.add_argument(
        "--reset-from",
        metavar="PHASE",
        choices=PHASES,
        default=None,
        help=("Force re-run from this phase onwards. Choices: " + ", ".join(PHASES)),
    )
    p.add_argument(
        "--hf-token",
        default=None,
        metavar="TOKEN",
        help="HuggingFace token (alternative to HF_TOKEN env-var)",
    )
    p.add_argument(
        "--no-reranker",
        action="store_true",
        help="Skip cross-encoder reranking (faster, slightly lower quality)",
    )
    p.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation on annotated_queries after indexing",
    )
    p.add_argument(
        "--ablation",
        action="store_true",
        help="Run ablation (RRF vs RRF+Reranker) instead of single eval",
    )
    p.add_argument(
        "--eval-sample",
        type=int,
        default=None,
        metavar="N",
        help="Evaluate on first N annotated queries only (default: all)",
    )
    p.add_argument(
        "--eval-k",
        type=int,
        default=10,
        metavar="K",
        help="Cutoff K for Recall@K / Precision@K (default: 10)",
    )
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
