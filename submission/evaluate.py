# =============================================================================
# evaluate.py — evaluation metrics: Recall@K, Precision@K, MRR
# =============================================================================
"""
Public API
----------
    from evaluate import evaluate_retriever, print_report

    metrics = evaluate_retriever(retriever, annotated_df, k=10)
    print_report(metrics)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import EVAL_K, RESULTS_DIR
from logger import get_logger
from retriever import HybridRetriever

log = get_logger(__name__)


# =============================================================================
# Metric functions
# =============================================================================
def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant docs found in the top-k retrieved."""
    if not relevant:
        return 0.0
    hits = sum(1 for r in retrieved[:k] if r in relevant)
    return hits / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of top-k retrieved docs that are relevant."""
    if k == 0:
        return 0.0
    hits = sum(1 for r in retrieved[:k] if r in relevant)
    return hits / k


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """1/rank of the first relevant document (0 if none found)."""
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


# =============================================================================
# Evaluation runner
# =============================================================================
def evaluate_retriever(
    retriever: HybridRetriever,
    annotated_df: pd.DataFrame,
    k: int = EVAL_K,
    use_reranker: bool = True,
    sample: int | None = None,
    tag: str = "",
) -> dict:
    """
    Run the retriever over all (or a sample of) annotated queries and
    compute mean Recall@K, Precision@K and MRR.

    Parameters
    ----------
    retriever    : HybridRetriever — fully initialised retriever
    annotated_df : pd.DataFrame   — must have columns "query" and "retrievals"
                                    where "retrievals" is a list of {"id":...}
    k            : int             — cutoff for Recall/Precision
    use_reranker : bool            — passed through to retriever.search()
    sample       : int|None        — evaluate only the first N queries (None=all)
    tag          : str             — label for logging / report

    Returns
    -------
    dict with keys:
        tag, n_queries, k,
        recall_at_k, precision_at_k, mrr,
        use_reranker
    """
    rows = annotated_df.to_dict("records")
    if sample is not None:
        rows = rows[:sample]
        log.info("Evaluation subset: first %d queries", sample)

    log.info(
        "=== Evaluation  tag='%s'  n=%d  k=%d  reranker=%s ===",
        tag,
        len(rows),
        k,
        use_reranker,
    )

    recalls, precisions, mrrs = [], [], []

    for row in tqdm(rows, desc=f"Eval [{tag or 'default'}]"):
        query = row["query"]
        golden_ids = {r["id"] for r in row["retrievals"]}

        results = retriever.search(query, use_reranker=use_reranker)
        retrieved_ids = [r["doc_id"] for r in results]

        rec = recall_at_k(retrieved_ids, golden_ids, k)
        prec = precision_at_k(retrieved_ids, golden_ids, k)
        mrr = reciprocal_rank(retrieved_ids, golden_ids)

        recalls.append(rec)
        precisions.append(prec)
        mrrs.append(mrr)

        log.debug(
            "query=%.60s | R@%d=%.3f P@%d=%.3f MRR=%.3f",
            query,
            k,
            rec,
            k,
            prec,
            mrr,
        )

    metrics = {
        "tag": tag,
        "n_queries": len(recalls),
        "k": k,
        "recall_at_k": float(np.mean(recalls)),
        "precision_at_k": float(np.mean(precisions)),
        "mrr": float(np.mean(mrrs)),
        "use_reranker": use_reranker,
    }

    log.info(
        "Eval results [%s] — Recall@%d=%.4f  Precision@%d=%.4f  MRR=%.4f",
        tag,
        k,
        metrics["recall_at_k"],
        k,
        metrics["precision_at_k"],
        metrics["mrr"],
    )
    return metrics


# =============================================================================
# Ablation runner
# =============================================================================
def run_ablation(
    retriever: HybridRetriever,
    annotated_df: pd.DataFrame,
    k: int = EVAL_K,
    sample: int = 100,
) -> list[dict]:
    """
    Run 2-way ablation: RRF-only vs RRF + Reranker.
    Returns list of metric dicts (one per configuration).
    """
    log.info("Running ablation over %d queries …", sample)
    configs = [
        ("Hybrid RRF (no reranker)", False),
        ("Hybrid RRF + Reranker", True),
    ]
    all_metrics = []
    for tag, use_reranker in configs:
        m = evaluate_retriever(
            retriever,
            annotated_df,
            k=k,
            use_reranker=use_reranker,
            sample=sample,
            tag=tag,
        )
        all_metrics.append(m)
    return all_metrics


# =============================================================================
# Reporting
# =============================================================================
def print_report(metrics_list: list[dict]) -> None:
    """Pretty-print a table of evaluation metrics."""
    k = metrics_list[0]["k"] if metrics_list else EVAL_K
    header = f"{'System':<30} {'Recall@'+str(k):>12} {'Precision@'+str(k):>14} {'MRR':>8} {'N':>6}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for m in metrics_list:
        print(
            f"{m['tag']:<30} "
            f"{m['recall_at_k']:>12.4f} "
            f"{m['precision_at_k']:>14.4f} "
            f"{m['mrr']:>8.4f} "
            f"{m['n_queries']:>6}"
        )
    print("=" * len(header) + "\n")


def save_metrics(
    metrics_list: list[dict], filename: str = "evaluation_metrics.json"
) -> None:
    """Persist evaluation results to the results directory."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    with open(path, "w") as fh:
        json.dump(metrics_list, fh, indent=2)
    log.info("Evaluation metrics saved → %s", path)
