"""
Evaluation metrics for the DevRev Search benchmark.

Computes standard IR metrics (NDCG@k, MRR, Recall@k, Precision@k, MAP)
by comparing predicted retrievals against golden annotated retrievals.
"""

import numpy as np
from typing import Sequence


def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of top-k retrieved documents that are relevant."""
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / k


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant documents found in top-k results."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def average_precision(retrieved_ids: Sequence[str], relevant_ids: set[str]) -> float:
    """Average precision for a single query (used to compute MAP)."""
    if not relevant_ids:
        return 0.0
    hits = 0
    sum_precisions = 0.0
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            hits += 1
            sum_precisions += hits / (i + 1)
    return sum_precisions / len(relevant_ids)


def reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: set[str]) -> float:
    """Reciprocal rank: 1/position of first relevant document."""
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k (binary relevance)."""
    top_k = retrieved_ids[:k]

    # DCG: sum of 1/log2(i+2) for relevant docs at position i
    dcg = 0.0
    for i, doc_id in enumerate(top_k):
        if doc_id in relevant_ids:
            dcg += 1.0 / np.log2(i + 2)

    # Ideal DCG: all relevant docs ranked at top positions
    ideal_count = min(len(relevant_ids), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_count))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def evaluate_queries(
    predictions: list[dict],
    golden: list[dict],
    k_values: tuple[int, ...] = (1, 3, 5, 10),
) -> dict:
    """
    Evaluate predicted retrievals against golden annotations.

    Args:
        predictions: list of {"query_id": str, "retrievals": [{"id": ...}, ...]}
        golden: list of {"query_id": str, "retrievals": [{"id": ...}, ...]}
        k_values: tuple of k values to evaluate at

    Returns:
        dict with per-metric averages and per-query breakdown
    """
    golden_by_id = {
        item["query_id"]: {r["id"] for r in item["retrievals"]}
        for item in golden
    }

    # Only evaluate queries that exist in both predictions and golden
    matched_query_ids = [
        p["query_id"] for p in predictions if p["query_id"] in golden_by_id
    ]

    if not matched_query_ids:
        raise ValueError("No overlapping query_ids between predictions and golden set")

    pred_by_id = {p["query_id"]: p for p in predictions}

    per_query = []
    metrics_accum = {}

    for qid in matched_query_ids:
        retrieved_ids = [r["id"] for r in pred_by_id[qid]["retrievals"]]
        relevant_ids = golden_by_id[qid]

        row = {"query_id": qid, "query": pred_by_id[qid].get("query", "")}

        rr = reciprocal_rank(retrieved_ids, relevant_ids)
        row["mrr"] = rr
        metrics_accum.setdefault("mrr", []).append(rr)

        ap = average_precision(retrieved_ids, relevant_ids)
        row["ap"] = ap
        metrics_accum.setdefault("map", []).append(ap)

        for k in k_values:
            p = precision_at_k(retrieved_ids, relevant_ids, k)
            r = recall_at_k(retrieved_ids, relevant_ids, k)
            n = ndcg_at_k(retrieved_ids, relevant_ids, k)

            row[f"precision@{k}"] = p
            row[f"recall@{k}"] = r
            row[f"ndcg@{k}"] = n

            metrics_accum.setdefault(f"precision@{k}", []).append(p)
            metrics_accum.setdefault(f"recall@{k}", []).append(r)
            metrics_accum.setdefault(f"ndcg@{k}", []).append(n)

        per_query.append(row)

    summary = {metric: float(np.mean(values)) for metric, values in metrics_accum.items()}
    summary["num_queries_evaluated"] = len(matched_query_ids)

    return {"summary": summary, "per_query": per_query}


def print_evaluation_report(results: dict) -> None:
    """Pretty-print the evaluation summary."""
    summary = results["summary"]
    n = summary["num_queries_evaluated"]

    print("=" * 60)
    print(f"  Evaluation Report  ({n} queries)")
    print("=" * 60)
    print(f"  {'Metric':<20} {'Score':>10}")
    print("-" * 60)
    print(f"  {'MRR':<20} {summary['mrr']:>10.4f}")
    print(f"  {'MAP':<20} {summary['map']:>10.4f}")

    # Group by k
    k_values = sorted({
        int(k.split("@")[1])
        for k in summary
        if "@" in k
    })
    for k in k_values:
        print(f"\n  --- @{k} ---")
        print(f"  {'Precision@' + str(k):<20} {summary[f'precision@{k}']:>10.4f}")
        print(f"  {'Recall@' + str(k):<20} {summary[f'recall@{k}']:>10.4f}")
        print(f"  {'NDCG@' + str(k):<20} {summary[f'ndcg@{k}']:>10.4f}")

    print("=" * 60)
