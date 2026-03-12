"""
Section 10 — Multi-Query Triple-Path Hybrid Retrieval Pipeline
Loads saved artefacts (embeddings, FAISS index) and generates the final submission.
"""

import os
# Prevent tokenizer / OMP forking that causes segfaults on macOS
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
# Disable MPS watermark — avoids OOM from stale MPS allocations across runs
# MPS watermark disabled to avoid stale-allocation OOM if MPS is used elsewhere
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import re, ast, json, time
import numpy as np
import faiss
import torch
from collections import defaultdict
from tqdm import tqdm

# ── Device ──────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
print(f"Device: {DEVICE.upper()}")

# ── Packages ─────────────────────────────────────────────────────────────────
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    import subprocess; subprocess.check_call(["pip", "install", "-q", "rank_bm25"])
    from rank_bm25 import BM25Okapi

try:
    import accelerate  # noqa
except ImportError:
    import subprocess; subprocess.check_call(["pip", "install", "-q", "accelerate"])

try:
    import protobuf  # noqa
except ImportError:
    import subprocess; subprocess.check_call(["pip", "install", "-q", "protobuf"])

from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers.utils import is_accelerate_available
is_accelerate_available.cache_clear()
torch.set_default_device(None)

# ── Load Dataset ─────────────────────────────────────────────────────────────
print("\n[1/8] Loading datasets...")
from datasets import load_dataset
knowledge_base    = load_dataset("devrev/search", "knowledge_base")
annotated_queries = load_dataset("devrev/search", "annotated_queries")
test_queries      = load_dataset("devrev/search", "test_queries")
corpus_data = knowledge_base["corpus"]
ground_truth = [dict(item) for item in annotated_queries["train"]]
print(f"  Knowledge base: {len(corpus_data):,} docs")
print(f"  Annotated queries: {len(ground_truth)}")
print(f"  Test queries: {len(test_queries['test'])}")

# ── Text Cleaning ─────────────────────────────────────────────────────────────
print("\n[2/8] Cleaning corpus text...")
doc_ids, doc_titles, doc_texts = [], [], []
for item in corpus_data:
    doc_ids.append(item["id"])
    doc_titles.append(item["title"])
    doc_texts.append(item["text"])

def clean_text(text: str) -> str:
    if isinstance(text, str) and (text.startswith("b'") or text.startswith('b"')):
        try:
            evaluated = ast.literal_eval(text)
            if isinstance(evaluated, bytes):
                text = evaluated.decode("utf-8", errors="replace")
        except (ValueError, SyntaxError):
            text = text[2:-1]
    text = text.replace("\\n", "\n")
    text = re.sub(r"\\x[0-9a-fA-F]{2}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()

cleaned_texts = [clean_text(t) for t in doc_texts]
cleaned_documents = [f"{title}\n\n{ct}" for title, ct in zip(doc_titles, cleaned_texts)]
print(f"  Cleaned {len(cleaned_documents):,} documents")

# ── Snowflake Embedding Model ─────────────────────────────────────────────────
print("\n[3/8] Loading snowflake-arctic-embed-l-v2.0 (CPU)...")
dense_model = SentenceTransformer("Snowflake/snowflake-arctic-embed-l-v2.0", device="cpu")
print(f"  Dim: {dense_model.get_sentence_embedding_dimension()}, Device: {next(dense_model.parameters()).device}")

def get_dense_embedding(text: str, is_query: bool = False) -> np.ndarray:
    if is_query:
        return dense_model.encode(text, prompt_name="query", normalize_embeddings=True)
    return dense_model.encode(text, normalize_embeddings=True)

# ── Load Saved Embeddings + Build FAISS Index ─────────────────────────────────
print("\n[4/8] Loading saved embeddings and building FAISS index...")
dense_embeddings = np.load("embeddings_snowflake.npy")
print(f"  Embeddings shape: {dense_embeddings.shape}")

enhanced_index = faiss.IndexFlatIP(dense_embeddings.shape[1])
enhanced_index.add(dense_embeddings.astype("float32"))
print(f"  FAISS index: {enhanced_index.ntotal:,} vectors")

# ── BM25 Indices ──────────────────────────────────────────────────────────────
print("\n[5/8] Building BM25 indices...")
tokenized_docs   = [doc.lower().split() for doc in cleaned_documents]
bm25_index       = BM25Okapi(tokenized_docs)
tokenized_titles = [title.lower().split() for title in doc_titles]
title_bm25_index = BM25Okapi(tokenized_titles)
print(f"  Full-text BM25: {len(tokenized_docs):,} docs")
print(f"  Title-only BM25: {len(tokenized_titles):,} docs")

# ── Reranker (BERT-based, fast on CPU, no MPS memory issues) ─────────────────
print("\n[6/8] Loading reranker on CPU...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
_ = reranker.predict([("warmup query", "quick warmup doc")], show_progress_bar=False)
print("  Warmup done.")
print("  Reranker loaded.")

RERANK_DOC_CHARS = 600   # truncate long docs before reranking

# ── Query Expansion (rule-based; no external API needed) ─────────────────────
# Generates lightweight semantic variations using simple transformations
def expand_query(query: str) -> list:
    variants = [query]
    q = query.strip()

    # Variation 1: rephrase imperatives as questions and vice versa
    lower = q.lower()
    if lower.startswith("how to "):
        variants.append("steps to " + q[7:])
    elif lower.startswith("how do i "):
        variants.append("guide for " + q[9:])
    elif lower.startswith("what is "):
        variants.append(q[8:] + " definition and overview")
    elif lower.startswith("what are "):
        variants.append(q[9:] + " list and examples")
    else:
        variants.append("how to " + q)

    # Variation 2: append common DevRev context terms based on keywords
    kw_map = {
        "api": "REST API integration endpoint",
        "webhook": "webhook event notification callback",
        "snap-in": "snap-in plugin DevRev app",
        "auth": "authentication token OAuth",
        "deploy": "deployment configuration setup",
        "error": "error troubleshooting fix",
        "account": "account settings management",
        "data": "data export import sync",
    }
    extra = next((v for k, v in kw_map.items() if k in lower), "")
    if extra:
        variants.append(f"{q} {extra}")

    return list(dict.fromkeys(variants))[:3]   # deduplicate, keep ≤3

# ── Multi-Query Triple-Path Pipeline ─────────────────────────────────────────
def multi_query_triple_path_rerank(
    query: str,
    top_k: int = 10,
    dense_candidates: int = 100,
    bm25_candidates: int = 100,
    title_bm25_candidates: int = 50,
    rerank_pool: int = 60,
    rrf_k: int = 60,
    title_rrf_weight: float = 2.0,
):
    query_variants = expand_query(query)
    rrf_scores = defaultdict(float)

    for q in query_variants:
        tokens = q.lower().split()

        # Path A — Dense
        q_emb = get_dense_embedding(q, is_query=True).astype("float32").reshape(1, -1)
        _, dense_indices = enhanced_index.search(q_emb, dense_candidates)
        for rank, idx in enumerate(dense_indices[0]):
            rrf_scores[int(idx)] += 1.0 / (rrf_k + rank + 1)

        # Path B — Full-text BM25
        bm25_sc = bm25_index.get_scores(tokens)
        for rank, idx in enumerate(np.argsort(bm25_sc)[::-1][:bm25_candidates]):
            rrf_scores[int(idx)] += 1.0 / (rrf_k + rank + 1)

        # Path C — Title-only BM25 (2× weight)
        title_sc = title_bm25_index.get_scores(tokens)
        for rank, idx in enumerate(np.argsort(title_sc)[::-1][:title_bm25_candidates]):
            rrf_scores[int(idx)] += title_rrf_weight / (rrf_k + rank + 1)

    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:rerank_pool]
    candidate_indices = [idx for idx, _ in fused]

    pairs = [(query, cleaned_documents[idx][:RERANK_DOC_CHARS]) for idx in candidate_indices]
    rerank_sc = reranker.predict(pairs, batch_size=32, show_progress_bar=False)
    scored = sorted(zip(candidate_indices, rerank_sc), key=lambda x: x[1], reverse=True)

    results = []
    for idx, _ in scored[:top_k]:
        doc = corpus_data[idx]
        results.append({"id": doc["id"], "text": doc["text"], "title": doc["title"]})
    return results

# ── Evaluation Helper ─────────────────────────────────────────────────────────
def evaluate_retrieval(predictions, ground_truth, k_values=[1, 3, 5, 10]):
    gt_map = {g["query_id"]: set(r["id"] for r in g["retrievals"]) for g in ground_truth}
    metrics = {}
    for k in k_values:
        recalls, precisions = [], []
        for pred in predictions:
            golden_ids = gt_map.get(pred["query_id"], set())
            if not golden_ids:
                continue
            predicted_ids = [r["id"] for r in pred["retrievals"][:k]]
            hits = sum(1 for pid in predicted_ids if pid in golden_ids)
            recalls.append(hits / len(golden_ids))
            precisions.append(hits / k)
        metrics[f"Recall@{k}"]    = 100 * sum(recalls)    / len(recalls)    if recalls    else 0
        metrics[f"Precision@{k}"] = 100 * sum(precisions) / len(precisions) if precisions else 0
    mrrs = []
    for pred in predictions:
        golden_ids = gt_map.get(pred["query_id"], set())
        if not golden_ids:
            continue
        for rank, r in enumerate(pred["retrievals"], 1):
            if r["id"] in golden_ids:
                mrrs.append(1.0 / rank)
                break
        else:
            mrrs.append(0.0)
    metrics["MRR"] = 100 * sum(mrrs) / len(mrrs) if mrrs else 0
    return metrics

# ── Sanity Test ───────────────────────────────────────────────────────────────
print("\n[7/8] Sanity test (1 query)...")
t0 = time.time()
test_r = multi_query_triple_path_rerank("How do I set up AirSync?", top_k=3)
print(f"  Done in {time.time()-t0:.1f}s")
for i, r in enumerate(test_r, 1):
    print(f"  [{i}] {r['title']}")

# ── Run on All 92 Test Queries ────────────────────────────────────────────────
print("\n[8/8] Running pipeline on all 92 test queries...")
s10_test_results = []
for item in tqdm(test_queries["test"], desc="Test queries"):
    retrievals = multi_query_triple_path_rerank(item["query"], top_k=10)
    s10_test_results.append({
        "query_id": item["query_id"],
        "query":    item["query"],
        "retrievals": retrievals,
    })

OUTPUT_FILE = "test_queries_results_s10_multiquery.json"
with open(OUTPUT_FILE, "w") as f:
    json.dump(s10_test_results, f, indent=2)

import pandas as pd
pd.DataFrame(s10_test_results).to_parquet(
    "test_queries_results_s10_multiquery.parquet", index=False
)

print(f"\n{'='*55}")
print(f"Submission saved:  {OUTPUT_FILE}")
print(f"  Queries:         {len(s10_test_results)}")
print(f"  Per query:       10 retrievals")
print(f"  Pipeline:        Multi-query(x3) + Triple-path RRF + zerank")
print(f"{'='*55}")

print("\nDone! Submit: test_queries_results_s10_multiquery.json")

# ── Optional: Eval on Annotated Queries (set RUN_EVAL=1 to enable) ────────────
if os.environ.get("RUN_EVAL") == "1":
    print("\n[OPTIONAL] Evaluating on annotated queries (~90 min)...")
    eval_results = []
    for item in tqdm(annotated_queries["train"], desc="Eval"):
        retrievals = multi_query_triple_path_rerank(item["query"], top_k=10)
        eval_results.append({"query_id": item["query_id"], "query": item["query"], "retrievals": retrievals})

    metrics = evaluate_retrieval(eval_results, ground_truth)

    LEADERBOARD_REF = {
        "Rank 1 (gemini+zerank, closed)": {"Precision@10": 26.85, "Recall@10": 36.50},
        "Rank 2 OS (Qwen3-8B+zerank)":    {"Precision@10": 26.63, "Recall@10": 36.09},
        "Rank 3 OS (snowflake+zerank)":   {"Precision@10": 26.63, "Recall@10": 36.09},
    }

    print(f"\n{'='*65}")
    print(f"  {'METRIC':<16} {'OURS':>10}   {'Rank 1 target':>14}   {'Gap':>8}")
    print(f"{'='*65}")
    for k in ["Precision@10", "Recall@10", "MRR", "Precision@5", "Recall@5"]:
        v   = metrics.get(k, 0)
        r1  = LEADERBOARD_REF["Rank 1 (gemini+zerank, closed)"].get(k)
        gap = f"{v - r1:+.2f}" if r1 else "—"
        r1s = f"{r1:.2f}%" if r1 else "—"
        print(f"  {k:<16} {v:>9.2f}%   {r1s:>14}   {gap:>8}")
    print(f"{'='*65}")
else:
    print("\nSkipping annotated-query eval (run with RUN_EVAL=1 to enable).")
