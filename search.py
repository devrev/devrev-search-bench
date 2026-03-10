#!/usr/bin/env python3
"""
DevRev Search Challenge - Hybrid BM25 + Semantic + Cross-Encoder Re-ranking Pipeline

Strategy:
1. BM25 (keyword search) - catches exact term matches
2. BGE embeddings + FAISS (semantic search) - catches meaning-based matches
3. Reciprocal Rank Fusion - merges both candidate lists
4. Cross-encoder re-ranking - neural model scores top candidates

No API keys required - uses fully open-source models.
"""

import json
import os
import re
import time
import pickle

import numpy as np
import faiss
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────────────
SEMANTIC_TOP_K = 50       # candidates from semantic search
BM25_TOP_K = 50           # candidates from BM25
RERANK_TOP_K = 20         # candidates sent to cross-encoder
FINAL_TOP_K = 10          # final results per query
RRF_K = 60                # RRF constant

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBEDDING_BATCH_SIZE = 64
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

EMBEDDINGS_CACHE = "embeddings_bge_base.npy"
FAISS_INDEX_CACHE = "bge_base.faiss"
BM25_CACHE = "bm25_index.pkl"
OUTPUT_FILE = "test_queries_results.json"


# ── Text Preprocessing ────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Remove markdown artifacts, excess whitespace, URLs."""
    text = re.sub(r"!\[\]\([^)]*\)", "", text)            # markdown images
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # markdown links -> text
    text = re.sub(r"https?://\S+", "", text)               # URLs
    text = re.sub(r"[#*\-]{2,}", " ", text)                # repeated markdown chars
    text = re.sub(r"\\n", " ", text)                        # escaped newlines
    text = re.sub(r"\s+", " ", text).strip()
    return text


def make_semantic_doc(title: str, text: str) -> str:
    """Format document for embedding model."""
    cleaned = clean_text(text)
    return f"{title}. {cleaned}"


def make_bm25_tokens(title: str, text: str) -> list[str]:
    """Tokenize document for BM25. Title repeated for emphasis."""
    cleaned = clean_text(text)
    combined = f"{title} {title} {cleaned}".lower()
    return combined.split()


# ── Data Loading ───────────────────────────────────────────────────────────────
def load_data():
    """Load all dataset splits from HuggingFace."""
    print("Loading datasets from HuggingFace...")
    knowledge_base = load_dataset("devrev/search", "knowledge_base", split="corpus")
    test_queries = load_dataset("devrev/search", "test_queries", split="test")
    annotated_queries = load_dataset("devrev/search", "annotated_queries", split="train")
    print(f"  Knowledge base: {len(knowledge_base):,} documents")
    print(f"  Test queries: {len(test_queries):,} queries")
    print(f"  Annotated queries: {len(annotated_queries):,} queries")
    return knowledge_base, test_queries, annotated_queries


# ── BM25 Index ─────────────────────────────────────────────────────────────────
def build_bm25_index(corpus):
    """Build BM25 index over knowledge base."""
    if os.path.exists(BM25_CACHE):
        print("Loading cached BM25 index...")
        with open(BM25_CACHE, "rb") as f:
            return pickle.load(f)

    print("Building BM25 index...")
    tokenized_corpus = [
        make_bm25_tokens(doc["title"], doc["text"])
        for doc in tqdm(corpus, desc="Tokenizing for BM25")
    ]
    bm25 = BM25Okapi(tokenized_corpus)

    with open(BM25_CACHE, "wb") as f:
        pickle.dump(bm25, f)
    print(f"  BM25 index built and cached to {BM25_CACHE}")
    return bm25


def bm25_search(bm25, query: str, k: int = BM25_TOP_K) -> list[tuple[int, float]]:
    """Search BM25 index, return list of (doc_index, score)."""
    tokens = clean_text(query).lower().split()
    scores = bm25.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:k]
    return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]


# ── Semantic Index ─────────────────────────────────────────────────────────────
def build_semantic_index(corpus):
    """Build FAISS index with BGE embeddings."""
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Load or generate embeddings
    if os.path.exists(EMBEDDINGS_CACHE):
        print(f"Loading cached embeddings from {EMBEDDINGS_CACHE}...")
        embeddings = np.load(EMBEDDINGS_CACHE)
        if embeddings.shape[0] != len(corpus):
            print(f"  Cache mismatch ({embeddings.shape[0]} vs {len(corpus)}), regenerating...")
            embeddings = None
        else:
            print(f"  Loaded embeddings: {embeddings.shape}")
    else:
        embeddings = None

    if embeddings is None:
        print(f"Generating embeddings with {EMBEDDING_MODEL}...")
        semantic_docs = [
            make_semantic_doc(doc["title"], doc["text"])
            for doc in tqdm(corpus, desc="Preparing documents")
        ]
        embeddings = model.encode(
            semantic_docs,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        embeddings = embeddings.astype("float32")
        np.save(EMBEDDINGS_CACHE, embeddings)
        print(f"  Embeddings saved to {EMBEDDINGS_CACHE}")

    # Build FAISS index
    if os.path.exists(FAISS_INDEX_CACHE):
        print(f"Loading cached FAISS index from {FAISS_INDEX_CACHE}...")
        index = faiss.read_index(FAISS_INDEX_CACHE)
    else:
        print("Building FAISS index...")
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        embeddings_norm = embeddings.copy()
        faiss.normalize_L2(embeddings_norm)
        index.add(embeddings_norm)
        faiss.write_index(index, FAISS_INDEX_CACHE)
        print(f"  FAISS index saved to {FAISS_INDEX_CACHE}")

    print(f"  Index contains {index.ntotal:,} vectors of dim {index.d}")
    return model, index


def semantic_search(
    model, index, query: str, k: int = SEMANTIC_TOP_K
) -> list[tuple[int, float]]:
    """Search semantic index, return list of (doc_index, score)."""
    query_embedding = model.encode(
        QUERY_PREFIX + query, normalize_embeddings=True
    ).astype("float32").reshape(1, -1)
    scores, indices = index.search(query_embedding, k)
    return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0])]


# ── Reciprocal Rank Fusion ─────────────────────────────────────────────────────
def reciprocal_rank_fusion(
    *results_lists, k: int = RRF_K
) -> list[tuple[int, float]]:
    """Merge multiple ranked lists using RRF."""
    scores = {}
    for results in results_lists:
        for rank, (doc_idx, _) in enumerate(results):
            if doc_idx not in scores:
                scores[doc_idx] = 0.0
            scores[doc_idx] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ── Cross-Encoder Re-ranking ──────────────────────────────────────────────────
def build_reranker():
    """Load cross-encoder model."""
    print(f"Loading cross-encoder: {CROSS_ENCODER_MODEL}...")
    return CrossEncoder(CROSS_ENCODER_MODEL)


def rerank(
    reranker, query: str, candidates: list[tuple[int, float]], corpus, top_k: int = FINAL_TOP_K
) -> list[tuple[int, float]]:
    """Re-rank top candidates with cross-encoder."""
    candidates_to_rerank = candidates[:RERANK_TOP_K]
    if not candidates_to_rerank:
        return []

    pairs = [
        (query, make_semantic_doc(corpus[idx]["title"], corpus[idx]["text"]))
        for idx, _ in candidates_to_rerank
    ]
    scores = reranker.predict(pairs)

    indexed_scores = [
        (idx, float(score))
        for (idx, _), score in zip(candidates_to_rerank, scores)
    ]
    indexed_scores.sort(key=lambda x: x[1], reverse=True)
    return indexed_scores[:top_k]


# ── Evaluation (optional, against annotated queries) ──────────────────────────
def evaluate_on_annotated(results_by_qid: dict, annotated_queries):
    """Compute recall@10 against annotated golden retrievals."""
    hits = 0
    total = 0
    for item in annotated_queries:
        qid = item["query_id"]
        if qid not in results_by_qid:
            continue
        golden_ids = {r["id"] for r in item["retrievals"]}
        predicted_ids = {r["id"] for r in results_by_qid[qid]}
        hits += len(golden_ids & predicted_ids)
        total += len(golden_ids)
    if total > 0:
        print(f"\n  Recall@{FINAL_TOP_K} on annotated queries: {hits}/{total} = {hits/total:.4f}")
    else:
        print("\n  No overlapping queries for evaluation")


# ── Main Pipeline ──────────────────────────────────────────────────────────────
def main():
    start_time = time.time()

    # Load data
    corpus, test_queries, annotated_queries = load_data()

    # Build indices
    bm25 = build_bm25_index(corpus)
    embedding_model, faiss_index = build_semantic_index(corpus)
    reranker = build_reranker()

    # Process test queries
    print(f"\nProcessing {len(test_queries)} test queries...")
    results = []
    for item in tqdm(test_queries, desc="Searching"):
        query = item["query"]
        query_id = item["query_id"]

        # Stage 1: Retrieve from both sources
        sem_results = semantic_search(embedding_model, faiss_index, query)
        bm25_results = bm25_search(bm25, query)

        # Stage 2: Fuse with RRF
        fused = reciprocal_rank_fusion(sem_results, bm25_results)

        # Stage 3: Re-rank top candidates
        reranked = rerank(reranker, query, fused, corpus)

        # Format output
        retrievals = []
        for idx, score in reranked:
            doc = corpus[int(idx)]
            retrievals.append({
                "id": doc["id"],
                "text": doc["text"],
                "title": doc["title"],
            })

        results.append({
            "query_id": query_id,
            "query": query,
            "retrievals": retrievals,
        })

    # Save results
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\nDone! Saved {len(results)} query results to {OUTPUT_FILE}")
    print(f"Total time: {elapsed:.1f}s")

    # Optional: evaluate against annotated queries
    results_by_qid = {r["query_id"]: r["retrievals"] for r in results}
    evaluate_on_annotated(results_by_qid, annotated_queries)


if __name__ == "__main__":
    main()
