"""
Hybrid Search — DevRev Enterprise Search Challenge

Pipeline: BM25 + Dense (OpenAI text-embedding-3-large) + HyDE + Multi-Query
          → Weighted RRF → Neighbor Expansion → Cohere Rerank v3.5 → Top-10

Prerequisites:
  1. aws sso login --profile dev
  2. kubectl port-forward -n llm-gateway service/llm-gateway 4000:80
  3. Set LITELLM_KEY and COHERE_API_KEY in .env
  4. Run embed_corpus.py first to generate document embeddings
"""

import json
import os
import pickle
import re
import time
from pathlib import Path

import cohere
import faiss
import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
from tqdm import tqdm

load_dotenv()
litellm_client = OpenAI(
    api_key=os.getenv("LITELLM_KEY"),
    base_url="http://localhost:4000/v1",
)
cohere_client = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))

EMBED_MODEL = "openai/text-embedding-3-large"
GENERATIVE_MODEL = "vertex_ai/gemini-2.5-flash"
COHERE_RERANK_MODEL = "rerank-v3.5"

BM25_TOP_K = 100
DENSE_TOP_K = 100
RERANK_POOL = 200
FINAL_TOP_K = 10
RRF_K = 40
BM25_WEIGHT = 2.0
DENSE_WEIGHT = 1.3
HYDE_DENSE_WEIGHT = 1.0
MULTI_QUERY_WEIGHT = 0.7
NEIGHBOR_TOP_FRACTION = 0.5

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

STOPWORDS = set("a an the is are was were be been being have has had do does did "
                "will would shall should may might can could of in to for on with "
                "at by from as into through during before after above below between "
                "and but or nor not no so yet both either neither each every all any "
                "few more most other some such than too very it its this that these "
                "those i me my we our you your he him his she her they them their "
                "what which who whom how when where why".split())


def load_disk_cache(name):
    path = CACHE_DIR / f"{name}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            cache = pickle.load(f)
        print(f"  Loaded {len(cache)} cached {name} entries")
        return cache
    return {}


def save_disk_cache(name, cache):
    with open(CACHE_DIR / f"{name}.pkl", "wb") as f:
        pickle.dump(cache, f)


def load_data():
    print("Loading datasets...")
    annotated = load_dataset("devrev/search", "annotated_queries")["train"]
    test = load_dataset("devrev/search", "test_queries")["test"]
    corpus = load_dataset("devrev/search", "knowledge_base")["corpus"]
    print(f"  Annotated: {len(annotated)}  Test: {len(test)}  Corpus: {len(corpus)}")
    return annotated, test, corpus


def prepare_corpus(corpus):
    doc_ids, doc_titles, doc_texts, doc_combined = [], [], [], []
    for item in corpus:
        doc_ids.append(item["id"])
        doc_titles.append(item["title"])
        doc_texts.append(item["text"])
        doc_combined.append(f"{item['title']}\n\n{item['text']}")
    return doc_ids, doc_titles, doc_texts, doc_combined


def build_neighbor_map(doc_ids):
    id_to_idx = {did: i for i, did in enumerate(doc_ids)}
    neighbor_map = {}
    for i, did in enumerate(doc_ids):
        neighbors = set()
        if "_KNOWLEDGE_NODE-" in did:
            parts = did.rsplit("-", 1)
            base, chunk_num = parts[0], int(parts[1])
            for offset in [-1, 1]:
                nid = f"{base}-{chunk_num + offset}"
                if nid in id_to_idx:
                    neighbors.add(id_to_idx[nid])
        neighbor_map[i] = neighbors
    return neighbor_map


def tokenize_bm25(text):
    text = re.sub(r'[^a-z0-9\s]', ' ', text.lower())
    return [t for t in text.split() if t not in STOPWORDS and len(t) > 1]


def build_bm25_index(doc_combined):
    cache_path = CACHE_DIR / "bm25_index.pkl"
    if cache_path.exists():
        print("Loading cached BM25 index...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    print("Building BM25 index...")
    tokenized = [tokenize_bm25(doc) for doc in tqdm(doc_combined, desc="Tokenizing")]
    bm25 = BM25Okapi(tokenized)
    with open(cache_path, "wb") as f:
        pickle.dump(bm25, f)
    return bm25


def bm25_search(bm25, query, top_k=100):
    scores = bm25.get_scores(tokenize_bm25(query))
    top_indices = np.argsort(scores)[::-1][:top_k]
    return top_indices, scores[top_indices]


query_embed_cache = load_disk_cache("query_embed_cache")


def embed_query(query_text):
    if query_text in query_embed_cache:
        return query_embed_cache[query_text]
    for attempt in range(10):
        try:
            response = litellm_client.embeddings.create(
                model=EMBED_MODEL,
                input=[query_text],
            )
            emb = np.array(response.data[0].embedding, dtype=np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-10)
            query_embed_cache[query_text] = emb
            return emb
        except Exception as e:
            wait = min(2 ** (attempt + 1), 60)
            print(f"\n  Embed error (attempt {attempt+1}): {e}\n  Retrying in {wait}s...")
            time.sleep(wait)
    raise ConnectionError(f"Failed to embed after 10 attempts: {query_text[:50]}")


def build_dense_index():
    embeddings_path = CACHE_DIR / "embeddings.npy"
    index_path = CACHE_DIR / "faiss.index"
    if index_path.exists():
        print("Loading cached FAISS index...")
        index = faiss.read_index(str(index_path))
        print(f"  Loaded: {index.ntotal} vectors")
        return index
    if not embeddings_path.exists():
        raise FileNotFoundError("Run embed_corpus.py first to generate embeddings!")
    print("Building FAISS index from cached embeddings...")
    embeddings = np.load(embeddings_path)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(index_path))
    print(f"  Index built: {index.ntotal} vectors, dim={dim}")
    return index


def dense_search(index, query_text, top_k=100):
    emb = embed_query(query_text)
    scores, indices = index.search(emb.reshape(1, -1), top_k)
    return indices[0], scores[0]


def generate_hyde_doc(query, hyde_cache):
    if query in hyde_cache:
        return hyde_cache[query]
    try:
        response = litellm_client.chat.completions.create(
            model=GENERATIVE_MODEL,
            messages=[{
                "role": "user",
                "content": (
                    "Write a short knowledge base article (2-3 sentences) that would perfectly "
                    "answer this enterprise software support query. Write ONLY the article content.\n\n"
                    f"Query: {query}"
                ),
            }],
        )
        result = response.choices[0].message.content.strip()
    except Exception:
        result = None
    hyde_cache[query] = result
    return result


def generate_multi_queries(query, mq_cache):
    if query in mq_cache:
        return mq_cache[query]
    try:
        response = litellm_client.chat.completions.create(
            model=GENERATIVE_MODEL,
            messages=[{
                "role": "user",
                "content": (
                    "Rewrite this enterprise software search query in 3 different ways to improve "
                    "search recall. Return ONLY the 3 queries, one per line, no numbering.\n\n"
                    f"Original: {query}"
                ),
            }],
        )
        result = [q.strip() for q in response.choices[0].message.content.strip().split("\n") if q.strip()][:3]
    except Exception:
        result = []
    mq_cache[query] = result
    return result


def reciprocal_rank_fusion(results_list, weights=None, k=60):
    if weights is None:
        weights = [1.0] * len(results_list)
    fused = {}
    for (indices, _scores), w in zip(results_list, weights):
        for rank, idx in enumerate(indices):
            idx = int(idx)
            fused[idx] = fused.get(idx, 0.0) + w / (k + rank + 1)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def cohere_rerank(query, candidate_indices, doc_combined, top_k=10):
    if not candidate_indices:
        return []
    documents = [doc_combined[idx] for idx in candidate_indices]
    for attempt in range(5):
        try:
            response = cohere_client.rerank(
                model=COHERE_RERANK_MODEL,
                query=query,
                documents=documents,
                top_n=top_k,
            )
            return [(candidate_indices[r.index], r.relevance_score) for r in response.results]
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                time.sleep(min(2 ** (attempt + 1), 30))
            else:
                raise
    return []


def main():
    annotated, test, corpus = load_data()
    doc_ids, doc_titles, doc_texts, doc_combined = prepare_corpus(corpus)
    neighbor_map = build_neighbor_map(doc_ids)
    bm25 = build_bm25_index(doc_combined)
    dense_index = build_dense_index()
    hyde_cache = load_disk_cache("hyde_cache")
    mq_cache = load_disk_cache("mq_cache")

    def hybrid_search(query, top_k=FINAL_TOP_K):
        bm25_idx, bm25_sc = bm25_search(bm25, query, BM25_TOP_K)
        d_idx, d_sc = dense_search(dense_index, query, DENSE_TOP_K)
        results = [(bm25_idx, bm25_sc), (d_idx, d_sc)]
        weights = [BM25_WEIGHT, DENSE_WEIGHT]

        hyde_doc = generate_hyde_doc(query, hyde_cache)
        if hyde_doc:
            h_idx, h_sc = dense_search(dense_index, hyde_doc, DENSE_TOP_K)
            results.append((h_idx, h_sc))
            weights.append(HYDE_DENSE_WEIGHT)

        alt_queries = generate_multi_queries(query, mq_cache)
        for alt_q in alt_queries:
            a_idx, a_sc = dense_search(dense_index, alt_q, DENSE_TOP_K)
            results.append((a_idx, a_sc))
            weights.append(MULTI_QUERY_WEIGHT / max(len(alt_queries), 1))

        fused = reciprocal_rank_fusion(results, weights, RRF_K)

        n_hi = max(1, int(min(len(fused), RERANK_POOL) * NEIGHBOR_TOP_FRACTION))
        thresh = fused[min(n_hi, len(fused) - 1)][1] if fused else 0
        candidates, seen = [], set()
        for idx, score in fused[:RERANK_POOL]:
            if idx not in seen:
                candidates.append(idx)
                seen.add(idx)
            if score >= thresh:
                for nb in neighbor_map.get(idx, set()):
                    if nb not in seen:
                        candidates.append(nb)
                        seen.add(nb)
        candidates = candidates[:RERANK_POOL]

        reranked = cohere_rerank(query, candidates, doc_combined, top_k)
        return [idx for idx, _ in reranked]

    print("\n" + "=" * 60)
    print("  EVALUATION — ANNOTATED QUERIES (291)")
    print("=" * 60)

    checkpoint_path = CACHE_DIR / "eval_checkpoint.json"
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        eval_results = checkpoint["results"]
        start_idx = checkpoint["next_idx"]
        print(f"  Resuming evaluation from query {start_idx}/291")
    else:
        eval_results = []
        start_idx = 0

    ks = [5, 10, 25, 50]
    for i in tqdm(range(start_idx, len(annotated)),
                  initial=start_idx, total=len(annotated), desc="Eval"):
        item = annotated[i]
        query = item["query"]
        gold_ids = set(r["id"] for r in item["retrievals"])
        retrieved = [doc_ids[idx] for idx in hybrid_search(query, max(ks))]
        row = {}
        for k in ks:
            hits = len(gold_ids & set(retrieved[:k]))
            row[str(k)] = hits / len(gold_ids) if gold_ids else 0
        eval_results.append(row)

        if (i + 1) % 10 == 0:
            save_disk_cache("hyde_cache", hyde_cache)
            save_disk_cache("mq_cache", mq_cache)
            save_disk_cache("query_embed_cache", query_embed_cache)
            with open(checkpoint_path, "w") as f:
                json.dump({"next_idx": i + 1, "results": eval_results}, f)

    save_disk_cache("hyde_cache", hyde_cache)
    save_disk_cache("mq_cache", mq_cache)
    save_disk_cache("query_embed_cache", query_embed_cache)

    print(f"\n{'=' * 60}")
    print(f"  HYBRID (OpenAI Embed + HyDE + MultiQuery + Cohere Rerank)")
    print(f"{'=' * 60}")
    for k in ks:
        mean_recall = np.mean([r[str(k)] for r in eval_results]) * 100
        print(f"  Recall@{k:<3}: {mean_recall:6.2f}%")
    print(f"{'=' * 60}")

    recall_10 = np.mean([r["10"] for r in eval_results]) * 100

    print("\n--- Generating test query results ---")
    test_checkpoint_path = CACHE_DIR / "test_checkpoint.json"
    if test_checkpoint_path.exists():
        with open(test_checkpoint_path) as f:
            test_checkpoint = json.load(f)
        test_results = test_checkpoint["results"]
        test_start = test_checkpoint["next_idx"]
        print(f"  Resuming test queries from {test_start}/92")
    else:
        test_results = []
        test_start = 0

    for i in tqdm(range(test_start, len(test)),
                  initial=test_start, total=len(test), desc="Test queries"):
        item = test[i]
        qid, q = item["query_id"], item["query"]
        idxs = hybrid_search(q, FINAL_TOP_K)
        test_results.append({
            "query_id": qid,
            "query": q,
            "retrievals": [{"id": doc_ids[idx], "text": doc_texts[idx], "title": doc_titles[idx]}
                           for idx in idxs],
        })
        if (i + 1) % 10 == 0:
            save_disk_cache("hyde_cache", hyde_cache)
            save_disk_cache("mq_cache", mq_cache)
            with open(test_checkpoint_path, "w") as f:
                json.dump({"next_idx": i + 1, "results": test_results}, f)

    out = "test_queries_results.json"
    with open(out, "w") as f:
        json.dump(test_results, f, indent=2)

    if checkpoint_path.exists():
        os.remove(checkpoint_path)
    if test_checkpoint_path.exists():
        os.remove(test_checkpoint_path)

    print(f"\nResults saved to {out}")
    print(f"  {len(test_results)} queries, {FINAL_TOP_K} retrievals each")
    print(f"  Estimated Recall@10: {recall_10:.2f}%")


if __name__ == "__main__":
    main()
