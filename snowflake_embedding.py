"""
Snowflake Arctic Embed Pipeline for DevRev Search Bench2

Embeds ~65K DevRev knowledge base articles using Snowflake/snowflake-arctic-embed-l-v2.0,
builds FAISS indexes (IVFFlat + FlatIP) and a BM25 index.

Usage:
    python snowflake_embedding.py

Outputs:
    embeddings_local.npy              -- corpus embeddings (65224 x 1024)
    bm25_local.pkl                    -- BM25 index
    faiss_index_local/
        doc_mapping.pkl               -- id/title/text mapping
        knowledge_base.index          -- IVFFlat index (fast)
        knowledge_base_flat.index     -- FlatIP index (exact)
"""

import os
import pickle
import re
import time

import faiss
import numpy as np
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EMBED_MODEL = "Snowflake/snowflake-arctic-embed-l-v2.0"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

INDEX_DIR = "faiss_index_local"
EMBEDDINGS_FILE = "embeddings_local.npy"
BM25_FILE = "bm25_local.pkl"
BATCH_SIZE = 64


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def get_device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # 1. Load corpus
    print("Loading knowledge base...")
    kb = load_dataset("devrev/search", "knowledge_base", split="corpus")
    print(f"  {len(kb):,} chunks")

    doc_ids, doc_titles, doc_texts, documents = [], [], [], []
    for item in tqdm(kb, desc="Preparing docs"):
        doc_ids.append(item["id"])
        doc_titles.append(item["title"])
        doc_texts.append(item["text"])
        documents.append(f"{item['title']}\n\n{item['text']}")

    # 2. Save doc mapping
    os.makedirs(INDEX_DIR, exist_ok=True)
    with open(os.path.join(INDEX_DIR, "doc_mapping.pkl"), "wb") as f:
        pickle.dump({
            "doc_ids": doc_ids,
            "doc_titles": doc_titles,
            "doc_texts": doc_texts,
            "documents": documents,
        }, f)
    print(f"  Saved doc_mapping.pkl")

    # 3. Embed corpus
    device = get_device()
    print(f"\nLoading {EMBED_MODEL} on {device}...")
    model = SentenceTransformer(EMBED_MODEL, device=device, trust_remote_code=True)
    embed_dim = model.get_sentence_embedding_dimension()
    print(f"  Embedding dim: {embed_dim}")

    print(f"\nEmbedding {len(documents):,} docs (batch_size={BATCH_SIZE})...")
    t0 = time.time()
    embeddings = model.encode(
        documents,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    elapsed = time.time() - t0
    print(f"  Done: {embeddings.shape} in {elapsed:.1f}s ({len(documents)/elapsed:.0f} docs/sec)")

    np.save(EMBEDDINGS_FILE, embeddings)
    print(f"  Saved {EMBEDDINGS_FILE}")

    # 4. Build FAISS indexes
    dim = embeddings.shape[1]
    n = embeddings.shape[0]

    # IVFFlat
    nlist = int(np.sqrt(n))
    quantizer = faiss.IndexFlatIP(dim)
    index_ivf = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    index_ivf.train(embeddings)
    index_ivf.add(embeddings)
    index_ivf.nprobe = 20
    print(f"\n  IVFFlat: {index_ivf.ntotal:,} vectors, nlist={nlist}, nprobe=20")

    # FlatIP (exact)
    index_flat = faiss.IndexFlatIP(dim)
    index_flat.add(embeddings)
    print(f"  FlatIP:  {index_flat.ntotal:,} vectors")

    faiss.write_index(index_ivf, os.path.join(INDEX_DIR, "knowledge_base.index"))
    faiss.write_index(index_flat, os.path.join(INDEX_DIR, "knowledge_base_flat.index"))
    print(f"  Saved to {INDEX_DIR}/")

    # 5. Build BM25 index
    print("\nBuilding BM25 index...")
    t0 = time.time()
    tokenized = [re.findall(r'\w+', doc.lower()) for doc in tqdm(documents, desc="Tokenizing")]
    bm25 = BM25Okapi(tokenized)
    print(f"  BM25 built in {time.time()-t0:.1f}s")

    with open(BM25_FILE, "wb") as f:
        pickle.dump(bm25, f)
    print(f"  Saved {BM25_FILE}")

    print("\nAll done!")


if __name__ == "__main__":
    main()
