"""
Embed the corpus with OpenAI text-embedding-3-large via LiteLLM proxy,
saving progress incrementally. Resumable — run multiple times if interrupted.

Prerequisites:
  1. aws sso login --profile dev
  2. kubectl port-forward -n llm-gateway service/llm-gateway 4000:80
  3. Set LITELLM_KEY in .env

Usage: python embed_corpus.py
Then run: python hybrid_search.py
"""

import os
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()
client = OpenAI(
    api_key=os.getenv("LITELLM_KEY"),
    base_url="http://localhost:4000/v1",
)

EMBED_MODEL = "openai/text-embedding-3-large"
EMBED_DIM = 3072
BATCH_SIZE = 10
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

PROGRESS_FILE = CACHE_DIR / "embed_progress.npy"
DONE_FILE = CACHE_DIR / "embeddings.npy"
COUNTER_FILE = CACHE_DIR / "embed_batch_idx.txt"


def main():
    if DONE_FILE.exists():
        emb = np.load(DONE_FILE)
        print(f"Embeddings already complete: {emb.shape}")
        return

    print("Loading corpus...")
    corpus = load_dataset("devrev/search", "knowledge_base")["corpus"]
    doc_combined = [f"{item['title']}\n\n{item['text']}" for item in corpus]
    n = len(doc_combined)
    print(f"  {n} documents")

    n_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE

    if PROGRESS_FILE.exists():
        embeddings = np.load(PROGRESS_FILE, allow_pickle=True)
        start_batch = int(open(COUNTER_FILE).read().strip()) if COUNTER_FILE.exists() else 0
        print(f"  Resuming from batch {start_batch}/{n_batches} ({start_batch * BATCH_SIZE} docs)")
    else:
        embeddings = np.zeros((n, EMBED_DIM), dtype=np.float32)
        start_batch = 0
        print(f"  Starting fresh: {n_batches} batches")

    for batch_idx in tqdm(range(start_batch, n_batches), initial=start_batch, total=n_batches,
                          desc="Embedding"):
        i = batch_idx * BATCH_SIZE
        batch = doc_combined[i:i + BATCH_SIZE]

        for attempt in range(8):
            try:
                response = client.embeddings.create(
                    model=EMBED_MODEL,
                    input=batch,
                )
                for j, item in enumerate(response.data):
                    embeddings[i + j] = item.embedding
                break
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    wait = min(2 ** (attempt + 1), 65)
                    print(f"\n  Rate limited (attempt {attempt+1}), waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"\n  Error: {e}")
                    if attempt >= 3:
                        print(f"  Saving progress at batch {batch_idx}.")
                        np.save(PROGRESS_FILE, embeddings)
                        with open(COUNTER_FILE, "w") as f:
                            f.write(str(batch_idx))
                        return
                    time.sleep(5)

        if (batch_idx + 1) % 50 == 0:
            np.save(PROGRESS_FILE, embeddings)
            with open(COUNTER_FILE, "w") as f:
                f.write(str(batch_idx + 1))
            print(f"\n  Progress saved: {(batch_idx + 1) * BATCH_SIZE}/{n} docs")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = (embeddings / norms).astype(np.float32)

    np.save(DONE_FILE, embeddings)
    if PROGRESS_FILE.exists():
        os.remove(PROGRESS_FILE)
    if COUNTER_FILE.exists():
        os.remove(COUNTER_FILE)

    print(f"\nDone! Embeddings saved: {DONE_FILE} — shape {embeddings.shape}")


if __name__ == "__main__":
    main()
