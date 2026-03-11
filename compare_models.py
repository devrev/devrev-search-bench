"""
Multi-model comparison for DevRev Search benchmark.

Embeds the knowledge base with multiple Ollama models, runs retrieval
on the annotated queries, and produces a metrics leaderboard.

Usage:
    python compare_models.py [--models nomic-embed-text mxbai-embed-large ...]
    python compare_models.py --top-k 10
"""

import argparse
import json
import os
import time

import faiss
import numpy as np
import ollama
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from evaluate import evaluate_queries, print_evaluation_report

# Task prefix configuration for models that benefit from prefixes.
# "doc_prefix" is prepended to knowledge base texts before embedding.
# "query_prefix" is prepended to queries before embedding.
MODEL_PREFIXES = {
    "nomic-embed-text": {
        "doc_prefix": "search_document: ",
        "query_prefix": "search_query: ",
    },
    "mxbai-embed-large": {
        "doc_prefix": "",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
    "snowflake-arctic-embed": {
        "doc_prefix": "",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
    "bge-m3": {
        "doc_prefix": "",
        "query_prefix": "",
    },
}

# Max characters per model (rough approx: 1 token ~ 4 chars).
# Texts longer than this are truncated before embedding.
MODEL_MAX_CHARS = {
    "all-minilm": 800,                # 256 token context
    "qwen3-embedding:0.6b": 32000,    # 32K token context
    "nomic-embed-text": 32000,        # 8K token context
    "mxbai-embed-large": 2000,        # 512 token context
    "snowflake-arctic-embed": 2000,   # 512 token context
    "bge-m3": 32000,                  # 8K token context
}

DEFAULT_MAX_CHARS = 8000

# Models to benchmark (name -> ollama model tag)
AVAILABLE_MODELS = {
    "qwen3-embedding:0.6b": {
        "tag": "qwen3-embedding:0.6b",
        "description": "Qwen3 0.6B embedding (default in repo)",
    },
    "nomic-embed-text": {
        "tag": "nomic-embed-text",
        "description": "Nomic Embed Text v1.5 (768-dim, 274M, task-prefixed)",
    },
    "mxbai-embed-large": {
        "tag": "mxbai-embed-large",
        "description": "MixedBread embed-large (1024-dim, 670M, query-prefixed)",
    },
    "snowflake-arctic-embed": {
        "tag": "snowflake-arctic-embed",
        "description": "Snowflake Arctic Embed (1024-dim, 335M, query-prefixed)",
    },
    "all-minilm": {
        "tag": "all-minilm",
        "description": "All-MiniLM-L6-v2 (384-dim, 46M, fast baseline)",
    },
    "bge-m3": {
        "tag": "bge-m3",
        "description": "BGE-M3 multilingual (1024-dim, 1.2G)",
    },
}


def pull_model_if_needed(model_tag: str) -> None:
    """Pull an Ollama model if it isn't already available locally."""
    try:
        models = ollama.list()
        model_list = models.models if hasattr(models, "models") else models.get("models", [])
        local_names = []
        for m in model_list:
            name = m.model if hasattr(m, "model") else m.get("model", "")
            local_names.append(name)
            # Also add without :latest suffix
            if name.endswith(":latest"):
                local_names.append(name.replace(":latest", ""))

        if model_tag in local_names:
            print(f"  Model {model_tag} already available locally")
            return
    except Exception:
        pass

    print(f"  Pulling {model_tag} (this may take a while)...")
    ollama.pull(model_tag)
    print(f"  Done pulling {model_tag}")


def get_prefix(model_tag: str, kind: str) -> str:
    """Return the task prefix for a model ('doc' or 'query')."""
    prefixes = MODEL_PREFIXES.get(model_tag, {})
    if kind == "doc":
        return prefixes.get("doc_prefix", "")
    return prefixes.get("query_prefix", "")


def truncate_text(text: str, model_tag: str) -> str:
    """Truncate text to the model's max context length."""
    max_chars = MODEL_MAX_CHARS.get(model_tag, DEFAULT_MAX_CHARS)
    return text[:max_chars] if len(text) > max_chars else text


def embed_single(text: str, model_tag: str) -> np.ndarray:
    """Embed a single text string using Ollama (auto-truncated to model limits)."""
    response = ollama.embed(model=model_tag, input=truncate_text(text, model_tag))
    embeddings = response.embeddings if hasattr(response, "embeddings") else response["embeddings"]
    embedding = embeddings[0] if embeddings and isinstance(embeddings[0], list) else embeddings
    return np.array(embedding, dtype=np.float32)


def embed_texts(texts: list[str], model_tag: str, prefix: str = "", batch_size: int = 10) -> np.ndarray:
    """Embed a list of texts one-by-one with an optional prefix."""
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Embedding ({model_tag})"):
        batch = texts[i : i + batch_size]
        batch_embs = [embed_single(f"{prefix}{t}", model_tag) for t in batch]
        all_embeddings.extend(batch_embs)
    return np.vstack(all_embeddings).astype(np.float32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build a FAISS cosine similarity index from normalized embeddings."""
    emb = embeddings.copy()
    faiss.normalize_L2(emb)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index


def retrieve_for_queries(
    queries: list[dict],
    index: faiss.IndexFlatIP,
    corpus_data: list[dict],
    model_tag: str,
    top_k: int = 10,
) -> list[dict]:
    """Run retrieval for a list of queries and return results."""
    query_prefix = get_prefix(model_tag, "query")
    if query_prefix:
        print(f"  Using query prefix: '{query_prefix}'")

    results = []
    for item in tqdm(queries, desc=f"Retrieving ({model_tag})"):
        qemb = embed_single(f"{query_prefix}{item['query']}", model_tag).reshape(1, -1)
        faiss.normalize_L2(qemb)
        _scores, indices = index.search(qemb, top_k)

        retrievals = [
            {
                "id": corpus_data[int(idx)]["id"],
                "text": corpus_data[int(idx)]["text"],
                "title": corpus_data[int(idx)]["title"],
            }
            for idx in indices[0]
        ]
        results.append({
            "query_id": item["query_id"],
            "query": item["query"],
            "retrievals": retrievals,
        })
    return results


def run_benchmark(
    model_tags: list[str],
    top_k: int = 10,
    output_dir: str = "benchmark_results",
) -> pd.DataFrame:
    """Run the full benchmark for the given models and return a leaderboard DataFrame."""
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    print("\nLoading dataset...")
    knowledge_base = load_dataset("devrev/search", "knowledge_base", split="corpus")
    annotated_queries = load_dataset("devrev/search", "annotated_queries", split="train")

    corpus_list = list(knowledge_base)
    documents = [f"{item['title']}\n\n{item['text']}" for item in corpus_list]
    golden = list(annotated_queries)

    print(f"  Knowledge base: {len(documents):,} documents")
    print(f"  Annotated queries: {len(golden)} (for evaluation)")

    leaderboard_rows = []

    for model_tag in model_tags:
        print(f"\n{'=' * 60}")
        print(f"  Benchmarking: {model_tag}")
        print(f"{'=' * 60}")

        # Pull model
        pull_model_if_needed(model_tag)

        # Resolve prefixes for this model
        doc_prefix = get_prefix(model_tag, "doc")
        if doc_prefix:
            print(f"  Using document prefix: '{doc_prefix}'")

        # Check for cached embeddings
        cache_path = os.path.join(output_dir, f"embeddings_{model_tag.replace(':', '_').replace('/', '_')}.npy")

        if os.path.exists(cache_path):
            print(f"  Loading cached embeddings from {cache_path}")
            embeddings = np.load(cache_path)
        else:
            start_time = time.time()
            embeddings = embed_texts(documents, model_tag, prefix=doc_prefix, batch_size=10)
            embed_time = time.time() - start_time
            np.save(cache_path, embeddings)
            print(f"  Embeddings cached to {cache_path}")
            print(f"  Embedding time: {embed_time:.1f}s ({embed_time / 60:.1f}min)")

        # Build index
        index = build_faiss_index(embeddings)
        print(f"  FAISS index: {index.ntotal:,} vectors, dim={embeddings.shape[1]}")

        # Retrieve for annotated queries
        predictions = retrieve_for_queries(golden, index, corpus_list, model_tag, top_k)

        # Evaluate
        eval_results = evaluate_queries(predictions, golden, k_values=(1, 3, 5, 10))
        print_evaluation_report(eval_results)

        # Save per-model results
        model_file = os.path.join(
            output_dir,
            f"results_{model_tag.replace(':', '_').replace('/', '_')}.json",
        )
        with open(model_file, "w") as f:
            json.dump(predictions, f, indent=2)

        # Build leaderboard row
        summary = eval_results["summary"]
        row = {
            "model": model_tag,
            "dim": embeddings.shape[1],
            "mrr": summary["mrr"],
            "map": summary["map"],
            "ndcg@1": summary["ndcg@1"],
            "ndcg@3": summary["ndcg@3"],
            "ndcg@5": summary["ndcg@5"],
            "ndcg@10": summary["ndcg@10"],
            "recall@5": summary["recall@5"],
            "recall@10": summary["recall@10"],
            "precision@5": summary["precision@5"],
            "precision@10": summary["precision@10"],
        }
        leaderboard_rows.append(row)

    # Build leaderboard
    leaderboard = pd.DataFrame(leaderboard_rows)
    leaderboard = leaderboard.sort_values("ndcg@10", ascending=False).reset_index(drop=True)
    leaderboard.index = leaderboard.index + 1  # 1-based rank
    leaderboard.index.name = "rank"

    # Save leaderboard
    lb_path = os.path.join(output_dir, "leaderboard.csv")
    leaderboard.to_csv(lb_path)
    print(f"\n{'=' * 60}")
    print("  LEADERBOARD")
    print(f"{'=' * 60}")
    print(leaderboard.to_string(float_format="%.4f"))
    print(f"\nSaved to {lb_path}")

    return leaderboard


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Ollama embedding models on DevRev Search")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["qwen3-embedding:0.6b", "nomic-embed-text", "all-minilm"],
        help="Ollama model tags to benchmark",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Number of retrievals per query")
    parser.add_argument("--output-dir", default="benchmark_results", help="Directory for results")
    parser.add_argument("--list-models", action="store_true", help="List available models and exit")
    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        for name, info in AVAILABLE_MODELS.items():
            print(f"  {name:<30} {info['description']}")
        return

    run_benchmark(args.models, top_k=args.top_k, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
