from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
TMP_DIR = PROJECT_ROOT / ".tmp"
TMP_DIR.mkdir(exist_ok=True)
os.environ.setdefault("TMPDIR", str(TMP_DIR))
os.environ.setdefault("TEMP", str(TMP_DIR))
os.environ.setdefault("TMP", str(TMP_DIR))

from datasets import load_dataset
import voyageai
import zvec


EMBED_DIM = 2048
EMBED_MODEL = "voyage-4-large"
RERANK_MODEL = "rerank-2.5"
RRF_K = 60


@dataclass(frozen=True)
class SearchConfig:
    name: str
    strategy: str = "rrf"
    dense_k: int = 200
    bm25_k: int = 200
    title_k: int = 100
    k_final: int = 10
    rerank_k: int = 120
    article_cap: int | None = None
    dense_weight: float = 1.25
    bm25_weight: float = 1.0
    title_weight: float = 0.35
    legacy_alpha: float = 0.65


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def article_id(chunk_id: str) -> str:
    return chunk_id.split("_KNOWLEDGE_NODE")[0]


def normalize_text(text: str) -> str:
    value = text
    if value.startswith(("b'", 'b"')) and value.endswith(("'", '"')):
        try:
            literal = ast.literal_eval(value)
            if isinstance(literal, bytes):
                value = literal.decode("utf-8", errors="replace")
            else:
                value = str(literal)
        except Exception:
            pass

    value = value.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def apply_article_cap(items: list[dict], k_final: int, cap: int | None) -> list[dict]:
    if cap is None:
        return items[:k_final]

    final_results: list[dict] = []
    per_article: Counter[str] = Counter()
    for item in items:
        art_id = article_id(item["id"])
        if per_article[art_id] >= cap:
            continue
        per_article[art_id] += 1
        final_results.append(item)
        if len(final_results) >= k_final:
            break
    return final_results


class DevRevSearchBench:
    def __init__(self, embeddings_path: Path, index_path: Path) -> None:
        load_dotenv()
        api_key = os.environ.get("VOYAGE_API_KEY")
        if not api_key:
            raise RuntimeError("VOYAGE_API_KEY is not set. Export it before running this script.")

        self.vo = voyageai.Client(api_key=api_key)
        self.embeddings_path = embeddings_path
        self.index_path = index_path

        print("Loading datasets...")
        self.annotated_queries = load_dataset("devrev/search", "annotated_queries", split="train")
        self.test_queries = load_dataset("devrev/search", "test_queries", split="test")
        self.knowledge_base = load_dataset("devrev/search", "knowledge_base", split="corpus")

        self.documents: list[str] = []
        self.doc_ids: list[str] = []
        self.doc_titles: list[str] = []
        self.doc_texts: list[str] = []
        self.query_embedding_cache: dict[str, list[float]] = {}

        print("Preparing knowledge base...")
        for item in tqdm(self.knowledge_base, desc="Preparing documents"):
            clean_text = normalize_text(item["text"])
            clean_title = normalize_text(item["title"])
            self.documents.append(f"{clean_title}\n\n{clean_text}")
            self.doc_ids.append(item["id"])
            self.doc_titles.append(clean_title)
            self.doc_texts.append(clean_text)

        print("Building BM25 indexes...")
        tokenized_corpus = [tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.title_bm25 = BM25Okapi([tokenize(title) for title in self.doc_titles])

        self.embeddings = self._load_embeddings()
        self.collection = self._rebuild_collection()

    def _load_embeddings(self) -> np.ndarray:
        if not self.embeddings_path.exists():
            raise FileNotFoundError(
                f"{self.embeddings_path} is missing. The repo currently depends on the cached document embeddings."
            )
        embeddings = np.load(self.embeddings_path)
        if embeddings.shape != (len(self.documents), EMBED_DIM):
            raise ValueError(
                f"Unexpected embeddings shape {embeddings.shape}; expected {(len(self.documents), EMBED_DIM)}."
            )
        return embeddings.astype(np.float32, copy=False)

    def _rebuild_collection(self):
        schema = zvec.CollectionSchema(
            name="devrev_kb",
            fields=[
                zvec.FieldSchema("title", zvec.DataType.STRING),
                zvec.FieldSchema("text", zvec.DataType.STRING),
                zvec.FieldSchema("doc_id", zvec.DataType.STRING),
            ],
            vectors=zvec.VectorSchema("embedding", zvec.DataType.VECTOR_FP32, EMBED_DIM),
        )

        if self.index_path.exists():
            shutil.rmtree(self.index_path)

        collection = zvec.create_and_open(path=str(self.index_path), schema=schema)

        print("Rebuilding Zvec index...")
        batch_docs = []
        for idx, (doc_id, title, text) in enumerate(zip(self.doc_ids, self.doc_titles, self.doc_texts)):
            batch_docs.append(
                zvec.Doc(
                    id=str(idx),
                    vectors={"embedding": self.embeddings[idx].tolist()},
                    fields={"doc_id": doc_id, "title": title, "text": text},
                )
            )
            if len(batch_docs) >= 1000:
                collection.insert(batch_docs)
                batch_docs = []
        if batch_docs:
            collection.insert(batch_docs)
        return collection

    def _embed_query(self, query: str) -> list[float]:
        if query in self.query_embedding_cache:
            return self.query_embedding_cache[query]
        response = self.vo.embed(
            [query],
            model=EMBED_MODEL,
            output_dimension=EMBED_DIM,
            input_type="query",
        )
        embedding = response.embeddings[0]
        self.query_embedding_cache[query] = embedding
        return embedding

    def _weighted_rrf(self, ranked_lists: list[tuple[list[int], float]]) -> list[int]:
        fused_scores: dict[int, float] = {}
        for ranked_ids, weight in ranked_lists:
            for rank, doc_idx in enumerate(ranked_ids, start=1):
                fused_scores[doc_idx] = fused_scores.get(doc_idx, 0.0) + weight / (RRF_K + rank)
        return sorted(fused_scores, key=lambda idx: fused_scores[idx], reverse=True)

    def search(self, query: str, config: SearchConfig) -> list[dict]:
        q_emb = self._embed_query(query)

        dense_results = self.collection.query(
            zvec.VectorQuery("embedding", vector=q_emb),
            topk=config.dense_k,
        )
        dense_ranked = [int(res.id) for res in dense_results]
        dense_scores = {int(res.id): float(res.score) for res in dense_results}

        bm25_scores = self.bm25.get_scores(tokenize(query))
        bm25_ranked = [int(idx) for idx in np.argsort(bm25_scores)[::-1][: config.bm25_k]]

        if config.strategy == "legacy_minmax":
            bm25_score_map = {int(idx): float(bm25_scores[idx]) for idx in bm25_ranked}
            candidate_items = sorted(set(dense_scores).union(bm25_score_map))
            dense_vals = [dense_scores.get(idx, 0.0) for idx in candidate_items]
            bm25_vals = [bm25_score_map.get(idx, 0.0) for idx in candidate_items]

            dense_min = min(dense_vals) if dense_vals else 0.0
            dense_max = max(dense_vals) if dense_vals else 0.0
            bm25_min = min(bm25_vals) if bm25_vals else 0.0
            bm25_max = max(bm25_vals) if bm25_vals else 0.0

            def scale(values: list[float], low: float, high: float) -> list[float]:
                if not values or high == low:
                    return [0.0 for _ in values]
                return [(value - low) / (high - low) for value in values]

            norm_dense = scale(dense_vals, dense_min, dense_max)
            norm_bm25 = scale(bm25_vals, bm25_min, bm25_max)
            fused_scores = {
                idx: config.legacy_alpha * norm_dense[pos] + (1.0 - config.legacy_alpha) * norm_bm25[pos]
                for pos, idx in enumerate(candidate_items)
            }
            fused_top_idx = sorted(candidate_items, key=lambda idx: fused_scores[idx], reverse=True)[
                : max(config.rerank_k, config.k_final)
            ]
        else:
            title_scores = self.title_bm25.get_scores(tokenize(query))
            title_ranked = [int(idx) for idx in np.argsort(title_scores)[::-1][: config.title_k]]
            fused_top_idx = self._weighted_rrf(
                [
                    (dense_ranked, config.dense_weight),
                    (bm25_ranked, config.bm25_weight),
                    (title_ranked, config.title_weight),
                ]
            )[: max(config.rerank_k, config.k_final)]
        rerank_docs = [self.documents[idx] for idx in fused_top_idx[: config.rerank_k]]

        rerank_results = self.vo.rerank(
            query,
            rerank_docs,
            model=RERANK_MODEL,
            top_k=min(config.rerank_k, len(rerank_docs)),
        )

        reranked_items = []
        for result in rerank_results.results:
            doc_idx = fused_top_idx[result.index]
            reranked_items.append(
                {
                    "id": self.doc_ids[doc_idx],
                    "title": self.doc_titles[doc_idx],
                    "text": self.doc_texts[doc_idx],
                    "score": float(result.relevance_score),
                }
            )

        return apply_article_cap(reranked_items, config.k_final, config.article_cap)

    def evaluate(self, config: SearchConfig, Ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict[str, float]:
        metrics: dict[str, list[float]] = {}
        for k in Ks:
            metrics[f"hit_rate@{k}"] = []
            metrics[f"recall@{k}"] = []
            metrics[f"precision@{k}"] = []
            metrics[f"mrr@{k}"] = []
        ndcg_scores: list[float] = []

        print(f"Evaluating variant: {config.name}")
        for item in tqdm(self.annotated_queries, desc=f"Evaluating {config.name}"):
            golden_ids = {retrieval["id"] for retrieval in item["retrievals"]}
            preds = [retrieval["id"] for retrieval in self.search(item["query"], config)]

            for k in Ks:
                preds_k = preds[:k]
                hits = len(golden_ids.intersection(preds_k))
                metrics[f"hit_rate@{k}"].append(1.0 if hits > 0 else 0.0)
                metrics[f"recall@{k}"].append(hits / len(golden_ids) if golden_ids else 0.0)
                metrics[f"precision@{k}"].append(hits / k)

                reciprocal_rank = 0.0
                for rank, pred_id in enumerate(preds_k, start=1):
                    if pred_id in golden_ids:
                        reciprocal_rank = 1.0 / rank
                        break
                metrics[f"mrr@{k}"].append(reciprocal_rank)

            gains = [1.0 if pred_id in golden_ids else 0.0 for pred_id in preds[:10]]
            dcg = sum(gain / np.log2(rank + 2) for rank, gain in enumerate(gains))
            ideal = sum(1.0 / np.log2(rank + 2) for rank in range(min(len(golden_ids), 10)))
            ndcg_scores.append(dcg / ideal if ideal > 0 else 0.0)

        summary = {name: float(np.mean(values)) for name, values in metrics.items()}
        summary["ndcg@10"] = float(np.mean(ndcg_scores))
        summary["selection_score"] = (summary["recall@10"] + summary["precision@10"]) / 2.0
        return summary

    def write_submission(self, config: SearchConfig, output_path: Path) -> None:
        print(f"Generating submission with variant: {config.name}")
        test_results = []
        for item in tqdm(self.test_queries, desc="Generating submission"):
            results = self.search(item["query"], config)
            test_results.append(
                {
                    "query_id": item["query_id"],
                    "query": item["query"],
                    "retrievals": [
                        {"id": result["id"], "title": result["title"], "text": result["text"]}
                        for result in results
                    ],
                }
            )

        with output_path.open("w") as handle:
            json.dump(test_results, handle, indent=2)


def print_summary(name: str, summary: dict[str, float]) -> None:
    print("=" * 48)
    print(f"Variant: {name}")
    for metric in (
        "hit_rate@10",
        "recall@10",
        "precision@10",
        "mrr@10",
        "ndcg@10",
        "selection_score",
    ):
        print(f"{metric:<16} {summary[metric]:.4f}")
    print("=" * 48)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval variants and generate a fresh submission JSON.")
    parser.add_argument("--embeddings", default="voyage_embeddings.npy", help="Path to cached document embeddings.")
    parser.add_argument("--index-path", default="zvec_devrev", help="Path to the zvec index directory.")
    parser.add_argument(
        "--output",
        default="test_queries_results.json",
        help="Output path for the generated submission JSON.",
    )
    args = parser.parse_args()

    bench = DevRevSearchBench(
        embeddings_path=PROJECT_ROOT / args.embeddings,
        index_path=PROJECT_ROOT / args.index_path,
    )

    variants = [
        SearchConfig(
            name="legacy_hybrid_article_cap_1",
            strategy="legacy_minmax",
            dense_k=200,
            bm25_k=200,
            rerank_k=150,
            article_cap=1,
            legacy_alpha=0.65,
        ),
        SearchConfig(name="chunk_level_rrf", strategy="rrf", article_cap=None),
    ]

    scored_variants = []
    for config in variants:
        summary = bench.evaluate(config)
        print_summary(config.name, summary)
        scored_variants.append((summary["selection_score"], summary["ndcg@10"], config, summary))
        time.sleep(0.25)

    _, _, best_config, best_summary = max(scored_variants, key=lambda item: (item[0], item[1]))
    print(f"Selected variant: {best_config.name}")
    print_summary(best_config.name, best_summary)

    bench.write_submission(best_config, PROJECT_ROOT / args.output)
    print(f"Submission written to {args.output}")


if __name__ == "__main__":
    main()
