"""
DevRev Search — BM42 Hybrid Retrieval with Haystack + Qdrant.

Hybrid search system combining sparse (BM42) and dense (mxbai-embed-large-v1)
embeddings for information retrieval on the DevRev knowledge base.

Indexing Pipeline:
    DocumentCleaner
    → FastembedSparseDocumentEmbedder (BM42)
    → SentenceTransformersDocumentEmbedder (mxbai-embed-large-v1, 1024-dim)
    → DocumentWriter (Qdrant)

Retrieval Pipeline:
    FastembedSparseTextEmbedder (BM42)
    + SentenceTransformersTextEmbedder (mxbai-embed-large-v1)
    → QdrantHybridRetriever (RRF fusion, top_k=RETRIEVER_TOP_K)
    → adjacent-chunk expansion (±1 KNOWLEDGE_NODE neighbour)
    → SentenceTransformersSimilarityRanker (bge-reranker-v2-m3, top_k=TOP_K)

Evaluation:
    run_annotated_evaluation() scores Recall@10, Precision@10, MRR@10, NDCG@10
    against annotated_queries (291 labelled queries).

Mac-specific notes:
    - BATCH_SIZE = 32   (avoids MPS OOM on Apple Silicon)
    - device = "mps"    (Metal GPU acceleration via PyTorch)
    - sentence-transformers picks up MPS automatically
"""

import json
import logging
import re
from enum import Enum

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from qdrant_client import QdrantClient

# Haystack core imports
from haystack import Document, Pipeline
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.rankers import SentenceTransformersSimilarityRanker
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy

# Qdrant document store and retriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import QdrantHybridRetriever

# Embedders
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack_integrations.components.embedders.fastembed import (
    FastembedSparseDocumentEmbedder,
    FastembedSparseTextEmbedder,
)

# -------------------------------------------
# Logging
# -------------------------------------------
logger = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure logging with a consistent format."""
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)


# -------------------------------------------
# Configuration
# -------------------------------------------

# Mac / Apple Silicon: keep batches small to stay within MPS memory budget.
# On CPU-only machines this can safely go up to 128–256.
BATCH_SIZE = 32

EMBEDDING_DIM = 1024

# Dense bi-encoder (indexing + retrieval)
DENSE_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
DENSE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Sparse bi-encoder (BM42)
SPARSE_MODEL = "Qdrant/bm42-all-minilm-l6-v2-attentions"

# Cross-encoder reranker.
# bge-reranker-v2-m3 outperforms ms-marco-MiniLM on BEIR/MIRACL but is heavier.
# On Mac CPU it is ~3–4× slower than MiniLM; swap if latency matters more than quality:
#   RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Qdrant connection
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "devrev_search"

# Retriever fetches a wide candidate pool; reranker then trims to TOP_K.
# Higher RETRIEVER_TOP_K → better recall ceiling, slower reranker step.
RETRIEVER_TOP_K = 50
TOP_K = 10  # final results returned per query

# Adjacent-chunk expansion: include ±NEIGHBOUR_WINDOW KNOWLEDGE_NODEs around
# each retrieved chunk.  Boosts recall when answers span consecutive nodes.
# Set to 0 to disable.
NEIGHBOUR_WINDOW = 1

# Output files (test_queries submission)
OUTPUT_JSON = "test_queries_results.json"
OUTPUT_PARQUET = "test_queries_results.parquet"


# -------------------------------------------
# Dataset helpers
# -------------------------------------------
class DatasetType(Enum):
    ANNOTATED_QUERIES = "annotated_queries"
    TEST_QUERIES = "test_queries"
    KNOWLEDGE_BASE = "knowledge_base"


def get_dataset(dataset_type: DatasetType) -> pd.DataFrame:
    """Load a split of devrev/search and return as a DataFrame."""
    if dataset_type == DatasetType.ANNOTATED_QUERIES:
        df = load_dataset("devrev/search", "annotated_queries")
        return df["train"].to_pandas()
    elif dataset_type == DatasetType.TEST_QUERIES:
        df = load_dataset("devrev/search", "test_queries")
        return df["test"].to_pandas()
    elif dataset_type == DatasetType.KNOWLEDGE_BASE:
        df = load_dataset("devrev/search", "knowledge_base")
        return df["corpus"].to_pandas()
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


# -------------------------------------------
# Text cleaning helpers
# -------------------------------------------
_BYTE_PREFIX = re.compile(r"""^b['\"](.*)['\"]\\s*$""", re.DOTALL)


def clean_text(raw: str) -> str:
    """Strip Python byte-string artifacts and normalise whitespace.

    Some knowledge-base chunks are serialised as  b'actual text here'
    (a Python bytes repr leaked into the dataset).  This strips the wrapper
    so the embedder sees clean prose rather than a Python literal.
    """
    if not isinstance(raw, str):
        return ""
    m = _BYTE_PREFIX.match(raw.strip())
    if m:
        # Unescape common escape sequences preserved in the repr
        text = m.group(1)
        text = text.replace("\\n", "\n").replace("\\t", "\t")
        text = text.replace("\'", "'").replace('\\"', '"')
        return text.strip()
    return raw.strip()


def build_content(title: str, text: str) -> str:
    """Concatenate title + text for indexing.

    Prepending the title gives the embedding model full article context and
    ensures product-specific terms (e.g. "AirSync", "snap-in") appear in
    every chunk, not just the first one.
    """
    title = (title or "").strip()
    text = clean_text(text or "")
    if title:
        return f"{title}\n\n{text}"
    return text


# -------------------------------------------
# Adjacent-chunk expansion
# -------------------------------------------
_NODE_RE = re.compile(r"^(ART-\d+)_KNOWLEDGE_NODE-(\d+)$")


def expand_with_neighbours(
    doc_ids: list[str],
    id_to_doc: dict[str, Document],
    window: int = NEIGHBOUR_WINDOW,
) -> list[Document]:
    """Expand a retrieved set with adjacent KNOWLEDGE_NODE chunks.

    IDs follow the pattern  ART-<num>_KNOWLEDGE_NODE-<idx>.
    For each retrieved chunk we also include ±window neighbours from the same
    article, then deduplicate while preserving retrieval-rank order.

    Args:
        doc_ids: Ordered list of retrieved document IDs (best first).
        id_to_doc: Mapping of all indexed doc_id → Document.
        window: How many adjacent nodes to include on each side.

    Returns:
        Deduplicated list of Documents (retrieved + neighbours), original
        order preserved, neighbours inserted immediately after their source.
    """
    if window == 0:
        return [id_to_doc[d] for d in doc_ids if d in id_to_doc]

    seen: set[str] = set()
    result: list[Document] = []

    for doc_id in doc_ids:
        m = _NODE_RE.match(doc_id)
        if not m:
            # ID does not follow the expected pattern — include as-is
            if doc_id not in seen and doc_id in id_to_doc:
                seen.add(doc_id)
                result.append(id_to_doc[doc_id])
            continue

        art_prefix, node_idx = m.group(1), int(m.group(2))

        # Build candidate IDs: retrieved chunk + its neighbours
        candidates = [
            f"{art_prefix}_KNOWLEDGE_NODE-{node_idx + delta}"
            for delta in range(-window, window + 1)
        ]

        for cid in candidates:
            if cid not in seen and cid in id_to_doc:
                seen.add(cid)
                result.append(id_to_doc[cid])

    return result


# -------------------------------------------
# Document store
# -------------------------------------------
def get_document_store(recreate: bool = False) -> QdrantDocumentStore:
    """Get or create a QdrantDocumentStore for hybrid search.

    When recreate=True we explicitly drop the collection via raw QdrantClient
    first.  This bypasses a silent failure in recreate_index=True when the
    qdrant-client version (1.17) is newer than the server (1.9).
    """
    if recreate:
        raw_client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            check_compatibility=False,
        )
        existing = {c.name for c in raw_client.get_collections().collections}
        if COLLECTION_NAME in existing:
            raw_client.delete_collection(COLLECTION_NAME)
            logger.info(f"Dropped existing Qdrant collection '{COLLECTION_NAME}'.")
        raw_client.close()

    return QdrantDocumentStore(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        index=COLLECTION_NAME,
        embedding_dim=EMBEDDING_DIM,
        use_sparse_embeddings=True,
        recreate_index=recreate,
        similarity="cosine",
        return_embedding=False,
    )


# -------------------------------------------
# Indexing pipeline
# -------------------------------------------
def build_indexing_pipeline(document_store: QdrantDocumentStore) -> Pipeline:
    """Build the indexing pipeline.

    DocumentCleaner
      → FastembedSparseDocumentEmbedder  (BM42 sparse vectors)
      → SentenceTransformersDocumentEmbedder  (mxbai dense, 1024-dim, MPS)
      → DocumentWriter
    """
    cleaner = DocumentCleaner(
        remove_empty_lines=True,
        remove_extra_whitespaces=True,
        remove_repeated_substrings=False,
    )

    sparse_doc_embedder = FastembedSparseDocumentEmbedder(
        model=SPARSE_MODEL,
        batch_size=BATCH_SIZE,
    )

    dense_doc_embedder = SentenceTransformersDocumentEmbedder(
        model=DENSE_MODEL,
        batch_size=BATCH_SIZE,
    )

    writer = DocumentWriter(
        document_store=document_store,
        policy=DuplicatePolicy.OVERWRITE,
    )

    indexing = Pipeline()
    indexing.add_component("cleaner", cleaner)
    indexing.add_component("sparse_doc_embedder", sparse_doc_embedder)
    indexing.add_component("dense_doc_embedder", dense_doc_embedder)
    indexing.add_component("writer", writer)

    indexing.connect("cleaner.documents", "sparse_doc_embedder.documents")
    indexing.connect("sparse_doc_embedder.documents", "dense_doc_embedder.documents")
    indexing.connect("dense_doc_embedder.documents", "writer.documents")

    return indexing


def index_knowledge_base(document_store: QdrantDocumentStore) -> None:
    """Load, clean, and index the DevRev knowledge base.

    Improvements applied at index time
    ------------------------------------
    1. b\'...' byte-string stripping  — clean_text()
    2. Title + text concatenation      — build_content()
       Prepending the title ensures every chunk carries article-level keywords
       ("AirSync", "snap-in") that may only appear in the title of later nodes.
    3. Null/empty row filtering        — avoids DocumentCleaner warning spam.

    Metadata stored
    ---------------
    doc_id   : original corpus ID  (e.g. ART-4216_KNOWLEDGE_NODE-26)
    title    : article title        (used for output and neighbour expansion)
    art_id   : article prefix       (e.g. ART-4216, for grouping/boosting)
    node_idx : integer chunk index  (for neighbour expansion ordering)
    source   : provenance string
    """
    kb_df = get_dataset(DatasetType.KNOWLEDGE_BASE)

    # Filter nulls upfront — DocumentCleaner would skip them but logs a
    # warning for every single one, flooding the output.
    original_count = len(kb_df)
    kb_df = kb_df[kb_df["text"].notna() & (kb_df["text"].str.strip() != "")]
    dropped = original_count - len(kb_df)
    if dropped:
        logger.warning(f"Dropped {dropped} row(s) with null/empty text.")

    documents: list[Document] = []
    for _, row in kb_df.iterrows():
        doc_id: str = row["id"]
        title: str = row.get("title", "") or ""
        raw_text: str = row["text"]

        # Parse ART-XXXX_KNOWLEDGE_NODE-YY for metadata
        m = _NODE_RE.match(doc_id)
        art_id = m.group(1) if m else doc_id
        node_idx = int(m.group(2)) if m else 0

        content = build_content(title, raw_text)

        doc = Document(
            content=content,
            meta={
                "doc_id": doc_id,
                "title": title,
                "art_id": art_id,
                "node_idx": node_idx,
                "source": "devrev/search::knowledge_base",
            },
        )
        documents.append(doc)

    logger.info(f"Indexing {len(documents):,} documents …")
    indexing_pipeline = build_indexing_pipeline(document_store)
    indexing_pipeline.run({"cleaner": {"documents": documents}})
    logger.info(f"Indexed {len(documents):,} documents into Qdrant.")


# -------------------------------------------
# Retrieval pipeline
# -------------------------------------------
def build_retrieval_pipeline(document_store: QdrantDocumentStore) -> Pipeline:
    """Build the hybrid retrieval + reranking pipeline.

    query
      ├─ FastembedSparseTextEmbedder  → query_sparse_embedding ─┐
      └─ SentenceTransformersTextEmbedder → query_embedding    ─┤
                                                                ↓
                               QdrantHybridRetriever (RRF, top_k=RETRIEVER_TOP_K)
                                                                ↓
                               [adjacent-chunk expansion in search()]
                                                                ↓
                               SentenceTransformersSimilarityRanker (top_k=TOP_K)
    """
    sparse_text_embedder = FastembedSparseTextEmbedder(
        model=SPARSE_MODEL,
    )

    dense_text_embedder = SentenceTransformersTextEmbedder(
        model=DENSE_MODEL,
        prefix=DENSE_QUERY_PREFIX,
    )

    retriever = QdrantHybridRetriever(
        document_store=document_store,
        top_k=RETRIEVER_TOP_K,
    )

    # Cross-encoder reranker: reads query + document text jointly with full
    # cross-attention — significantly sharper relevance signals than bi-encoder
    # cosine similarity alone.
    reranker = SentenceTransformersSimilarityRanker(
        model=RERANKER_MODEL,
        top_k=TOP_K,
    )

    retrieval = Pipeline()
    retrieval.add_component("sparse_text_embedder", sparse_text_embedder)
    retrieval.add_component("dense_text_embedder", dense_text_embedder)
    retrieval.add_component("retriever", retriever)
    retrieval.add_component("reranker", reranker)

    retrieval.connect(
        "sparse_text_embedder.sparse_embedding",
        "retriever.query_sparse_embedding",
    ) 
    retrieval.connect(
        "dense_text_embedder.embedding",
        "retriever.query_embedding",
    )
    retrieval.connect("retriever.documents", "reranker.documents")

    return retrieval


def search(
    query: str,
    retrieval_pipeline: Pipeline,
    id_to_doc: dict[str, Document] | None = None,
) -> list[dict]:
    """Run a hybrid search query and return ranked results.

    Flow:
      1. Embed query (sparse + dense) and retrieve RETRIEVER_TOP_K candidates.
      2. Expand each candidate with ±NEIGHBOUR_WINDOW adjacent chunks from the
         same article (improves recall when answers span multiple nodes).
      3. Rerank the expanded set with the cross-encoder (top_k=TOP_K).

    Args:
        query: Natural-language search query.
        retrieval_pipeline: Warmed-up Haystack Pipeline.
        id_to_doc: Optional mapping of doc_id → Document for neighbour
                   expansion.  If None, expansion is skipped.

    Returns:
        List of dicts with keys: id, text, title  (TOP_K entries).
    """
    result = retrieval_pipeline.run(
        {
            "sparse_text_embedder": {"text": query},
            "dense_text_embedder": {"text": query},
            "reranker": {"query": query},
        }
    )

    reranked_docs: list[Document] = result["reranker"]["documents"]

    # --- Adjacent-chunk expansion (post-rerank) ----------------------------
    # We expand AFTER the reranker so the cross-encoder scores the clean
    # retrieval set.  Neighbours are appended at the end only if they are not
    # already present in the reranked top-k.
    if id_to_doc and NEIGHBOUR_WINDOW > 0:
        ranked_ids = [d.meta["doc_id"] for d in reranked_docs]
        expanded = expand_with_neighbours(ranked_ids, id_to_doc, NEIGHBOUR_WINDOW)
        # Trim back to TOP_K after expansion
        final_docs = expanded[:TOP_K]
    else:
        final_docs = reranked_docs[:TOP_K]

    return [
        {
            "id": doc.meta.get("doc_id", ""),
            "text": doc.content or "",
            "title": doc.meta.get("title", ""),
        }
        for doc in final_docs
    ]


# -------------------------------------------
# Build id_to_doc lookup for neighbour expansion
# -------------------------------------------
def build_id_to_doc(document_store: QdrantDocumentStore) -> dict[str, Document]:
    """Fetch all documents from Qdrant and build a doc_id → Document mapping.

    Used by expand_with_neighbours() to look up adjacent chunks without
    hitting Qdrant for every neighbour individually.

    Note: loads all 65 k documents into memory (~500 MB).  Fine on a Mac with
    ≥16 GB RAM; reduce NEIGHBOUR_WINDOW to 0 if memory is tight.
    """
    logger.info("Building id→doc lookup table for neighbour expansion …")
    all_docs = document_store.filter_documents()
    mapping = {doc.meta["doc_id"]: doc for doc in all_docs if "doc_id" in doc.meta}
    logger.info(f"Lookup table ready: {len(mapping):,} documents.")
    return mapping


# -------------------------------------------
# Test-query evaluation (submission output)
# -------------------------------------------
def run_evaluation(
    retrieval_pipeline: Pipeline,
    id_to_doc: dict[str, Document] | None = None,
    output_json: str = OUTPUT_JSON,
    output_parquet: str = OUTPUT_PARQUET,
) -> list[dict]:
    """Run retrieval on all test queries and save submission files."""
    test_df = get_dataset(DatasetType.TEST_QUERIES)
    test_results: list[dict] = []

    for _, row in tqdm(
        test_df.iterrows(), total=len(test_df), desc="Evaluating test queries"
    ):
        retrievals = search(row["query"], retrieval_pipeline, id_to_doc)
        test_results.append(
            {
                "query_id": row["query_id"],
                "query": row["query"],
                "retrievals": retrievals,
            }
        )

    with open(output_json, "w") as f:
        json.dump(test_results, f, indent=2)
    logger.info(f"Results saved to {output_json}")

    pd.DataFrame(test_results).to_parquet(output_parquet, index=False)
    logger.info(f"Results saved to {output_parquet}")

    logger.info(f"Total queries: {len(test_results)} | Retrievals/query: {TOP_K}")
    return test_results


# -------------------------------------------
# Annotated-query evaluation (recall / precision)
# -------------------------------------------
def run_annotated_evaluation(
    retrieval_pipeline: Pipeline,
    id_to_doc: dict[str, Document] | None = None,
) -> dict:
    """Score the pipeline against annotated_queries (291 labelled pairs).

    Uses ranx for standard IR metrics.  Install with:
        pip install ranx

    Metrics computed
    ----------------
    Recall@10    — fraction of golden docs retrieved in top-10  (primary)
    Precision@10 — fraction of top-10 that are golden docs
    MRR@10       — mean reciprocal rank of first golden doc
    NDCG@10      — rank-discounted cumulative gain

    Returns:
        Dict mapping metric name → float score.
    """
    try:
        from ranx import Qrels, Run, evaluate
    except ImportError:
        logger.error("ranx is not installed.  Run: pip install ranx")
        raise

    annotated_df = get_dataset(DatasetType.ANNOTATED_QUERIES)

    # Build ground-truth Qrels: {query_id: {doc_id: 1}}
    qrels_dict: dict[str, dict[str, int]] = {}
    for _, row in annotated_df.iterrows():
        qid = str(row["query_id"])
        qrels_dict[qid] = {str(r["id"]): 1 for r in row["retrievals"]}

    qrels = Qrels(qrels_dict)

    # Build Run: {query_id: {doc_id: score}}
    # Scores are descending rank positions (TOP_K, TOP_K-1, …, 1).
    run_dict: dict[str, dict[str, float]] = {}
    for _, row in tqdm(
        annotated_df.iterrows(),
        total=len(annotated_df),
        desc="Scoring annotated queries",
    ):
        qid = str(row["query_id"])
        retrievals = search(row["query"], retrieval_pipeline, id_to_doc)
        run_dict[qid] = {
            str(r["id"]): float(TOP_K - i)
            for i, r in enumerate(retrievals)
        }

    run = Run(run_dict)

    metrics = evaluate(
        qrels,
        run,
        ["recall@10", "precision@10", "mrr@10", "ndcg@10"],
    )

    logger.info("=" * 50)
    logger.info("Retrieval evaluation — annotated_queries (291 queries)")
    logger.info("=" * 50)
    for k, v in metrics.items():
        logger.info(f"  {k:<20} {v:.4f}")
    logger.info("=" * 50)

    return dict(metrics)


# -------------------------------------------
# Main
# -------------------------------------------
def main() -> None:
    """End-to-end flow.

    Steps
    -----
    1. Drop + recreate Qdrant collection and index the knowledge base.
       → Set recreate=False on subsequent runs to skip re-indexing.
    2. Build retrieval pipeline (sparse + dense + reranker).
    3. Build id→doc lookup for neighbour expansion.
    4. Score against annotated_queries (Recall / Precision / MRR / NDCG).
    5. Run on test_queries and save submission JSON + Parquet.
    """
    configure_logging(level=logging.INFO)
    logger.info("Starting DevRev Search …")

    try:
        # ── Step 1 : Index ─────────────────────────────────────────────────
        # Flip to recreate=False once the collection is already indexed.
        document_store = get_document_store(recreate=False)
        index_knowledge_base(document_store)

        # ── Step 2 : Retrieval pipeline ────────────────────────────────────
        retrieval_pipeline = build_retrieval_pipeline(document_store)

        # ── Step 3 : Neighbour-expansion lookup ────────────────────────────
        id_to_doc = build_id_to_doc(document_store)

        # ── Step 4 : Score on annotated_queries ────────────────────────────
        logger.info("Running evaluation on annotated_queries …")
        run_annotated_evaluation(retrieval_pipeline, id_to_doc)

        # ── Step 5 : Generate test_queries submission ───────────────────────
        logger.info("Generating test_queries submission …")
        run_evaluation(retrieval_pipeline, id_to_doc)

        logger.info("Done.")
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
