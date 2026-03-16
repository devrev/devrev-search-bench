"""
DevRev Search — BM42 Hybrid Retrieval with Haystack + Qdrant.

Hybrid search system combining sparse (BM42) and dense (mxbai-embed-large-v1)
embeddings for information retrieval on the DevRev knowledge base.

Indexing Pipeline:
    DocumentCleaner → FastembedSparseDocumentEmbedder (BM42)
    → SentenceTransformersDocumentEmbedder (1024-dim)
    → DocumentWriter (Qdrant)

Retrieval Pipeline:
    FastembedSparseTextEmbedder (BM42) + SentenceTransformersTextEmbedder
    → QdrantHybridRetriever (RRF fusion)
    → SentenceTransformersSimilarityRanker (bge-reranker-v2-m3, cross-encoder)
"""

from __future__ import annotations

import json
import logging
from enum import Enum

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from qdrant_client import QdrantClient

# Haystack
from haystack import Document, Pipeline
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy

# Qdrant document store (pip install qdrant-haystack)
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import QdrantHybridRetriever

# HF Embedders (pip install sentence-transformers)
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)

# FastEmbed Embedders (pip install fastembed-haystack)
from haystack_integrations.components.embedders.fastembed import (
    FastembedSparseDocumentEmbedder,
    FastembedSparseTextEmbedder,
)

from haystack.components.rankers import SentenceTransformersSimilarityRanker

# -------------------------------------------
# Logging configuration
# -------------------------------------------
logger = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure logging with a consistent format.
    
    Args:
        level: Logging level (default: INFO)
    """
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
BATCH_SIZE = 256
EMBEDDING_DIM = 1024

# Dense model
DENSE_MODEL = "mixedbread-ai/mxbai-embed-large-v1"

# BGE query prefix recommended by the model authors
DENSE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Sparse model — BM42
SPARSE_MODEL = "Qdrant/bm42-all-minilm-l6-v2-attentions"

# Ranker Model
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Qdrant connection
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

# Number of documents to retrieve per query
RETRIEVER_TOP_K = 30
TOP_K = 10

# Output files
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
    """Load a split of the devrev/search dataset and return as a DataFrame.

    Args:
        dataset_type: Enum specifying which dataset split to load.

    Returns:
        DataFrame containing the requested dataset split.

    Raises:
        ValueError: If dataset_type is not recognized.
    """
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
# Document store
# -------------------------------------------
def get_document_store(recreate: bool = False) -> QdrantDocumentStore:
    """
    Get or create a QdrantDocumentStore configured for hybrid search.

    Args:
        recreate: If True, delete and recreate the collection. Uses raw
            QdrantClient to bypass version-mismatch bugs with recreate_index.

    Returns:
        Configured QdrantDocumentStore instance.
    """
    collection_name = "devrev_search"

    if recreate:
        raw_client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            check_compatibility=False,
        )
        existing = {c.name for c in raw_client.get_collections().collections}
        if collection_name in existing:
            raw_client.delete_collection(collection_name)
            logger.info(f"Dropped existing Qdrant collection '{collection_name}'.")
        raw_client.close()

    return QdrantDocumentStore(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        index=collection_name,
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
    """
    Build the document indexing pipeline
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
    """
    Load and index the DevRev knowledge base
    """
    kb_df = get_dataset(DatasetType.KNOWLEDGE_BASE)

    # Drop rows with null/empty text upfront (DocumentCleaner would skip them anyway)
    original_count = len(kb_df)
    kb_df = kb_df[kb_df["text"].notna() & (kb_df["text"].str.strip() != "")]
    dropped = original_count - len(kb_df)
    if dropped:
        logger.warning(f"Dropped {dropped} row(s) with null/empty 'text' before indexing.")

    documents: list[Document] = []
    for _, row in kb_df.iterrows():
        doc = Document(
            content=row["text"],
            meta={
                "doc_id": row["id"],
                "title": row["title"],
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
    """
    Build the hybrid query retrieval pipeline
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


def search(query: str, retrieval_pipeline: Pipeline) -> list[dict]:
    """
    Execute a hybrid search query and return ranked results
    """
    result = retrieval_pipeline.run(
        {
            "sparse_text_embedder": {"text": query},
            "dense_text_embedder": {"text": query},
            "reranker": {"query": query},
        }
    )

    retrievals = []
    for doc in result["reranker"]["documents"]:
        retrievals.append(
            {
                "id": doc.meta.get("doc_id", ""),
                "text": doc.content or "",
                "title": doc.meta.get("title", ""),
            }
        )

    return retrievals


# -------------------------------------------
# Evaluation
# -------------------------------------------
def run_evaluation(
    retrieval_pipeline: Pipeline,
    output_json: str = OUTPUT_JSON,
    output_parquet: str = OUTPUT_PARQUET,
) -> list[dict]:
    """
    Run hybrid retrieval on all test queries and save results
    """
    test_df = get_dataset(DatasetType.TEST_QUERIES)

    test_results: list[dict] = []

    for _, row in tqdm(
        test_df.iterrows(), total=len(test_df), desc="Evaluating test queries"
    ):
        query_id = row["query_id"]
        query = row["query"]

        retrievals = search(query, retrieval_pipeline)

        test_results.append(
            {
                "query_id": query_id,
                "query": query,
                "retrievals": retrievals,
            }
        )

    with open(output_json, "w") as f:
        json.dump(test_results, f, indent=2)
    logger.info(f"Results saved to {output_json}")

    results_df = pd.DataFrame(test_results)
    results_df.to_parquet(output_parquet, index=False)
    logger.info(f"Results also saved to {output_parquet}")

    # Display summary
    logger.info("=" * 60)
    logger.info("Test Queries Results Summary")
    logger.info("=" * 60)
    logger.info(f"Total queries    : {len(test_results)}")
    logger.info(f"Retrievals/query : {TOP_K}")
    logger.info(f"\nOutput files:")
    logger.info(f"  - {output_json}")
    logger.info(f"  - {output_parquet}")
    logger.info("\nFormat matches annotated_queries structure:")
    logger.info("  - query_id  : string")
    logger.info("  - query     : string")
    logger.info("  - retrievals: list of {id, text, title}")

    return test_results


# -------------------------------------------
# Main function
# -------------------------------------------
def main() -> None:
    """
    Run the complete search pipeline: index and retrieve
    """
    configure_logging(level=logging.INFO)
    logger.info("Starting DevRev Search benchmark...")
    
    try:
        # Index knowledge base
        logger.info("Initializing document store...")
        document_store = get_document_store(recreate=False)
        index_knowledge_base(document_store)

        # Build retrieval pipeline
        logger.info("Building retrieval pipeline...")
        retrieval_pipeline = build_retrieval_pipeline(document_store)

        # Evaluate on test queries
        logger.info("Running evaluation on test queries...")
        run_evaluation(retrieval_pipeline)
        logger.info("Evaluation completed successfully.")
    except Exception as e:
        logger.error(f"Error during pipeline execution: {e}", exc_info=True)
        raise


# -------------------------------------------
# Driver code
# -------------------------------------------
if __name__ == "__main__":
    main()
