# =============================================================================
# dataset_loader.py — download & cache DevRev HuggingFace splits
# =============================================================================
"""
Public API
----------
    from dataset_loader import load_datasets

    dfs = load_datasets(ck)
    # dfs["annotated"]  → pd.DataFrame  (query + golden retrievals)
    # dfs["kb"]         → pd.DataFrame  (knowledge-base corpus)
    # dfs["test"]       → pd.DataFrame  (held-out test queries)
"""

from __future__ import annotations

import pandas as pd
from datasets import load_dataset
from pathlib import Path

from checkpoint import Checkpoint
from config import HF_DATASET_ID, HF_TOKEN, DATA_DIR
from logger import get_logger

log = get_logger(__name__)

_SPLITS: dict[str, tuple[str, str]] = {
    "annotated": ("annotated_queries", "train"),
    "kb": ("knowledge_base", "corpus"),
    "test": ("test_queries", "test"),
}


def _parquet_path(key: str) -> Path:
    return DATA_DIR / f"{key}.parquet"


def _download_split(config_name: str, split: str) -> pd.DataFrame:
    log.info("Downloading split: %s [%s] from %s …", config_name, split, HF_DATASET_ID)
    ds = load_dataset(
        HF_DATASET_ID,
        config_name,
        split=split,
        token=HF_TOKEN,
    )
    df = ds.to_pandas()
    log.debug("  → %d rows, columns: %s", len(df), list(df.columns))
    return df


def _save(df: pd.DataFrame, key: str) -> None:
    path = _parquet_path(key)
    df.to_parquet(path, index=False)
    log.info("Saved %s → %s  (%d rows)", key, path, len(df))


def _load_cached(key: str) -> pd.DataFrame:
    path = _parquet_path(key)
    log.info("Loading %s from cache: %s", key, path)
    df = pd.read_parquet(path)
    log.debug("  → %d rows", len(df))
    return df


def load_datasets(ck: Checkpoint) -> dict[str, pd.DataFrame]:
    """
    Download all three dataset splits from HuggingFace and cache them
    as parquet files.  On subsequent runs the cached files are used directly.

    Parameters
    ----------
    ck : Checkpoint
        Shared pipeline checkpoint; marks 'datasets_downloaded' when done.

    Returns
    -------
    dict with keys "annotated", "kb", "test" → pd.DataFrame
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if ck.done("datasets_downloaded"):
        log.info("Phase 'datasets_downloaded' already done — loading from cache")
        return {key: _load_cached(key) for key in _SPLITS}

    log.info("=== Phase: datasets_downloaded ===")
    dfs: dict[str, pd.DataFrame] = {}

    for key, (config_name, split) in _SPLITS.items():
        df = _download_split(config_name, split)
        _save(df, key)
        dfs[key] = df

    ck.mark_done(
        "datasets_downloaded",
        corpus_size=len(dfs["kb"]),
        annotated_size=len(dfs["annotated"]),
        test_size=len(dfs["test"]),
    )

    log.info(
        "Datasets ready — kb=%d  annotated=%d  test=%d",
        len(dfs["kb"]),
        len(dfs["annotated"]),
        len(dfs["test"]),
    )
    return dfs
