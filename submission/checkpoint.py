# =============================================================================
# checkpoint.py — atomic JSON checkpoint manager
# =============================================================================
"""
State file (checkpoints/pipeline_state.json):

    {
      "phases": {
        "datasets_downloaded": true,
        "embeddings_computed":  false,
        ...
      },
      "meta": {
        "corpus_size": 12345,
        "embed_model": "Qwen/Qwen3-Embedding-0.6B",
        "datasets_downloaded_completed_at": "2025-06-10T12:34:56+00:00",
        ...
      }
    }

Writes are atomic: we write to a `.tmp` file then rename, so a crash
mid-write never corrupts the checkpoint.

Usage
-----
    from checkpoint import Checkpoint

    ck = Checkpoint()

    if not ck.done("qdrant_indexed"):
        do_indexing()
        ck.mark_done("qdrant_indexed", corpus_size=12345)

    ck.summary()        # pretty-print all phase statuses

    # Force a phase (and everything after) to re-run:
    ck.reset_from("embeddings_computed")
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from config import CHECKPOINT_FILE, PHASES
from logger import get_logger

log = get_logger(__name__)


class Checkpoint:
    def __init__(self, path: Path = CHECKPOINT_FILE) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._state = self._load()
        log.debug("Checkpoint loaded from %s", self.path)

    # ── I/O ──────────────────────────────────────────────────────────────────
    def _load(self) -> dict:
        if self.path.exists():
            with open(self.path, encoding="utf-8") as fh:
                state = json.load(fh)
            # Back-fill any phases added to config after the file was created
            for phase in PHASES:
                state["phases"].setdefault(phase, False)
            log.debug("Loaded existing checkpoint: %s", state["phases"])
            return state

        log.info("No checkpoint found — starting fresh at %s", self.path)
        return {"phases": {p: False for p in PHASES}, "meta": {}}

    def _save(self) -> None:
        """Atomic write: tmp file → rename."""
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self._state, fh, indent=2)
        os.replace(tmp, self.path)
        log.debug("Checkpoint saved to %s", self.path)

    # ── Phase control ─────────────────────────────────────────────────────────
    def done(self, phase: str) -> bool:
        """Return True if *phase* has already been completed."""
        result = self._state["phases"].get(phase, False)
        log.debug("Phase '%s' done=%s", phase, result)
        return result

    def mark_done(self, phase: str, **meta) -> None:
        """Mark *phase* as completed and persist.  Extra kwargs go into meta."""
        self._state["phases"][phase] = True
        ts = datetime.now(timezone.utc).isoformat()
        self._state["meta"][f"{phase}_completed_at"] = ts
        for k, v in meta.items():
            self._state["meta"][k] = v
        self._save()
        log.info("✓ Phase '%s' marked done  (ts=%s)", phase, ts)

    def reset(self, phase: str) -> None:
        """Force a single phase to re-run on next execution."""
        self._state["phases"][phase] = False
        self._save()
        log.info("↺ Phase '%s' reset — will re-run", phase)

    def reset_from(self, phase: str) -> None:
        """Reset *phase* and every phase after it in PHASES order."""
        if phase not in PHASES:
            raise ValueError(f"Unknown phase '{phase}'. Valid: {PHASES}")
        idx = PHASES.index(phase)
        for p in PHASES[idx:]:
            self._state["phases"][p] = False
        self._save()
        log.info("↺ Phases reset from '%s' onwards: %s", phase, PHASES[idx:])

    # ── Metadata ──────────────────────────────────────────────────────────────
    def set_meta(self, key: str, value) -> None:
        self._state["meta"][key] = value
        self._save()
        log.debug("Meta set: %s = %r", key, value)

    def get_meta(self, key: str, default=None):
        return self._state["meta"].get(key, default)

    # ── Diagnostics ───────────────────────────────────────────────────────────
    def summary(self) -> None:
        print("\n── Checkpoint Summary ──────────────────────────────")
        for phase in PHASES:
            ts_key = f"{phase}_completed_at"
            ts     = self._state["meta"].get(ts_key, "")
            status = "✓ done   " if self._state["phases"].get(phase) else "✗ pending"
            suffix = f"  @ {ts}" if ts else ""
            print(f"  {status}  {phase}{suffix}")
        print("────────────────────────────────────────────────────\n")
