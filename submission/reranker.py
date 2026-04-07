# =============================================================================
# reranker.py — Qwen3-Reranker-0.6B cross-encoder
# =============================================================================
"""
Qwen3-Reranker-0.6B is a causal LM reranker — it is NOT a bi-encoder.
It must be run through the raw transformers AutoModelForCausalLM API,
NOT sentence-transformers CrossEncoder.

How it works (from the Qwen3 reranker model card):
  1. Format each (query, document) pair with a fixed chat template.
  2. Run a forward pass and read the logit for token "yes" vs "no"
     at the final position.
  3. Apply softmax over [yes_logit, no_logit] → relevance probability.
  4. Sort candidates by probability descending.

Public API
----------
    from reranker import Reranker

    rr = Reranker()
    ranked = rr.rerank(query, candidates, top_k=10)
    # candidates → list of {"doc_id", "title", "text", ...}
    # ranked    → same list, sorted by relevance score, length=top_k
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import RERANKER_MODEL_ID, RERANKER_MAX_LEN, HF_TOKEN
from logger import get_logger

log = get_logger(__name__)

# Tokens the reranker scores against (from Qwen3-Reranker model card)
_YES = "yes"
_NO = "no"

# Instruction used by the model card for passage retrieval
_TASK_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)

_RERANK_BATCH = 16  # (query, doc) pairs per forward pass; lower on < 8 GB VRAM


def _format_pair(query: str, document: str, instruction: str) -> str:
    """
    Build the chat-template prompt the Qwen3-Reranker expects.
    Format from the official model card.
    """
    return (
        f"<|im_start|>system\n"
        f"Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
        f'Note that the answer can only be "yes" or "no".<|im_end|>\n'
        f"<|im_start|>user\n"
        f"<Instruct>: {instruction}\n"
        f"<Query>: {query}\n"
        f"<Document>: {document}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"<think>\n\n</think>\n\n"
    )


class Reranker:
    """
    Cross-encoder reranker backed by Qwen3-Reranker-0.6B.

    The model is loaded once per process (class-level singleton).
    """

    _tokenizer: AutoTokenizer | None = None
    _model: AutoModelForCausalLM | None = None
    _yes_id: int | None = None
    _no_id: int | None = None

    def __init__(self) -> None:
        self._tokenizer, self._model, self._yes_id, self._no_id = self._load()
        self._device = next(self._model.parameters()).device
        log.info("Reranker on device: %s", self._device)

    @classmethod
    def _load(cls):
        if cls._model is None:
            log.info("Loading reranker: %s …", RERANKER_MODEL_ID)
            tokenizer = AutoTokenizer.from_pretrained(
                RERANKER_MODEL_ID,
                padding_side="left",
                token=HF_TOKEN,
            )
            # Pick dtype: float16 on CUDA/MPS, float32 on CPU
            if torch.cuda.is_available():
                _dtype = torch.float16
            elif torch.backends.mps.is_available():
                _dtype = torch.float16  # MPS supports float16
            else:
                _dtype = torch.float32

            model = AutoModelForCausalLM.from_pretrained(
                RERANKER_MODEL_ID,
                dtype=_dtype,  # dtype= replaces deprecated torch_dtype=
                device_map="auto",
                token=HF_TOKEN,
            )
            model.eval()

            yes_id = tokenizer.convert_tokens_to_ids(_YES)
            no_id = tokenizer.convert_tokens_to_ids(_NO)
            log.info("Reranker loaded  yes_token_id=%d  no_token_id=%d", yes_id, no_id)
            cls._tokenizer = tokenizer
            cls._model = model
            cls._yes_id = yes_id
            cls._no_id = no_id

        return cls._tokenizer, cls._model, cls._yes_id, cls._no_id

    def _score_batch(self, prompts: list[str]) -> list[float]:
        """
        Run one batch of formatted (query, doc) prompts through the model and
        return the P(yes) relevance score for each.
        """
        inputs = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=RERANKER_MAX_LEN,
        ).to(self._device)

        with torch.no_grad():
            logits = self._model(**inputs).logits  # (B, seq_len, vocab)

        # Last token logits → slice yes/no
        last_logits = logits[:, -1, :]  # (B, vocab)
        yes_no = last_logits[:, [self._yes_id, self._no_id]]  # (B, 2)
        probs = F.softmax(yes_no, dim=-1)[:, 0]  # P(yes)

        scores = probs.float().cpu().tolist()
        log.debug(
            "_score_batch: %d prompts → scores=%s",
            len(prompts),
            [f"{s:.4f}" for s in scores],
        )
        return scores

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int | None = None,
        instruction: str = _TASK_INSTRUCTION,
    ) -> list[dict]:
        """
        Rerank *candidates* by relevance to *query*.

        Parameters
        ----------
        query      : str       — the search query
        candidates : list[dict]— each dict must have at least {"title", "text"}
        top_k      : int|None  — keep only the top-k after reranking (None = all)
        instruction: str       — task instruction for the reranker

        Returns
        -------
        list[dict] — candidates sorted by rerank_score descending,
                     with a new "rerank_score" key added to each dict.
        """
        if not candidates:
            log.warning("rerank() called with empty candidate list — returning []")
            return []

        log.info("Reranking %d candidates for query: %.80s …", len(candidates), query)

        prompts = [
            _format_pair(
                query,
                f"{c.get('title', '')}\n\n{c.get('text', '')}",
                instruction,
            )
            for c in candidates
        ]

        scores: list[float] = []
        for i in tqdm(range(0, len(prompts), _RERANK_BATCH), desc="Reranking"):
            batch = prompts[i : i + _RERANK_BATCH]
            scores.extend(self._score_batch(batch))

        # Attach scores and sort
        for cand, score in zip(candidates, scores):
            cand["rerank_score"] = score

        ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)

        if top_k is not None:
            ranked = ranked[:top_k]

        log.info(
            "Reranking done — top score=%.4f  bottom score=%.4f",
            ranked[0]["rerank_score"] if ranked else 0.0,
            ranked[-1]["rerank_score"] if ranked else 0.0,
        )
        return ranked
