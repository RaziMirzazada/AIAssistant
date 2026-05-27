"""
Cross-encoder reranker — Stage 2 of the retrieval pipeline.

The hybrid retriever in :pymod:`services.rag` is fast and recall-oriented
(it must consider every chunk in the corpus). Its scores, however, are a
cheap proxy: vector cosine + BM25 never actually let the model see query
and chunk *together*.

A cross-encoder fixes that. It takes a small batch of ``(query, chunk)``
pairs and runs both through a single transformer with full cross-attention,
producing a single relevance logit per pair. We rank by that and keep only
the top N before handing them to the answer-generation LLM. Empirically
this lifts retrieval precision by 5–15 % at the cost of ~200–300 ms per
query on a single mid-range GPU.

Design notes
------------

* **Lazy singleton.** The model is large (~600 MB on disk, ~2.5 GB in
  GPU RAM). We load it once on first use and reuse it for every query.
* **Async-friendly.** The model itself is synchronous and GPU-bound, so
  we run it inside :func:`asyncio.to_thread` to keep the event loop
  responsive while the GPU works.
* **Fail-safe.** A reranker error MUST NEVER break a chat call. Any
  exception falls back to the hybrid order untouched.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
from typing import Sequence

from config import settings
from services.rag import SearchResult

logger = logging.getLogger(__name__)


class Reranker:
    """Lazy-loaded multilingual cross-encoder."""

    def __init__(self, model_name: str, device: str | None = None) -> None:
        self._model_name = model_name
        self._device = device  # None means auto-detect inside _ensure_loaded
        self._model = None  # set on first use
        self._load_lock = threading.Lock()

    # ------------------------------------------------------------------ loading
    def _resolve_device(self) -> str:
        if self._device:
            return self._device
        try:
            import torch  # local import — torch is heavy
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _ensure_loaded(self):
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model
            device = self._resolve_device()
            logger.info(
                "Loading cross-encoder reranker '%s' on %s …",
                self._model_name, device,
            )
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is not installed. "
                    "Run: pip install sentence-transformers"
                ) from exc
            self._model = CrossEncoder(
                self._model_name,
                device=device,
                max_length=512,
            )
            logger.info("Reranker '%s' ready on %s.", self._model_name, device)
            return self._model

    def warmup(self) -> bool:
        """Public wrapper to trigger model load eagerly (e.g. on app startup).

        Returns True on success, False on failure. The reranker remains usable
        either way — a False result just means the next chat call will retry
        the load lazily and pay the load cost on the request hot path.
        """
        try:
            self._ensure_loaded()
        except Exception:  # noqa: BLE001
            logger.exception("Reranker warmup failed — will retry at query time.")
            return False
        return True

    @property
    def is_loaded(self) -> bool:
        """True iff the underlying model is in memory and ready to serve."""
        return self._model is not None

    # ----------------------------------------------------------------- scoring
    @staticmethod
    def _sigmoid(x: float) -> float:
        # Cross-encoder logits typically land in ~[-10, +10]. Sigmoid squashes
        # them to (0, 1) so they're directly comparable with hybrid scores.
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        ex = math.exp(x)
        return ex / (1.0 + ex)

    def _rerank_sync(
        self,
        query: str,
        results: list[SearchResult],
        top_n: int,
    ) -> list[SearchResult]:
        if not results:
            return results
        model = self._ensure_loaded()
        pairs: list[tuple[str, str]] = [(query, r.text) for r in results]
        raw_scores = model.predict(
            pairs,
            batch_size=settings.RERANKER_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        # Apply scoring strategy.
        mode = settings.RERANKER_SCORE_MODE.lower()
        for r, raw in zip(results, raw_scores):
            normalised = self._sigmoid(float(raw))
            r.rerank_score = normalised
            if mode == "blend":
                # Hedged: keep some hybrid signal in case reranker is weird.
                r.score = 0.3 * r.score + 0.7 * normalised
            else:
                # "two_stage" (default) and "replace" both let the reranker
                # decide final order; the hybrid score has already done its
                # job upstream (it picked which candidates reached us).
                r.score = normalised
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_n]

    async def rerank(
        self,
        query: str,
        results: Sequence[SearchResult],
        top_n: int | None = None,
    ) -> list[SearchResult]:
        """Run the reranker in a worker thread.

        On any failure we log and return ``results[:top_n]`` unchanged so the
        chat pipeline can keep going. A reranker failure is never fatal.
        """
        n = top_n or settings.RERANKER_TOP_N
        result_list = list(results)
        try:
            return await asyncio.to_thread(
                self._rerank_sync, query, result_list, n
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "Reranker failed — returning hybrid-ordered top %d.", n
            )
            return result_list[:n]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_singleton: Reranker | None = None
_singleton_lock = threading.Lock()


def get_reranker() -> Reranker:
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = Reranker(
                    model_name=settings.RERANKER_MODEL,
                    device=settings.RERANKER_DEVICE,
                )
    return _singleton
