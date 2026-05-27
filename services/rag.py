"""
High-Scale Hybrid Search Engine.

Designed for 200+ ingested books on a single RunPod instance.

Key properties
--------------

* **Disk-persisted vectors.** ChromaDB ``PersistentClient`` keeps every
  embedding on disk under ``settings.CHROMA_PATH``. Full books are NEVER
  loaded into active RAM; we stream chunks through the pipeline.
* **Boot-time BM25 rebuild.** On application boot ``HybridSearchEngine``
  reads :pymod:`sources_metadata.json` and pulls the persisted chunk text
  out of ChromaDB to rehydrate the in-memory ``BM25Okapi`` index, so the
  keyword search survives restarts without re-ingestion.
* **Local embeddings.** We call the local Ollama ``nomic-embed-text``
  model so no document text ever leaves the box during indexing.
* **Hybrid scoring.** Cosine similarity (Chroma) is weighted 65% and BM25
  is weighted 35%; the two result sets are merged and re-ranked.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import chromadb
import httpx
from chromadb.api.types import EmbeddingFunction, Embeddings
from chromadb.config import Settings as ChromaSettings
from rank_bm25 import BM25Okapi

from config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding function — calls local Ollama
# ---------------------------------------------------------------------------

class OllamaEmbeddingFunction(EmbeddingFunction):
    """Synchronous Chroma-compatible embedding function backed by Ollama."""

    def __init__(self, host: str, model: str) -> None:
        self._url = f"{host.rstrip('/')}/api/embeddings"
        self._model = model
        self._client = httpx.Client(timeout=60.0)

    def __call__(self, input: list[str]) -> Embeddings:  # noqa: A002 — Chroma signature
        vectors: list[list[float]] = []
        for text in input:
            try:
                response = self._client.post(
                    self._url,
                    json={"model": self._model, "prompt": text},
                )
                response.raise_for_status()
                payload = response.json()
                embedding = payload.get("embedding")
                if not embedding:
                    raise ValueError(f"Ollama returned empty embedding: {payload}")
                vectors.append(embedding)
            except httpx.HTTPError as exc:
                logger.exception("Ollama embedding failure")
                raise RuntimeError(f"Ollama embedding request failed: {exc}") from exc
        return vectors

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        """Async helper used by ingestion paths."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            results: list[list[float]] = []
            for text in texts:
                response = await client.post(
                    self._url,
                    json={"model": self._model, "prompt": text},
                )
                response.raise_for_status()
                embedding = response.json().get("embedding")
                if not embedding:
                    raise RuntimeError("Ollama returned empty embedding.")
                results.append(embedding)
            return results


# ---------------------------------------------------------------------------
# Recursive character text splitter (LangChain-style, no external dep)
# ---------------------------------------------------------------------------

@dataclass
class SplitterConfig:
    chunk_size: int
    chunk_overlap: int
    separators: tuple[str, ...] = ("\n\n", "\n", " ", "")


class RecursiveCharacterSplitter:
    """Splits text on a cascade of separators until chunks fit the target size."""

    def __init__(self, config: SplitterConfig) -> None:
        if config.chunk_overlap >= config.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self._cfg = config

    def split(self, text: str) -> list[str]:
        cleaned = re.sub(r"[ \t]+", " ", text).strip()
        if not cleaned:
            return []
        chunks = self._split_recursive(cleaned, list(self._cfg.separators))
        return self._merge_with_overlap(chunks)

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self._cfg.chunk_size:
            return [text]
        if not separators:
            return [text[i : i + self._cfg.chunk_size]
                    for i in range(0, len(text), self._cfg.chunk_size)]

        separator = separators[0]
        rest = separators[1:]
        if separator == "":
            return [text[i : i + self._cfg.chunk_size]
                    for i in range(0, len(text), self._cfg.chunk_size)]

        pieces = text.split(separator)
        out: list[str] = []
        for piece in pieces:
            if len(piece) <= self._cfg.chunk_size:
                out.append(piece)
            else:
                out.extend(self._split_recursive(piece, rest))
        return [p for p in out if p.strip()]

    def _merge_with_overlap(self, pieces: Sequence[str]) -> list[str]:
        merged: list[str] = []
        buffer = ""
        for piece in pieces:
            candidate = (buffer + " " + piece).strip() if buffer else piece
            if len(candidate) <= self._cfg.chunk_size:
                buffer = candidate
                continue
            if buffer:
                merged.append(buffer)
                tail = buffer[-self._cfg.chunk_overlap :] if self._cfg.chunk_overlap else ""
                buffer = (tail + " " + piece).strip()
            else:
                merged.append(piece[: self._cfg.chunk_size])
                buffer = piece[self._cfg.chunk_size - self._cfg.chunk_overlap :]
        if buffer:
            merged.append(buffer)
        return merged


# ---------------------------------------------------------------------------
# Search result container
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    chunk_id: str
    source_id: str
    source_title: str
    text: str
    score: float
    vector_score: float
    bm25_score: float
    rerank_score: float | None = None  # V4 Stage 2 — set by services.reranker

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "source_id": self.source_id,
            "source_title": self.source_title,
            "text": self.text,
            "score": round(self.score, 4),
            "vector_score": round(self.vector_score, 4),
            "bm25_score": round(self.bm25_score, 4),
        }
        if self.rerank_score is not None:
            out["rerank_score"] = round(self.rerank_score, 4)
        return out


# ---------------------------------------------------------------------------
# Hybrid Search Engine
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9əçşöüğıƏÇŞÖÜĞİ]+")


def _tokenise(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


class HybridSearchEngine:
    """Hybrid (Chroma + BM25) retrieval engine with disk persistence."""

    def __init__(self) -> None:
        Path(settings.CHROMA_PATH).mkdir(parents=True, exist_ok=True)
        self._embed_fn = OllamaEmbeddingFunction(
            host=settings.OLLAMA_HOST, model=settings.OLLAMA_EMBED_MODEL
        )
        self._client = chromadb.PersistentClient(
            path=settings.CHROMA_PATH,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        self._splitter = RecursiveCharacterSplitter(
            SplitterConfig(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
                separators=("\n\n", "\n", " ", ""),
            )
        )

        # BM25 state — guarded by a lock for concurrent ingestion / search.
        self._bm25_lock = threading.RLock()
        self._bm25: BM25Okapi | None = None
        self._bm25_chunk_ids: list[str] = []
        self._bm25_tokens: list[list[str]] = []

        # Source registry persisted to disk.
        self._sources_path = Path(settings.SOURCES_METADATA_PATH)
        self._sources_lock = threading.RLock()
        self._sources: dict[str, dict[str, Any]] = self._load_sources_metadata()

        self._rebuild_bm25_from_disk()

    # ------------------------------------------------------------------ sources
    def _load_sources_metadata(self) -> dict[str, dict[str, Any]]:
        if not self._sources_path.exists():
            return {}
        try:
            with self._sources_path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            logger.exception("Failed to load sources_metadata.json — starting empty.")
            return {}

    def _persist_sources_metadata(self) -> None:
        tmp = self._sources_path.with_suffix(".json.tmp")
        with self._sources_lock:
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(self._sources, fh, ensure_ascii=False, indent=2)
            tmp.replace(self._sources_path)

    def list_sources(self) -> list[dict[str, Any]]:
        with self._sources_lock:
            return sorted(
                (dict(meta) for meta in self._sources.values()),
                key=lambda m: m.get("created_at", ""),
                reverse=True,
            )

    def get_source(self, source_id: str) -> dict[str, Any] | None:
        with self._sources_lock:
            meta = self._sources.get(source_id)
            return dict(meta) if meta else None

    # ----------------------------------------------------------------- BM25
    def _rebuild_bm25_from_disk(self) -> None:
        """Pull every chunk out of Chroma and rebuild the BM25 index in RAM."""
        with self._bm25_lock:
            self._bm25_chunk_ids = []
            self._bm25_tokens = []

            try:
                payload = self._collection.get(include=["documents"])
            except Exception:  # noqa: BLE001 — Chroma raises a broad exception family
                logger.exception("Failed to read from ChromaDB during BM25 rebuild.")
                self._bm25 = None
                return

            documents: list[str] = payload.get("documents") or []
            ids: list[str] = payload.get("ids") or []
            if not documents:
                self._bm25 = None
                logger.info("BM25 rebuild skipped — no chunks in ChromaDB yet.")
                return

            for chunk_id, doc in zip(ids, documents):
                self._bm25_chunk_ids.append(chunk_id)
                self._bm25_tokens.append(_tokenise(doc))

            self._bm25 = BM25Okapi(self._bm25_tokens)
            logger.info("BM25 rebuilt from disk: %d chunks indexed.", len(self._bm25_chunk_ids))

    def _add_to_bm25(self, chunk_ids: Sequence[str], texts: Sequence[str]) -> None:
        with self._bm25_lock:
            for cid, text in zip(chunk_ids, texts):
                self._bm25_chunk_ids.append(cid)
                self._bm25_tokens.append(_tokenise(text))
            if self._bm25_tokens:
                self._bm25 = BM25Okapi(self._bm25_tokens)

    def _remove_from_bm25(self, chunk_ids: Iterable[str]) -> None:
        drop = set(chunk_ids)
        with self._bm25_lock:
            kept_ids: list[str] = []
            kept_tokens: list[list[str]] = []
            for cid, toks in zip(self._bm25_chunk_ids, self._bm25_tokens):
                if cid in drop:
                    continue
                kept_ids.append(cid)
                kept_tokens.append(toks)
            self._bm25_chunk_ids = kept_ids
            self._bm25_tokens = kept_tokens
            self._bm25 = BM25Okapi(self._bm25_tokens) if self._bm25_tokens else None

    # ----------------------------------------------------------------- ingestion
    async def add_source(
        self,
        *,
        title: str,
        text: str,
        source_type: str,
        origin: str | None = None,
    ) -> dict[str, Any]:
        """Chunk, embed and persist a new source. Returns its metadata."""
        if not text or not text.strip():
            raise ValueError("Cannot ingest an empty source.")

        chunks = self._splitter.split(text)
        if not chunks:
            raise ValueError("Splitting produced zero chunks.")

        source_id = uuid.uuid4().hex
        chunk_ids = [f"{source_id}:{i:06d}" for i in range(len(chunks))]
        metadatas = [
            {
                "source_id": source_id,
                "source_title": title,
                "source_type": source_type,
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]

        # Off-load the blocking ChromaDB write to a worker thread so the event
        # loop stays responsive while embeddings are computed by Ollama.
        await asyncio.to_thread(
            self._collection.add,
            ids=chunk_ids,
            documents=chunks,
            metadatas=metadatas,
        )
        await asyncio.to_thread(self._add_to_bm25, chunk_ids, chunks)

        from datetime import datetime, timezone

        meta = {
            "id": source_id,
            "title": title,
            "type": source_type,
            "origin": origin,
            "chunk_count": len(chunks),
            "char_count": len(text),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with self._sources_lock:
            self._sources[source_id] = meta
        await asyncio.to_thread(self._persist_sources_metadata)
        logger.info("Indexed source %s (%s chunks).", source_id, len(chunks))
        return dict(meta)

    async def delete_source(self, source_id: str) -> bool:
        with self._sources_lock:
            if source_id not in self._sources:
                return False
            del self._sources[source_id]

        # Locate Chroma chunks for this source.
        payload = await asyncio.to_thread(
            self._collection.get,
            where={"source_id": source_id},
            include=[],
        )
        chunk_ids: list[str] = payload.get("ids") or []
        if chunk_ids:
            await asyncio.to_thread(self._collection.delete, ids=chunk_ids)
            await asyncio.to_thread(self._remove_from_bm25, chunk_ids)
        await asyncio.to_thread(self._persist_sources_metadata)
        return True

    # ----------------------------------------------------------------- search
    async def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        if not query or not query.strip():
            return []
        k = top_k or settings.RAG_TOP_K
        candidates = max(k * 3, settings.RAG_CANDIDATES_PER_INDEX)

        vector_task = asyncio.to_thread(self._vector_search, query, candidates)
        bm25_task = asyncio.to_thread(self._bm25_search, query, candidates)
        vector_hits, bm25_hits = await asyncio.gather(vector_task, bm25_task)

        return self._merge(vector_hits, bm25_hits, k=k)

    def _vector_search(self, query: str, n: int) -> dict[str, dict[str, Any]]:
        try:
            response = self._collection.query(
                query_texts=[query],
                n_results=n,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:  # noqa: BLE001 — Chroma raises various exception types
            logger.exception("Vector search failed.")
            return {}

        results: dict[str, dict[str, Any]] = {}
        ids = (response.get("ids") or [[]])[0]
        docs = (response.get("documents") or [[]])[0]
        metas = (response.get("metadatas") or [[]])[0]
        dists = (response.get("distances") or [[]])[0]
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            similarity = max(0.0, 1.0 - float(dist))  # cosine distance -> similarity
            results[cid] = {
                "text": doc,
                "metadata": meta or {},
                "vector_score": similarity,
            }
        return results

    def _bm25_search(self, query: str, n: int) -> dict[str, dict[str, Any]]:
        with self._bm25_lock:
            if self._bm25 is None or not self._bm25_chunk_ids:
                return {}
            tokens = _tokenise(query)
            if not tokens:
                return {}
            scores = self._bm25.get_scores(tokens)
            max_score = float(scores.max()) if len(scores) else 0.0
            if max_score <= 0:
                return {}
            ranked = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True,
            )[:n]
            return {
                self._bm25_chunk_ids[i]: {
                    "bm25_score": float(scores[i]) / max_score,
                }
                for i in ranked
                if scores[i] > 0
            }

    def _merge(
        self,
        vector_hits: dict[str, dict[str, Any]],
        bm25_hits: dict[str, dict[str, Any]],
        k: int,
    ) -> list[SearchResult]:
        chunk_ids = set(vector_hits) | set(bm25_hits)
        if not chunk_ids:
            return []

        # Hydrate missing rows from Chroma (a BM25-only hit needs its text/meta).
        missing = [cid for cid in chunk_ids if cid not in vector_hits]
        if missing:
            payload = self._collection.get(
                ids=missing,
                include=["documents", "metadatas"],
            )
            ids = payload.get("ids") or []
            docs = payload.get("documents") or []
            metas = payload.get("metadatas") or []
            for cid, doc, meta in zip(ids, docs, metas):
                vector_hits[cid] = {
                    "text": doc,
                    "metadata": meta or {},
                    "vector_score": 0.0,
                }

        merged: list[SearchResult] = []
        for cid in chunk_ids:
            vinfo = vector_hits.get(cid, {})
            text = vinfo.get("text", "")
            meta = vinfo.get("metadata") or {}
            vscore = float(vinfo.get("vector_score", 0.0))
            bscore = float(bm25_hits.get(cid, {}).get("bm25_score", 0.0))
            combined = (
                settings.HYBRID_VECTOR_WEIGHT * vscore
                + settings.HYBRID_BM25_WEIGHT * bscore
            )
            merged.append(
                SearchResult(
                    chunk_id=cid,
                    source_id=str(meta.get("source_id", "")),
                    source_title=str(meta.get("source_title", "")),
                    text=text or "",
                    score=combined,
                    vector_score=vscore,
                    bm25_score=bscore,
                )
            )

        merged.sort(key=lambda r: r.score, reverse=True)
        return merged[:k]

    # ----------------------------------------------------------------- diagnostics
    def stats(self) -> dict[str, Any]:
        with self._sources_lock:
            source_count = len(self._sources)
        with self._bm25_lock:
            chunk_count = len(self._bm25_chunk_ids)
        return {
            "sources": source_count,
            "chunks": chunk_count,
            "chroma_path": settings.CHROMA_PATH,
        }
