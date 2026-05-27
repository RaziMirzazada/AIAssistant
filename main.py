"""
AI Assistant V3 — FastAPI application.

Hardened entry point:
    * X-API-KEY gatekeeper on every endpoint (no anonymous traffic).
    * Strict CORS allow-list (never ``*``).
    * AZ <-> EN translation bridge so the UI stays in Azerbaijani while the
      retrieval and LLM pipeline runs on English text for maximum accuracy.
    * Streaming chat over HTTP chunked transfer with per-stream backpressure.

Run locally:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
from pathlib import Path
from typing import Any

import httpx
from bs4 import BeautifulSoup
from fastapi import Depends, FastAPI, File, Form, HTTPException, Security, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field, HttpUrl
from pypdf import PdfReader

from config import settings
from services import llm
from services.llm import (
    CloudProvider,
    LLMMode,
    StreamRequest,
    memory_store,
    probe_all,
    stream_response,
    translate_az_to_en,
    translate_en_to_az,
)
from services.rag import HybridSearchEngine
from services.reranker import get_reranker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
)
logger = logging.getLogger("ai-assistant-v3")


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

_api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)


async def require_api_key(api_key: str | None = Security(_api_key_header)) -> str:
    if not api_key or api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-API-KEY başlığı tələb olunur və düzgün olmalıdır.",
        )
    return api_key


# ---------------------------------------------------------------------------
# Engine bootstrap
# ---------------------------------------------------------------------------

engine: HybridSearchEngine | None = None
app = FastAPI(
    title="AI Assistant V3 — Advanced Edition",
    version="3.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-KEY"],
)


@app.on_event("startup")
async def _startup() -> None:
    global engine
    settings.ensure_directories()
    logger.info("Booting HybridSearchEngine — Chroma path: %s", settings.CHROMA_PATH)
    engine = await asyncio.to_thread(HybridSearchEngine)
    logger.info("Engine ready: %s", engine.stats())

    # Preload the cross-encoder reranker so the first chat call is fast.
    # ~600 MB download on first boot, then cached at ~/.cache/huggingface.
    if settings.ENABLE_RERANKER and settings.RERANKER_WARMUP_ON_STARTUP:
        logger.info("Warming up reranker '%s' …", settings.RERANKER_MODEL)
        try:
            await asyncio.to_thread(get_reranker().warmup)
            logger.info("Reranker warmup complete.")
        except Exception:  # noqa: BLE001 — warmup failure is non-fatal
            logger.exception("Reranker warmup failed (will retry on first query).")


def get_engine() -> HybridSearchEngine:
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG mühərriki hələ hazır deyil.",
        )
    return engine


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    message: str = Field(..., min_length=1)
    mode: LLMMode = LLMMode.OFFLINE
    provider: CloudProvider = CloudProvider.GEMINI
    secret_keywords: list[str] = Field(default_factory=list)


class UrlIngestRequest(BaseModel):
    url: HttpUrl
    title: str | None = None


class TextIngestRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    text: str = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_pdf_text(payload: bytes) -> str:
    reader = PdfReader(io.BytesIO(payload))
    parts: list[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:  # noqa: BLE001 — pypdf can raise diverse errors per page
            text = ""
        if text:
            parts.append(text)
    return "\n\n".join(parts)


async def _fetch_url_text(url: str) -> str:
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        response = await client.get(url, headers={"User-Agent": "AI-Assistant-V3/1.0"})
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "pdf" in content_type.lower() or url.lower().endswith(".pdf"):
            return await asyncio.to_thread(_extract_pdf_text, response.content)
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    """Serve the Azerbaijani single-file UI."""
    path = Path(settings.FRONTEND_PATH)
    if not path.exists():
        # Fallback: try the repo-local copy (dev mode).
        local = Path(__file__).parent / "frontend.html"
        if local.exists():
            return FileResponse(local, media_type="text/html")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"frontend.html tapılmadı: {settings.FRONTEND_PATH}",
        )
    return FileResponse(path, media_type="text/html")


@app.get("/api/health")
async def health(_: str = Depends(require_api_key)) -> dict[str, Any]:
    probes = await probe_all()
    eng = get_engine()
    return {
        "status": "ok",
        "engine": eng.stats(),
        "services": probes,
        "workspace": settings.WORKSPACE_ROOT,
    }


@app.get("/api/sources")
async def list_sources(_: str = Depends(require_api_key)) -> list[dict[str, Any]]:
    return get_engine().list_sources()


@app.delete("/api/sources/{source_id}")
async def delete_source(
    source_id: str, _: str = Depends(require_api_key)
) -> dict[str, Any]:
    deleted = await get_engine().delete_source(source_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Mənbə tapılmadı.")
    return {"status": "deleted", "id": source_id}


@app.post("/api/sources/upload")
async def upload_source(
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    _: str = Depends(require_api_key),
) -> dict[str, Any]:
    eng = get_engine()
    filename = file.filename or "naməlum.pdf"
    final_title = (title or filename).strip()
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Boş fayl yüklənə bilməz.")

    try:
        if filename.lower().endswith(".pdf"):
            text = await asyncio.to_thread(_extract_pdf_text, payload)
        else:
            text = payload.decode("utf-8", errors="ignore")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Fayl emal edilə bilmədi: {exc}") from exc

    if not text.strip():
        raise HTTPException(status_code=400, detail="Fayldan mətn çıxarıla bilmədi.")

    meta = await eng.add_source(
        title=final_title,
        text=text,
        source_type="pdf" if filename.lower().endswith(".pdf") else "file",
        origin=filename,
    )
    return meta


@app.post("/api/sources/url")
async def ingest_url(
    body: UrlIngestRequest, _: str = Depends(require_api_key)
) -> dict[str, Any]:
    eng = get_engine()
    try:
        text = await _fetch_url_text(str(body.url))
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=400, detail=f"URL açıla bilmədi: {exc}") from exc
    if not text.strip():
        raise HTTPException(status_code=400, detail="URL-dən mətn çıxarıla bilmədi.")
    title = (body.title or str(body.url)).strip()
    return await eng.add_source(title=title, text=text, source_type="url", origin=str(body.url))


@app.post("/api/sources/text")
async def ingest_text(
    body: TextIngestRequest, _: str = Depends(require_api_key)
) -> dict[str, Any]:
    eng = get_engine()
    return await eng.add_source(
        title=body.title.strip(),
        text=body.text,
        source_type="text",
        origin="manual",
    )


# ---------------------------------------------------------------------------
# Chat (streaming, NDJSON event protocol)
# ---------------------------------------------------------------------------

@app.post("/api/chat")
async def chat(req: ChatRequest, _: str = Depends(require_api_key)) -> StreamingResponse:
    """Stream a chat response as NDJSON events.

    Event shapes:
        {"type": "translation",  "text": "<english query>"}
        {"type": "sources",      "items": [...]}
        {"type": "run_meta",     "mode": "...", "provider": "...", "model": "...",
                                 "depth": "maximum", "two_pass": true, "top_k": N,
                                 "outline": ["...", ...], "max_output_tokens": N}
        {"type": "token_en",     "text": "..."}      # streamed English tokens
        {"type": "answer_az",    "text": "..."}      # final Azerbaijani answer
        {"type": "error",        "message": "..."}
        {"type": "done"}
    """
    eng = get_engine()

    logger.info(
        "chat() session=%s mode=%s provider=%s msg_chars=%d",
        req.session_id, req.mode.value, req.provider.value, len(req.message),
    )

    async def _producer():
        try:
            # 1) AZ -> EN
            english_query = await translate_az_to_en(req.message)
            yield _ndjson({"type": "translation", "text": english_query})

            # 2) Retrieve in English (Stage 1 — hybrid retrieval, recall-oriented)
            retrieve_k = (
                settings.RERANKER_RETRIEVE_K
                if settings.ENABLE_RERANKER
                else settings.RAG_TOP_K
            )
            results = await eng.search(english_query, top_k=retrieve_k)

            # 2b) Rerank (Stage 2 — cross-encoder, precision-oriented).
            # Fail-safe: any error inside .rerank() returns the hybrid order.
            reranked_flag = False
            if settings.ENABLE_RERANKER and results:
                results = await get_reranker().rerank(english_query, results)
                # If at least one result actually got a rerank_score we mark
                # this run as truly reranked. Otherwise the reranker fell back.
                reranked_flag = any(r.rerank_score is not None for r in results)

            snippets = [r.to_dict() for r in results]
            yield _ndjson({
                "type": "sources",
                "items": snippets,
                "reranked": reranked_flag,
            })

            # 3) LLM stream (English) — emits a meta event then tokens
            stream_req = StreamRequest(
                session_id=req.session_id,
                mode=req.mode,
                provider=req.provider,
                user_prompt_english=english_query,
                snippets=snippets,
                secret_keywords=req.secret_keywords,
            )
            english_answer = ""
            async for event in stream_response(stream_req):
                if event.kind == "meta":
                    payload = {"type": "run_meta"}
                    payload.update(event.meta or {})
                    logger.info(
                        "run_meta: %s",
                        {k: v for k, v in (event.meta or {}).items() if k != "outline"},
                    )
                    yield _ndjson(payload)
                elif event.kind == "token":
                    english_answer += event.text
                    yield _ndjson({"type": "token_en", "text": event.text})

            # 4) EN -> AZ (full final answer)
            az_answer = await translate_en_to_az(english_answer)
            yield _ndjson({"type": "answer_az", "text": az_answer})
            yield _ndjson({"type": "done"})
        except asyncio.CancelledError:
            logger.info("Chat stream cancelled by client.")
            raise
        except Exception as exc:  # noqa: BLE001 — surface clean error to client
            logger.exception("Chat stream failed.")
            yield _ndjson({"type": "error", "message": str(exc)})

    return StreamingResponse(
        _producer(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


def _ndjson(obj: dict[str, Any]) -> bytes:
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


# ---------------------------------------------------------------------------
# Session memory controls
# ---------------------------------------------------------------------------

@app.post("/api/session/{session_id}/reset")
async def reset_session(
    session_id: str, _: str = Depends(require_api_key)
) -> dict[str, str]:
    memory_store.reset(session_id)
    return {"status": "reset", "session_id": session_id}


# ---------------------------------------------------------------------------
# Default error envelope
# ---------------------------------------------------------------------------

@app.exception_handler(HTTPException)
async def http_exception_handler(_, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status": exc.status_code},
    )
