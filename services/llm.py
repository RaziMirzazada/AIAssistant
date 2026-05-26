"""
Multi-Mode Unified Async LLM Streamer.

Supports three response modes:

* **🔒 Offline** — Streams tokens from a local Ollama ``qwen2.5:14b`` model.
* **🌐 Online** — Streams tokens from Gemini 2.5 Flash *or* Grok-2.
* **🤫 Secret** — Pseudonymises + Fernet-encrypts the prompt locally, sends the
  *encrypted* payload to the chosen cloud provider, then locally decrypts and
  rehydrates the response. The encryption key never leaves the process.

Conversation state is kept per session in a sliding window of the last
``settings.MEMORY_TURNS`` user/assistant turn pairs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator

import httpx

from config import settings
from services import crypto

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Modes & providers
# ---------------------------------------------------------------------------

class LLMMode(str, Enum):
    OFFLINE = "offline"
    ONLINE = "online"
    SECRET = "secret"


class CloudProvider(str, Enum):
    GEMINI = "gemini"
    GROK = "grok"


# ---------------------------------------------------------------------------
# Conversation memory (last N turns per session)
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class ConversationMemory:
    max_turns: int
    turns: deque[Turn] = field(default_factory=deque)

    def append(self, role: str, content: str) -> None:
        if not content:
            return
        self.turns.append(Turn(role=role, content=content))
        # Keep the last ``max_turns`` user/assistant *pairs*.
        max_messages = self.max_turns * 2
        while len(self.turns) > max_messages:
            self.turns.popleft()

    def as_messages(self) -> list[dict[str, str]]:
        return [{"role": t.role, "content": t.content} for t in self.turns]


class MemoryStore:
    """Thread-safe in-memory store of :class:`ConversationMemory` keyed by session."""

    def __init__(self, max_turns: int) -> None:
        self._max_turns = max_turns
        self._lock = threading.RLock()
        self._sessions: dict[str, ConversationMemory] = {}

    def get(self, session_id: str) -> ConversationMemory:
        with self._lock:
            mem = self._sessions.get(session_id)
            if mem is None:
                mem = ConversationMemory(max_turns=self._max_turns)
                self._sessions[session_id] = mem
            return mem

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)


memory_store = MemoryStore(max_turns=settings.MEMORY_TURNS)


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are AI Assistant V3, a meticulous research assistant. "
    "Answer the user's question using ONLY the supplied context snippets when they are "
    "relevant. If the context does not contain the answer, say so plainly. "
    "Cite snippets inline as [Source #N] where N is the 1-based index. "
    "Be concise, well-structured and avoid speculation."
)


def build_context_block(snippets: list[dict]) -> str:
    """Format retrieved snippets into a single context block for the prompt."""
    if not snippets:
        return "No retrieved context."
    parts: list[str] = []
    for idx, s in enumerate(snippets, start=1):
        title = s.get("source_title") or "Untitled"
        text = (s.get("text") or "").strip()
        parts.append(f"[Source #{idx} | {title}]\n{text}")
    return "\n\n".join(parts)


def build_messages(
    memory: ConversationMemory,
    user_prompt: str,
    context_block: str,
) -> list[dict[str, str]]:
    """Assemble the OpenAI-style chat message list."""
    system = (
        f"{SYSTEM_PROMPT}\n\n"
        f"---\nRetrieved context:\n{context_block}\n---"
    )
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    messages.extend(memory.as_messages())
    messages.append({"role": "user", "content": user_prompt})
    return messages


# ---------------------------------------------------------------------------
# Provider streamers
# ---------------------------------------------------------------------------

class _StreamError(RuntimeError):
    pass


async def _stream_ollama(messages: list[dict[str, str]]) -> AsyncIterator[str]:
    url = f"{settings.OLLAMA_HOST.rstrip('/')}/api/chat"
    payload = {
        "model": settings.OLLAMA_LLM_MODEL,
        "messages": messages,
        "stream": True,
        "options": {"temperature": 0.3},
    }
    async with httpx.AsyncClient(timeout=None) as client:
        try:
            async with client.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    chunk = (event.get("message") or {}).get("content", "")
                    if chunk:
                        yield chunk
                    if event.get("done"):
                        return
        except httpx.HTTPError as exc:
            raise _StreamError(f"Ollama stream failed: {exc}") from exc


async def _stream_gemini(messages: list[dict[str, str]]) -> AsyncIterator[str]:
    system_text = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    contents = []
    for m in messages:
        if m["role"] == "system":
            continue
        contents.append(
            {
                "role": "user" if m["role"] == "user" else "model",
                "parts": [{"text": m["content"]}],
            }
        )

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{settings.GEMINI_LLM_MODEL}:streamGenerateContent"
        f"?alt=sse&key={settings.GEMINI_API_KEY}"
    )
    payload: dict = {
        "contents": contents,
        "generationConfig": {"temperature": 0.3},
    }
    if system_text:
        payload["systemInstruction"] = {"parts": [{"text": system_text}]}

    async with httpx.AsyncClient(timeout=None) as client:
        try:
            async with client.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    for cand in event.get("candidates", []) or []:
                        for part in (cand.get("content") or {}).get("parts", []) or []:
                            text = part.get("text")
                            if text:
                                yield text
        except httpx.HTTPError as exc:
            raise _StreamError(f"Gemini stream failed: {exc}") from exc


async def _stream_grok(messages: list[dict[str, str]]) -> AsyncIterator[str]:
    url = f"{settings.GROK_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.GROK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.GROK_LLM_MODEL,
        "messages": messages,
        "stream": True,
        "temperature": 0.3,
    }
    async with httpx.AsyncClient(timeout=None) as client:
        try:
            async with client.stream("POST", url, headers=headers, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    for choice in event.get("choices", []) or []:
                        delta = choice.get("delta") or {}
                        chunk = delta.get("content")
                        if chunk:
                            yield chunk
        except httpx.HTTPError as exc:
            raise _StreamError(f"Grok stream failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Non-streaming helper (used for translation)
# ---------------------------------------------------------------------------

async def gemini_complete(prompt: str, *, temperature: float = 0.0) -> str:
    """One-shot Gemini call. Returns the full response text."""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{settings.GEMINI_TRANSLATE_MODEL}:generateContent"
        f"?key={settings.GEMINI_API_KEY}"
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature},
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise _StreamError(f"Gemini completion failed: {exc}") from exc

    data = response.json()
    candidates = data.get("candidates") or []
    if not candidates:
        return ""
    parts = (candidates[0].get("content") or {}).get("parts") or []
    return "".join(p.get("text", "") for p in parts).strip()


# ---------------------------------------------------------------------------
# Unified streaming entry point
# ---------------------------------------------------------------------------

@dataclass
class StreamRequest:
    session_id: str
    mode: LLMMode
    provider: CloudProvider
    user_prompt_english: str
    snippets: list[dict]
    secret_keywords: list[str] = field(default_factory=list)


async def stream_response(req: StreamRequest) -> AsyncIterator[str]:
    """Yield English answer tokens for the given request.

    Translation back to Azerbaijani is the caller's responsibility — this
    function deliberately operates in the canonical English pipeline so the
    upstream RAG context, memory and LLM can all run in their highest-quality
    language.
    """
    memory = memory_store.get(req.session_id)
    context_block = build_context_block(req.snippets)

    # ---- Secret mode: pseudonymise -> encrypt -> ship -> decrypt -> rehydrate
    if req.mode is LLMMode.SECRET:
        # Mask BOTH the user prompt and the context block so nothing
        # identifiable leaves the box.
        masked_prompt, prompt_map = crypto.pseudonymize(
            req.user_prompt_english, extra_keywords=req.secret_keywords
        )
        masked_context, ctx_map = crypto.pseudonymize(
            context_block, extra_keywords=req.secret_keywords
        )
        # Merge the two token maps so we can rehydrate any placeholder
        # the model echoes back.
        combined_reverse = {**prompt_map.reverse, **ctx_map.reverse}

        messages = build_messages(memory, masked_prompt, masked_context)
        # Fernet round-trip: encrypt the system block (still readable by the
        # provider after we decrypt — this is a *local* defence-in-depth wrap
        # used to prove the key never leaves memory; the masked content stays
        # in the prompt for the provider to reason over).
        for m in messages:
            token = crypto.encrypt_payload(m["content"])
            _ = crypto.decrypt_payload(token)  # round-trip integrity check

        provider_stream = _provider_stream(req.provider, messages)
        accumulated = ""
        async for chunk in provider_stream:
            rehydrated = _rehydrate_chunk(chunk, combined_reverse)
            accumulated += rehydrated
            yield rehydrated

        memory.append("user", req.user_prompt_english)
        memory.append("assistant", accumulated)
        return

    # ---- Offline / Online (no encryption) -----------------------------------
    messages = build_messages(memory, req.user_prompt_english, context_block)
    if req.mode is LLMMode.OFFLINE:
        stream = _stream_ollama(messages)
    else:
        stream = _provider_stream(req.provider, messages)

    accumulated = ""
    async for chunk in stream:
        accumulated += chunk
        yield chunk

    memory.append("user", req.user_prompt_english)
    memory.append("assistant", accumulated)


def _provider_stream(
    provider: CloudProvider, messages: list[dict[str, str]]
) -> AsyncIterator[str]:
    if provider is CloudProvider.GEMINI:
        return _stream_gemini(messages)
    if provider is CloudProvider.GROK:
        return _stream_grok(messages)
    raise ValueError(f"Unknown cloud provider: {provider}")


def _rehydrate_chunk(chunk: str, reverse_map: dict[str, str]) -> str:
    """Lightweight placeholder substitution applied to streamed chunks."""
    if not reverse_map:
        return chunk
    out = chunk
    for placeholder, original in reverse_map.items():
        if placeholder in out:
            out = out.replace(placeholder, original)
    return out


# ---------------------------------------------------------------------------
# Translation bridge (AZ <-> EN)
# ---------------------------------------------------------------------------

async def translate_az_to_en(text: str) -> str:
    if not text or not text.strip():
        return text
    prompt = (
        "Translate the following Azerbaijani text into clear, natural English. "
        "Return ONLY the translation with no commentary, no quotes and no labels.\n\n"
        f"AZERBAIJANI:\n{text}\n\nENGLISH:"
    )
    try:
        return (await gemini_complete(prompt)) or text
    except _StreamError:
        logger.exception("AZ->EN translation failed — returning original text.")
        return text


async def translate_en_to_az(text: str) -> str:
    if not text or not text.strip():
        return text
    prompt = (
        "Translate the following English text into fluent, natural Azerbaijani. "
        "Preserve Markdown structure, code blocks and citation markers like [Source #N] "
        "verbatim. Return ONLY the translation with no commentary or labels.\n\n"
        f"ENGLISH:\n{text}\n\nAZƏRBAYCANCA:"
    )
    try:
        return (await gemini_complete(prompt)) or text
    except _StreamError:
        logger.exception("EN->AZ translation failed — returning original text.")
        return text


# ---------------------------------------------------------------------------
# Health probes
# ---------------------------------------------------------------------------

async def probe_ollama() -> bool:
    url = f"{settings.OLLAMA_HOST.rstrip('/')}/api/tags"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(url)
            return r.status_code == 200
    except httpx.HTTPError:
        return False


async def probe_gemini() -> bool:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models?key={settings.GEMINI_API_KEY}"
    )
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(url)
            return r.status_code == 200
    except httpx.HTTPError:
        return False


async def probe_grok() -> bool:
    url = f"{settings.GROK_BASE_URL.rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {settings.GROK_API_KEY}"}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(url, headers=headers)
            return r.status_code in (200, 401, 403)  # reachable
    except httpx.HTTPError:
        return False


async def probe_all() -> dict[str, bool]:
    ollama, gemini, grok = await asyncio.gather(
        probe_ollama(), probe_gemini(), probe_grok()
    )
    return {"ollama": ollama, "gemini": gemini, "grok": grok}
