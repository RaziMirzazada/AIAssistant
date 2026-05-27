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
import re
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

# ---------------------------------------------------------------------------
# V4 Adaptive Depth — three system prompts, picked by question complexity
# ---------------------------------------------------------------------------

CONCISE_SYSTEM_PROMPT = (
    "You are AI Assistant V4, a precise research assistant.\n"
    "The question is SIMPLE (a definition, a factual lookup, a single-concept "
    "query). Give a focused, direct answer — do NOT pad with unnecessary "
    "structural sections.\n"
    "\n"
    "STRICT RULES:\n"
    "1. Ground every claim in the supplied context. Cite [Source #N] inline.\n"
    "2. If the context is insufficient, say so plainly in 1 sentence.\n"
    "3. Aim for 150–400 words. Be precise, not exhaustive.\n"
    "4. Use simple Markdown — short paragraphs, occasional bullet list, bold "
    "for key terms. NO H2 headers like 'Executive Summary' or 'Knowledge Gaps' "
    "— those are reserved for deep analysis. Just answer the question.\n"
    "5. If you genuinely cannot answer from the context, say so in one line "
    "and stop."
)

MODERATE_SYSTEM_PROMPT = (
    "You are AI Assistant V4, a thorough research assistant.\n"
    "The question needs a structured but focused answer. Provide substantive "
    "depth without exhaustively covering every angle.\n"
    "\n"
    "STRICT RULES:\n"
    "1. Ground every claim in the supplied context. Cite [Source #N] inline.\n"
    "2. If the context is insufficient, say so explicitly.\n"
    "3. Aim for 500–900 words. Substantive but not exhaustive.\n"
    "4. Use this Markdown structure with exactly these H2 headers:\n"
    "## Summary\n"
    "A 3–5 sentence digest of the answer with citations.\n"
    "## Key Findings\n"
    "3–6 bulleted findings, each with citations.\n"
    "## Analysis\n"
    "2–3 substantive paragraphs (or a few H3 subsections) that explain the "
    "'why' and connect the findings, with citations.\n"
    "5. Add a '## Caveats' section ONLY if there are genuine caveats worth "
    "raising. Otherwise omit it — do not pad."
)

DEEP_SYSTEM_PROMPT = (
    "You are AI Assistant V4, a senior research analyst.\n"
    "The question requires DEEP, COMPREHENSIVE, MULTI-ANGLE analysis. The "
    "question is complex (comparison, evaluation, multi-angle research) and "
    "deserves exhaustive treatment.\n"
    "\n"
    "STRICT RULES:\n"
    "1. Ground every factual claim in the supplied context snippets and cite "
    "them inline as [Source #N] (1-based). Multiple citations per claim are "
    "encouraged: [Source #2, #7].\n"
    "2. If the context is insufficient for a part of the question, say so "
    "explicitly in the 'Knowledge Gaps' section — do not hallucinate.\n"
    "3. Be exhaustive. Aim for 1500–4000 words of substantive analysis.\n"
    "4. Always use Markdown: H2/H3 headers, bullet lists, numbered lists, "
    "bold for key terms, tables for comparisons.\n"
    "5. Show the reasoning, not just the conclusion. Compare sources where "
    "they agree or disagree.\n"
    "\n"
    "REQUIRED STRUCTURE (use these exact H2 headers):\n"
    "## Executive Summary\n"
    "A 4–8 sentence digest of the answer with the most important citations.\n"
    "## Key Findings\n"
    "5–10 bulleted findings, each with citations.\n"
    "## Detailed Analysis\n"
    "Multiple H3 subsections (one per theme from the outline). Each subsection "
    "is several paragraphs of in-depth analysis with citations.\n"
    "## Counter-arguments & Caveats\n"
    "Where the sources disagree, where evidence is weak, where alternative "
    "interpretations are possible. Cite both sides.\n"
    "## Knowledge Gaps\n"
    "What the supplied sources do NOT cover, and what would be needed to "
    "fully answer the question.\n"
    "## Conclusion\n"
    "A definitive synthesis (3–6 sentences) tying everything together."
)

PLANNER_SYSTEM_PROMPT = (
    "You are a research planner. Given a user question and retrieved context, "
    "(a) classify the question's complexity, (b) classify its type, and "
    "(c) design an outline of analytical themes for the deep-answer path.\n"
    "\n"
    "Return ONLY valid JSON in this exact shape (no markdown fences, no prose):\n"
    "{\n"
    '  "question_type": "definition" | "factual" | "comparison" | "analysis" '
    '| "research" | "how_to" | "opinion",\n'
    '  "complexity":   "simple" | "moderate" | "deep",\n'
    '  "themes": [\n'
    '    {"title": "<H3 subsection title>", "focus": "<1-sentence guidance on '
    'what to cover and which Source #N rows are most relevant>"},\n'
    "    ...\n"
    "  ]\n"
    "}\n"
    "\n"
    "COMPLEXITY RULES (be honest — do not inflate):\n"
    "* simple   — definitions, single-fact lookups, 'what is X', 'when did Y "
    "happen'. Use 0 themes (themes array empty).\n"
    "* moderate — explanations needing 2–4 angles, brief comparisons, focused "
    "how-to. Use 3–4 themes.\n"
    "* deep     — multi-dimensional comparison, evaluative research, complex "
    "'why' / 'should we' / 'compare X across Y' questions, multi-source "
    "synthesis. Use 5–7 themes.\n"
    "\n"
    "When uncertain, choose 'moderate'. Do NOT default to 'deep' just "
    "because the corpus is large. Match depth to question, not to corpus size."
)


def build_context_block(snippets: list[dict]) -> str:
    """Format retrieved snippets into a single context block for the prompt."""
    if not snippets:
        return "No retrieved context was found for this query."
    parts: list[str] = []
    for idx, s in enumerate(snippets, start=1):
        title = s.get("source_title") or "Untitled"
        text = (s.get("text") or "").strip()
        parts.append(f"[Source #{idx} | {title}]\n{text}")
    return "\n\n".join(parts)


def _format_outline(themes: list[dict[str, str]]) -> str:
    """Render the planner's outline back into the expansion prompt."""
    if not themes:
        return ""
    lines = ["OUTLINE TO FOLLOW (use these H3 subsections inside the 'Detailed Analysis' H2):"]
    for i, t in enumerate(themes, 1):
        lines.append(f"  {i}. {t.get('title', '').strip()}")
        focus = (t.get("focus") or "").strip()
        if focus:
            lines.append(f"     → {focus}")
    return "\n".join(lines)


def build_messages(
    memory: ConversationMemory,
    user_prompt: str,
    context_block: str,
    outline: list[dict[str, str]] | None = None,
    system_prompt: str = DEEP_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    """Assemble the OpenAI-style chat message list for the expansion pass.

    ``system_prompt`` should be one of CONCISE/MODERATE/DEEP — picked by
    :func:`_system_prompt_for_depth`. ``outline`` only meaningfully applies
    to the DEEP prompt (the others ignore it and answer directly).
    """
    system_parts = [system_prompt]
    outline_block = _format_outline(outline or [])
    if outline_block:
        system_parts.append(outline_block)
    system_parts.append(f"---\nRetrieved context (cite as [Source #N]):\n{context_block}\n---")
    system = "\n\n".join(system_parts)

    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    messages.extend(memory.as_messages())
    messages.append({"role": "user", "content": user_prompt})
    return messages


def build_planner_messages(
    user_prompt: str, context_block: str
) -> list[dict[str, str]]:
    """Assemble messages for the lightweight planning pass."""
    system = (
        f"{PLANNER_SYSTEM_PROMPT}\n\n"
        f"---\nRetrieved context (Source #N rows):\n{context_block}\n---"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt},
    ]


# ---------------------------------------------------------------------------
# Provider streamers
# ---------------------------------------------------------------------------

class _StreamError(RuntimeError):
    pass


async def _stream_ollama(
    messages: list[dict[str, str]],
    max_tokens: int | None = None,
) -> AsyncIterator[str]:
    url = f"{settings.OLLAMA_HOST.rstrip('/')}/api/chat"
    payload = {
        "model": settings.OLLAMA_LLM_MODEL,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": settings.TEMPERATURE,
            "num_ctx": settings.OLLAMA_NUM_CTX,
            "num_predict": max_tokens or settings.MAX_OUTPUT_TOKENS,
            "top_p": 0.9,
            "repeat_penalty": 1.05,
        },
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


async def _stream_gemini(
    messages: list[dict[str, str]],
    max_tokens: int | None = None,
) -> AsyncIterator[str]:
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
    out_tokens = max_tokens or settings.MAX_OUTPUT_TOKENS
    # Scale thinking budget to depth: small budget for simple, full for deep.
    thinking = min(settings.GEMINI_THINKING_BUDGET, max(256, out_tokens // 4))
    payload: dict = {
        "contents": contents,
        "generationConfig": {
            "temperature": settings.TEMPERATURE,
            "maxOutputTokens": out_tokens,
            "topP": 0.95,
            # Gemini 2.5 extended reasoning. The thinking budget burns reasoning
            # tokens before the visible answer, producing markedly deeper output.
            "thinkingConfig": {"thinkingBudget": thinking},
        },
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


async def _stream_grok(
    messages: list[dict[str, str]],
    max_tokens: int | None = None,
) -> AsyncIterator[str]:
    url = f"{settings.GROK_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.GROK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.GROK_LLM_MODEL,
        "messages": messages,
        "stream": True,
        "temperature": settings.TEMPERATURE,
        "max_tokens": max_tokens or settings.MAX_OUTPUT_TOKENS,
        "top_p": 0.95,
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
# Non-streaming helpers
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


async def _ollama_complete(messages: list[dict[str, str]]) -> str:
    """Non-streaming Ollama chat call — used for the planning pass."""
    url = f"{settings.OLLAMA_HOST.rstrip('/')}/api/chat"
    payload = {
        "model": settings.OLLAMA_LLM_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_ctx": settings.OLLAMA_NUM_CTX,
            "num_predict": 2048,
        },
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            r = await client.post(url, json=payload)
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise _StreamError(f"Ollama plan call failed: {exc}") from exc
    data = r.json()
    return ((data.get("message") or {}).get("content") or "").strip()


async def _grok_complete(messages: list[dict[str, str]]) -> str:
    """Non-streaming Grok call — used for the planning pass."""
    url = f"{settings.GROK_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.GROK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.GROK_LLM_MODEL,
        "messages": messages,
        "stream": False,
        "temperature": 0.2,
        "max_tokens": 2048,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise _StreamError(f"Grok plan call failed: {exc}") from exc
    data = r.json()
    choices = data.get("choices") or []
    if not choices:
        return ""
    return ((choices[0].get("message") or {}).get("content") or "").strip()


async def _gemini_complete_chat(messages: list[dict[str, str]]) -> str:
    """Non-streaming Gemini call from chat-style messages — for planning."""
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
        f"{settings.GEMINI_LLM_MODEL}:generateContent"
        f"?key={settings.GEMINI_API_KEY}"
    )
    payload: dict = {
        "contents": contents,
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2048},
    }
    if system_text:
        payload["systemInstruction"] = {"parts": [{"text": system_text}]}
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            r = await client.post(url, json=payload)
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise _StreamError(f"Gemini plan call failed: {exc}") from exc
    data = r.json()
    candidates = data.get("candidates") or []
    if not candidates:
        return ""
    parts = (candidates[0].get("content") or {}).get("parts") or []
    return "".join(p.get("text", "") for p in parts).strip()


async def _plan_complete(
    mode: "LLMMode",
    provider: "CloudProvider",
    messages: list[dict[str, str]],
) -> str:
    """Dispatch the planning call to the same family of model as the answer."""
    if mode is LLMMode.OFFLINE:
        return await _ollama_complete(messages)
    if provider is CloudProvider.GROK:
        return await _grok_complete(messages)
    return await _gemini_complete_chat(messages)


_VALID_COMPLEXITIES: tuple[str, ...] = ("simple", "moderate", "deep")
_VALID_QUESTION_TYPES: tuple[str, ...] = (
    "definition", "factual", "comparison", "analysis",
    "research", "how_to", "opinion",
)


@dataclass
class Plan:
    """Output of the planner pass.

    ``complexity`` and ``question_type`` always have valid values (falling
    back to ``settings.DEFAULT_DEPTH`` and ``"analysis"`` respectively on
    parse failure). ``themes`` may be empty (legitimately, for simple
    questions, or as a fallback if the JSON failed to parse).
    """

    complexity: str
    question_type: str
    themes: list[dict[str, str]]


def _parse_plan(raw: str) -> Plan:
    """Extract complexity + question_type + themes from the planner response.

    Defensive: planners sometimes wrap JSON in ```json fences or add prose.
    """
    fallback = Plan(
        complexity=settings.DEFAULT_DEPTH,
        question_type="analysis",
        themes=[],
    )
    if not raw:
        return fallback

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        return fallback
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return fallback
    if not isinstance(data, dict):
        return fallback

    complexity = str(data.get("complexity", "")).strip().lower()
    if complexity not in _VALID_COMPLEXITIES:
        complexity = settings.DEFAULT_DEPTH

    qtype = str(data.get("question_type", "")).strip().lower()
    if qtype not in _VALID_QUESTION_TYPES:
        qtype = "analysis"

    themes_raw = data.get("themes")
    themes: list[dict[str, str]] = []
    if isinstance(themes_raw, list):
        for item in themes_raw:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            focus = str(item.get("focus", "")).strip()
            if title:
                themes.append({"title": title, "focus": focus})
        if len(themes) > settings.OUTLINE_MAX_SECTIONS:
            themes = themes[: settings.OUTLINE_MAX_SECTIONS]

    return Plan(complexity=complexity, question_type=qtype, themes=themes)


def _resolve_depth(plan: Plan, override: str | None) -> tuple[str, str]:
    """Decide the final depth and where it came from.

    Returns (depth, source) where source is one of "manual" or "auto".
    """
    if override and override in _VALID_COMPLEXITIES:
        return override, "manual"
    return plan.complexity, "auto"


def _system_prompt_for_depth(depth: str) -> str:
    return {
        "simple":   CONCISE_SYSTEM_PROMPT,
        "moderate": MODERATE_SYSTEM_PROMPT,
        "deep":     DEEP_SYSTEM_PROMPT,
    }[depth]


def _budget_for_depth(depth: str) -> tuple[int, int]:
    """Return (max_output_tokens, snippet_cap) for this depth.

    snippet_cap re-tightens the LLM context: even though Stage-2 returns 15
    chunks, simple questions only need a handful of them. Saves tokens and
    keeps attention focused.
    """
    return {
        "simple":   (settings.DEPTH_SIMPLE_MAX_TOKENS,   settings.DEPTH_SIMPLE_TOP_N),
        "moderate": (settings.DEPTH_MODERATE_MAX_TOKENS, settings.DEPTH_MODERATE_TOP_N),
        "deep":     (settings.DEPTH_DEEP_MAX_TOKENS,     settings.DEPTH_DEEP_TOP_N),
    }[depth]


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
    # V4 adaptive depth — None means "let the planner decide".
    depth_override: str | None = None


@dataclass
class StreamEvent:
    """Internal event yielded by :func:`stream_response`.

    Two kinds are emitted:
    * ``kind="meta"`` once at the start with diagnostic metadata
      (mode actually run, provider, depth, outline themes).
    * ``kind="token"`` for every streamed token of the final answer.
    """

    kind: str
    text: str = ""
    meta: dict | None = None


async def stream_response(req: StreamRequest) -> AsyncIterator[StreamEvent]:
    """Yield events for the given request.

    V4 adaptive-depth pipeline:

        1. PLAN    — non-streamed call that classifies complexity + question
                     type and (for deep questions) emits a 5–7 theme outline.
        2. RESOLVE — combine planner output with optional user override to
                     pick a final depth (simple | moderate | deep).
        3. EXPAND  — streamed call using the depth-appropriate system prompt,
                     top-N snippet cap, and max_tokens budget.

    Translation back to Azerbaijani is the caller's responsibility — this
    function deliberately operates on canonical English so retrieval, memory
    and the LLM all run in their highest-quality language.
    """
    memory = memory_store.get(req.session_id)

    # ---- Secret mode setup --------------------------------------------------
    full_context_block = build_context_block(req.snippets)
    if req.mode is LLMMode.SECRET:
        masked_prompt, prompt_map = crypto.pseudonymize(
            req.user_prompt_english, extra_keywords=req.secret_keywords
        )
        masked_context, ctx_map = crypto.pseudonymize(
            full_context_block, extra_keywords=req.secret_keywords
        )
        combined_reverse = {**prompt_map.reverse, **ctx_map.reverse}
        prompt_for_llm = masked_prompt
        context_for_llm = masked_context
        try:
            token = crypto.encrypt_payload(masked_prompt + masked_context)
            _ = crypto.decrypt_payload(token)
        except Exception:  # noqa: BLE001
            logger.exception("Secret-mode Fernet integrity check failed.")
            raise
    else:
        combined_reverse = {}
        prompt_for_llm = req.user_prompt_english
        context_for_llm = full_context_block

    # ---- Pass 1: PLAN -------------------------------------------------------
    plan = Plan(
        complexity=settings.DEFAULT_DEPTH,
        question_type="analysis",
        themes=[],
    )
    if settings.ENABLE_TWO_PASS:
        plan_msgs = build_planner_messages(prompt_for_llm, context_for_llm)
        try:
            raw_plan = await _plan_complete(req.mode, req.provider, plan_msgs)
            plan = _parse_plan(raw_plan)
            logger.info(
                "Plan: complexity=%s qtype=%s themes=%d (mode=%s provider=%s).",
                plan.complexity, plan.question_type, len(plan.themes),
                req.mode.value, req.provider.value,
            )
        except _StreamError as exc:
            logger.warning("Plan pass failed, using defaults: %s", exc)

    # ---- Resolve final depth (planner vs user override) ---------------------
    final_depth, depth_source = _resolve_depth(plan, req.depth_override)
    max_tokens, snippet_cap = _budget_for_depth(final_depth)

    # For simple/moderate depths we tighten the snippet pile — fewer chunks
    # in the LLM context means crisper, less padded answers.
    snippets_for_prompt = req.snippets[:snippet_cap]
    capped_context = build_context_block(snippets_for_prompt)
    if req.mode is LLMMode.SECRET and snippet_cap < len(req.snippets):
        # Re-mask only what we'll actually ship.
        capped_context, ctx_map2 = crypto.pseudonymize(
            capped_context, extra_keywords=req.secret_keywords
        )
        combined_reverse = {**combined_reverse, **ctx_map2.reverse}
        context_for_llm = capped_context
    else:
        context_for_llm = capped_context if req.mode is not LLMMode.SECRET else context_for_llm

    # Outline only matters for deep — concise/moderate ignore it.
    outline_for_prompt = plan.themes if final_depth == "deep" else []
    system_prompt = _system_prompt_for_depth(final_depth)

    # ---- Emit meta so the UI can show what actually ran --------------------
    reranked = any(
        isinstance(s, dict) and s.get("rerank_score") is not None
        for s in req.snippets
    )
    yield StreamEvent(
        kind="meta",
        meta={
            "mode": req.mode.value,
            "provider": req.provider.value if req.mode is not LLMMode.OFFLINE else "ollama",
            "model": (
                settings.OLLAMA_LLM_MODEL
                if req.mode is LLMMode.OFFLINE
                else (
                    settings.GROK_LLM_MODEL
                    if req.provider is CloudProvider.GROK
                    else settings.GEMINI_LLM_MODEL
                )
            ),
            "depth_chosen": final_depth,
            "depth_source": depth_source,            # "auto" | "manual"
            "question_type": plan.question_type,
            "two_pass": settings.ENABLE_TWO_PASS and bool(outline_for_prompt),
            "top_k": len(snippets_for_prompt),
            "outline": [t["title"] for t in outline_for_prompt],
            "max_output_tokens": max_tokens,
            "reranked": reranked,
            "reranker_model": settings.RERANKER_MODEL if reranked else None,
        },
    )

    # ---- Pass 2: EXPAND (streamed) -----------------------------------------
    messages = build_messages(
        memory,
        prompt_for_llm,
        context_for_llm,
        outline=outline_for_prompt,
        system_prompt=system_prompt,
    )

    if req.mode is LLMMode.OFFLINE:
        provider_stream = _stream_ollama(messages, max_tokens=max_tokens)
    else:
        provider_stream = _provider_stream(
            req.provider, messages, max_tokens=max_tokens
        )

    accumulated = ""
    async for chunk in provider_stream:
        if combined_reverse:
            chunk = _rehydrate_chunk(chunk, combined_reverse)
        accumulated += chunk
        yield StreamEvent(kind="token", text=chunk)

    memory.append("user", req.user_prompt_english)
    memory.append("assistant", accumulated)


def _provider_stream(
    provider: CloudProvider,
    messages: list[dict[str, str]],
    max_tokens: int | None = None,
) -> AsyncIterator[str]:
    if provider is CloudProvider.GEMINI:
        return _stream_gemini(messages, max_tokens=max_tokens)
    if provider is CloudProvider.GROK:
        return _stream_grok(messages, max_tokens=max_tokens)
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
