"""
Citation Verifier — async faithfulness check that runs AFTER the answer
has finished streaming.

Pipeline
--------

1. **Parse claims.** Split the streamed English answer into sentences and
   keep only those that carry one or more ``[Source #N]`` citations. Each
   such sentence is treated as a single atomic claim.

2. **Judge each claim.** For every claim, send the claim text + the
   cited chunks to Gemini 2.5 Flash with a strict JSON-only prompt that
   returns a verdict ("supported" / "partial" / "unsupported"), a
   confidence in [0, 1] and a one-sentence note explaining the call.

3. **Report.** Caller (``main.py``) emits the results as a single
   ``{"type": "verification", ...}`` NDJSON event so the UI can render
   the panel under the answer without blocking the streaming token loop.

The verifier never raises into the caller — any internal failure
produces a "partial" / "unsupported" verdict for that one claim instead
of crashing the chat path.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import asdict, dataclass
from typing import Iterable

from config import settings
from services.llm import gemini_complete

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------

@dataclass
class CitationCheck:
    """One judged claim."""

    claim: str
    cited_sources: list[int]
    verdict: str            # "supported" | "partial" | "unsupported"
    confidence: float       # 0.0 – 1.0
    note: str               # short justification

    def to_dict(self) -> dict:
        d = asdict(self)
        d["confidence"] = round(self.confidence, 3)
        return d


# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------

# Roughly: split on sentence boundaries; preserve the original sentence so
# the judge can read the full claim in context.
_SENTENCE_SPLIT = re.compile(r"(?<=[\.\?!])\s+(?=[A-Z\[])")
_CITATION_PATTERN = re.compile(r"\[Sources?\s*#?(\d+)(?:\s*[,&]?\s*#?(\d+))*\]")


def _strip_markdown(text: str) -> str:
    """Drop headers / bullet prefixes / inline markdown so the judge sees
    clean prose. Keeps the [Source #N] markers intact."""
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[-*]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return text


def extract_claims(answer: str, max_claims: int) -> list[tuple[str, list[int]]]:
    """Return up to ``max_claims`` (claim_text, [cited_source_ns]) tuples.

    A "claim" is any sentence in the answer that contains at least one
    ``[Source #N]`` citation. Returns them in document order.
    """
    if not answer or not answer.strip():
        return []
    cleaned = _strip_markdown(answer)
    sentences = _SENTENCE_SPLIT.split(cleaned)
    out: list[tuple[str, list[int]]] = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        nums: list[int] = []
        for m in _CITATION_PATTERN.finditer(s):
            for grp in m.groups():
                if grp:
                    try:
                        nums.append(int(grp))
                    except ValueError:
                        pass
            # Also catch the rest of the inner numbers (regex only captures 2)
            for n in re.findall(r"\d+", m.group(0)):
                try:
                    n_int = int(n)
                    if n_int not in nums:
                        nums.append(n_int)
                except ValueError:
                    pass
        if nums:
            out.append((s, sorted(set(nums))))
            if len(out) >= max_claims:
                break
    return out


# ---------------------------------------------------------------------------
# Judge prompt + parsing
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = (
    "You are a strict fact-checker. Decide whether a CLAIM is fully "
    "supported by the CITED SOURCES below. Be honest — partial overlap is "
    "NOT 'supported'. If the source mentions the topic but does not "
    "back the specific assertion, that is 'partial'.\n"
    "\n"
    "Output ONLY a single JSON object on one line, no markdown fences, no "
    "prose, no commentary:\n"
    '{"verdict": "supported"|"partial"|"unsupported", '
    '"confidence": <0.0-1.0>, "note": "<one short sentence>"}\n'
    "\n"
    "Verdict meanings:\n"
    "* supported   — every factual element of the claim is explicitly "
    "stated or directly implied by the cited sources.\n"
    "* partial     — the topic is in the sources but some part of the "
    "claim goes beyond what they say.\n"
    "* unsupported — the cited sources do not contain the claim, even "
    "if related material is present."
)


def _build_judge_prompt(claim: str, snippets: dict[int, dict]) -> str:
    parts = ["CLAIM:", claim, "", "CITED SOURCES:"]
    if not snippets:
        parts.append("(no matching sources were retrievable — assume unsupported)")
    for n, s in snippets.items():
        title = s.get("source_title") or "Untitled"
        text = (s.get("text") or "").strip()[: settings.VERIFIER_CHUNK_PREVIEW_CHARS]
        parts.append(f"[Source #{n} | {title}]")
        parts.append(text)
        parts.append("")
    return "\n".join(parts)


def _parse_judge_response(raw: str) -> tuple[str, float, str]:
    """Best-effort parse of the judge's JSON. Falls back safely."""
    if not raw:
        return "partial", 0.0, "no response from judge"
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    m = re.search(r"\{[\s\S]*\}", cleaned)
    if not m:
        return "partial", 0.0, "judge returned non-JSON"
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return "partial", 0.0, "judge JSON parse failed"
    verdict = str(data.get("verdict", "")).strip().lower()
    if verdict not in ("supported", "partial", "unsupported"):
        verdict = "partial"
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    note = str(data.get("note", "")).strip()[:240]
    return verdict, confidence, note


# ---------------------------------------------------------------------------
# Single-claim judge call
# ---------------------------------------------------------------------------

async def _judge_one(
    claim: str,
    cited_ns: list[int],
    snippets_by_n: dict[int, dict],
) -> CitationCheck:
    # Build a small sub-dict of just the cited snippets so the judge
    # gets focused context rather than the full retrieval pile.
    cited_snippets = {
        n: snippets_by_n[n] for n in cited_ns if n in snippets_by_n
    }
    user_msg = _build_judge_prompt(claim, cited_snippets)
    prompt = f"{_JUDGE_SYSTEM}\n\n{user_msg}"
    try:
        raw = await gemini_complete(prompt, temperature=0.0)
    except Exception:  # noqa: BLE001 — never break the chat path
        logger.exception("Judge call failed for claim: %r", claim[:80])
        return CitationCheck(
            claim=claim,
            cited_sources=cited_ns,
            verdict="partial",
            confidence=0.0,
            note="verifier error — could not check",
        )
    verdict, confidence, note = _parse_judge_response(raw)
    if confidence < settings.VERIFIER_MIN_CONFIDENCE and verdict == "supported":
        verdict = "partial"
        note = note or "low judge confidence"
    return CitationCheck(
        claim=claim,
        cited_sources=cited_ns,
        verdict=verdict,
        confidence=confidence,
        note=note,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def verify_answer(
    answer: str,
    snippets: list[dict],
) -> list[CitationCheck]:
    """Verify every cited claim in ``answer`` against the retrieved
    ``snippets`` (in original retrieval order — #N is 1-based).

    Returns an empty list if the answer has no citations.
    Never raises — internal errors are surfaced as per-claim verdicts.
    """
    claims = extract_claims(answer, max_claims=settings.VERIFIER_MAX_CLAIMS)
    if not claims:
        return []
    snippets_by_n: dict[int, dict] = {
        i + 1: s for i, s in enumerate(snippets)
    }
    coros = [_judge_one(c, ns, snippets_by_n) for c, ns in claims]
    # Bounded concurrency — Gemini handles parallel requests but we don't
    # want to swamp the API. 4 in flight is conservative; tune if needed.
    sem = asyncio.Semaphore(4)

    async def _bound(coro):
        async with sem:
            return await coro

    return await asyncio.gather(*[_bound(c) for c in coros])


def summarise(results: Iterable[CitationCheck]) -> dict:
    """Compact summary suitable for the verification NDJSON payload."""
    results = list(results)
    counts = {"supported": 0, "partial": 0, "unsupported": 0}
    for r in results:
        counts[r.verdict] = counts.get(r.verdict, 0) + 1
    return {
        "total": len(results),
        "supported": counts["supported"],
        "partial": counts["partial"],
        "unsupported": counts["unsupported"],
        "results": [r.to_dict() for r in results],
    }
