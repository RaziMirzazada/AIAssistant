"""
Secret Mode encryption pipeline.

Provides a local pseudonymisation + AES-256 (Fernet) layer that runs BEFORE
any payload is forwarded to a cloud LLM and AFTER any cloud response is
received. The Fernet key lives strictly in process memory and is regenerated
on every application boot — it is never persisted to disk.

Workflow:

    1.  ``pseudonymize(text)`` scans the text for sensitive structures
        (emails, phone numbers, credit-card-shaped numbers, IBAN-style codes,
        IPv4 addresses, dates, capitalised name tokens and a small list of
        user-registered keywords) and replaces every match with a stable
        placeholder of the form ``[[TOKEN_<hash>]]``. The mapping is stored
        in a per-request token map.

    2.  ``encrypt_payload(text)`` Fernet-encrypts the pseudonymised text so
        even the placeholder structure is opaque while travelling.

    3.  On the way back, ``decrypt_payload`` reverses Fernet and
        ``rehydrate(text, token_map)`` substitutes the original values
        back in, producing the final plaintext for the user.
"""

from __future__ import annotations

import hashlib
import re
import threading
from dataclasses import dataclass, field
from typing import Final

from cryptography.fernet import Fernet, InvalidToken


# ---------------------------------------------------------------------------
# In-memory key store (never persisted).
# ---------------------------------------------------------------------------

_KEY_LOCK = threading.Lock()
_SESSION_KEY: bytes = Fernet.generate_key()


def rotate_session_key() -> None:
    """Generate a fresh Fernet key. Existing ciphertext becomes unreadable."""
    global _SESSION_KEY
    with _KEY_LOCK:
        _SESSION_KEY = Fernet.generate_key()


def _fernet() -> Fernet:
    with _KEY_LOCK:
        return Fernet(_SESSION_KEY)


# ---------------------------------------------------------------------------
# Sensitive-pattern detection
# ---------------------------------------------------------------------------

_PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "EMAIL": re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    "PHONE": re.compile(r"(?:\+?\d{1,3}[\s\-]?)?(?:\(?\d{2,4}\)?[\s\-]?){2,4}\d{2,4}"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ \-]*?){13,19}\b"),
    "IBAN": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),
    "IPV4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "DATE_ISO": re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    "URL": re.compile(r"https?://[^\s)>\]]+"),
    "NAME": re.compile(r"\b[A-ZƏÇŞÖÜĞİ][a-zəçşöüği]{2,}\s+[A-ZƏÇŞÖÜĞİ][a-zəçşöüği]{2,}\b"),
}


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]


@dataclass
class TokenMap:
    """Bidirectional mapping between placeholders and original values."""

    forward: dict[str, str] = field(default_factory=dict)  # original -> placeholder
    reverse: dict[str, str] = field(default_factory=dict)  # placeholder -> original

    def register(self, original: str, category: str) -> str:
        if original in self.forward:
            return self.forward[original]
        placeholder = f"[[{category}_{_short_hash(original)}]]"
        self.forward[original] = placeholder
        self.reverse[placeholder] = original
        return placeholder


def pseudonymize(
    text: str,
    extra_keywords: list[str] | None = None,
) -> tuple[str, TokenMap]:
    """Replace sensitive substrings with deterministic placeholders.

    Returns the masked text and the :class:`TokenMap` needed to rehydrate it.
    """
    if not text:
        return text, TokenMap()

    token_map = TokenMap()
    masked = text

    # Pattern-driven masking
    for category, pattern in _PATTERNS.items():
        def _replace(match: re.Match[str], _cat: str = category) -> str:
            return token_map.register(match.group(0), _cat)

        masked = pattern.sub(_replace, masked)

    # User-defined sensitive keywords
    if extra_keywords:
        for kw in extra_keywords:
            kw_stripped = kw.strip()
            if not kw_stripped:
                continue
            pattern = re.compile(rf"\b{re.escape(kw_stripped)}\b", flags=re.IGNORECASE)
            masked = pattern.sub(
                lambda m: token_map.register(m.group(0), "KW"),
                masked,
            )

    return masked, token_map


def rehydrate(text: str, token_map: TokenMap) -> str:
    """Replace placeholders in ``text`` with their original values."""
    if not text or not token_map.reverse:
        return text

    restored = text
    # Sort by length descending so longer placeholders do not interfere with
    # shorter overlapping ones.
    for placeholder in sorted(token_map.reverse.keys(), key=len, reverse=True):
        restored = restored.replace(placeholder, token_map.reverse[placeholder])
    return restored


# ---------------------------------------------------------------------------
# Encryption / decryption
# ---------------------------------------------------------------------------

def encrypt_payload(text: str) -> str:
    """Fernet-encrypt ``text`` using the current in-memory session key."""
    if text is None:
        raise ValueError("Cannot encrypt a None payload.")
    token: bytes = _fernet().encrypt(text.encode("utf-8"))
    return token.decode("utf-8")


def decrypt_payload(token: str) -> str:
    """Reverse :func:`encrypt_payload`. Raises :class:`InvalidToken` on failure."""
    try:
        return _fernet().decrypt(token.encode("utf-8")).decode("utf-8")
    except InvalidToken as exc:
        raise InvalidToken("Decryption failed — session key may have rotated.") from exc


# ---------------------------------------------------------------------------
# High-level orchestration helpers
# ---------------------------------------------------------------------------

@dataclass
class SecretBundle:
    """A bundle ready to be transmitted to a cloud LLM."""

    encrypted: str
    masked: str
    token_map: TokenMap


def seal(text: str, extra_keywords: list[str] | None = None) -> SecretBundle:
    """Pseudonymise then encrypt — the full outbound Secret Mode pipeline."""
    masked, token_map = pseudonymize(text, extra_keywords=extra_keywords)
    encrypted = encrypt_payload(masked)
    return SecretBundle(encrypted=encrypted, masked=masked, token_map=token_map)


def unseal(encrypted_response: str, token_map: TokenMap) -> str:
    """Decrypt then rehydrate placeholders — the full inbound pipeline."""
    masked = decrypt_payload(encrypted_response)
    return rehydrate(masked, token_map)
