"""Heuristics when Gemini Live does not emit [[END_CALL]] in audio transcripts."""

from __future__ import annotations

import re

_USER_END_RE = re.compile(
    r"(?i)\b("
    r"bye|goodbye|good bye|disconnect|hang up|hangup|"
    r"call cut|call end|end call|end the call|"
    r"band karo|call band|phone rakh|rakho|"
    r"cut kar do|call kat|bas itna|that's all|thank you bye"
    r")\b"
)

_CLOSING_RE = re.compile(
    r"(?i)\b("
    r"bye|goodbye|alvida|dhanyavad|dhanyavaad|shukriya|thank you|"
    r"namaste|aapka din shubh|call rakh|phone rakh"
    r")\b"
)


def user_requested_hangup(text: str) -> bool:
    return bool(_USER_END_RE.search((text or "").strip()))


def assistant_sounds_like_closing(text: str) -> bool:
    return bool(_CLOSING_RE.search((text or "").strip()))
