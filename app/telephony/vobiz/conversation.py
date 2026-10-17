"""Vobiz-specific STT gating — fewer echo/repeat turns on phone audio."""

from __future__ import annotations

import asyncio
import re

from app import config

# Short backchannels / echo that STT fires while the caller waits for the bot.
_TIME_CHECK = re.compile(r"(?:2|do|two)\s*minute", re.IGNORECASE)

_FILLER_ONLY = re.compile(
    r"^(?:"
    r"hello+|hi+|hey+|hiya|"
    r"yes+|yeah|yep|yup|ok(?:ay)?|"
    r"haan|ji+|namaste|"
    r"uh+|um+|hmm+|mm+hmm"
    r")[\s\?\.!,]*$",
    re.IGNORECASE,
)


def normalize_reply(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def is_filler_transcript(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if len(t.split()) <= 4 and _FILLER_ONLY.match(t):
        return True
    return False


def stt_gate_open(session) -> bool:
    if getattr(session, "telephony_provider", "") != "vobiz":
        return True
    if not config.VOBIZ_STT_GATE:
        return True
    until = getattr(session, "_stt_gate_until", 0.0)
    return asyncio.get_running_loop().time() >= until


def arm_stt_gate_after_bot_spoke(session, *, playback_ms: float = 0) -> None:
    """Block new STT turns while phone echo + caller 'hello?' tail fades."""
    if getattr(session, "telephony_provider", "") != "vobiz":
        return
    base = config.VOBIZ_STT_COOLDOWN_MS / 1000.0
    extra = min(max(playback_ms, 0) / 1000.0 * 0.35, 1.2)
    session._stt_gate_until = asyncio.get_running_loop().time() + base + extra
    session._last_bot_spoke_at = asyncio.get_running_loop().time()


def should_ignore_committed(session, text: str) -> str | None:
    """
    Return a short log reason if this transcript should not start a turn.
    """
    if getattr(session, "telephony_provider", "") != "vobiz":
        return None
    if not stt_gate_open(session):
        return "stt_gate"
    if is_filler_transcript(text):
        last = getattr(session, "_last_bot_spoke_at", 0.0)
        if last and asyncio.get_running_loop().time() - last < 10.0:
            return "filler_after_bot"
    return None


def note_assistant_reply(session, reply: str) -> None:
    if getattr(session, "telephony_provider", "") != "vobiz":
        return
    if _TIME_CHECK.search(reply or ""):
        session._time_check_asked = True


def llm_user_content(session, user_text: str) -> str:
    """Steer the model: natural phone tone, no scripted repeats."""
    if getattr(session, "telephony_provider", "") != "vobiz":
        return user_text
    notes: list[str] = []
    if getattr(session, "_time_check_asked", False):
        notes.append(
            "You already asked if they have 2 minutes — do not ask again; move the conversation forward."
        )
    if is_filler_transcript(user_text):
        notes.append(
            "Caller is only checking the line. One brief different sentence, then continue — "
            "never repeat your last line word-for-word."
        )
    elif len(user_text.split()) <= 6:
        notes.append("Reply in one short spoken sentence, friendly and natural.")
    if not notes:
        return user_text
    return f"{user_text.strip()}\n\n(" + " ".join(notes) + ")"


def is_duplicate_assistant_reply(session, reply: str) -> bool:
    if getattr(session, "telephony_provider", "") != "vobiz":
        return False
    if not config.VOBIZ_SKIP_DUPLICATE_REPLIES:
        return False
    prev = normalize_reply(getattr(session, "_last_spoken_reply", ""))
    cur = normalize_reply(reply)
    if not prev or not cur:
        return False
    return prev == cur
