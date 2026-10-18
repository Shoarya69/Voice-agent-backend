"""
Outbound call pre-warm store for the Gemini Live engine.

Prepares the agent bundle, opens/authenticates the Gemini Live WebSocket, and
pre-renders the opening greeting audio BEFORE the customer picks up so the
first greeting audio frame plays within ~50–300ms of WebSocket connect.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

try:
    import audioop
except ImportError:  # Python 3.13+
    audioop = None  # type: ignore[assignment]

from app import config, lovable_client
from app.audio_utils import chunk_pcm, pcm_bytes_to_ulaw, scale_pcm_volume
from app.supabase_agent_provider import normalize_language
from app.voice_pipeline.gemini_live.link import GeminiLiveLink
from app.voice_pipeline.tools import gemini_live_system_instructions

logger = logging.getLogger(__name__)

_SENTENCE_END = re.compile(r"(?<=[.!?।])\s+")


def _opening_for_call(text: str) -> str:
    parts = [p.strip() for p in _SENTENCE_END.split(text or "") if p.strip()]
    if not parts:
        return (text or "").strip()
    kept = [parts[0]]
    if len(parts) > 1 and len(parts[0].split()) <= 3:
        kept.append(parts[1])
    opening = " ".join(kept)
    words = opening.split()
    if len(words) > 18:
        opening = " ".join(words[:18]).strip()
    return opening


def build_welcome_text(agent_config: lovable_client.AgentConfig | None) -> str:
    text = ""
    if agent_config and (agent_config.first_message or "").strip():
        text = agent_config.first_message.strip()
    if not text:
        text = "Namaste, main aapki madad ke liye hoon."
    if config.VOBIZ_GREETING_FIRST_SENTENCE:
        text = _opening_for_call(text)
    words = text.split()
    if len(words) > config.MAX_WELCOME_WORDS:
        text = " ".join(words[: config.MAX_WELCOME_WORDS]).strip()
    return text


def build_gemini_instructions_for_agent(
    agent_config: lovable_client.AgentConfig,
) -> str:
    system_prompt = (agent_config.system_prompt or "").strip() or config.SYSTEM_PROMPT
    full_prompt = (
        f"{system_prompt}\n\n"
        f"{config.MANDATORY_PLATFORM_RULES_HEADER}\n"
        f"{config.REPLY_LENGTH_INSTRUCTION}\n\n"
        f"{config.CALL_BEHAVIOR_INSTRUCTION}\n\n"
        f"{config.CALL_END_INSTRUCTION}"
    )
    return gemini_live_system_instructions(full_prompt)


def _merge_inline_agent_overrides(
    base: lovable_client.AgentConfig,
    body: dict[str, Any],
) -> lovable_client.AgentConfig:
    inline = body.get("agent")
    if not isinstance(inline, dict):
        inline = {}
    customer_name = (body.get("customer_name") or "").strip()

    system_prompt = (base.system_prompt or inline.get("system_prompt") or "").strip()
    if customer_name and customer_name.lower() not in system_prompt.lower():
        system_prompt = (
            f"{system_prompt}\n\nThe person you are calling is named {customer_name}."
        ).strip()

    first_message = (base.first_message or inline.get("first_message") or "").strip()
    if customer_name and first_message:
        first_message = re.sub(
            r"\[lead name\]", customer_name, first_message, flags=re.IGNORECASE
        )
        first_message = re.sub(
            r"\{\{\s*name\s*\}\}", customer_name, first_message, flags=re.IGNORECASE
        )
        first_message = re.sub(
            r"\{name\}", customer_name, first_message, flags=re.IGNORECASE
        )

    voice_id = (base.voice_id or inline.get("voice") or "").strip()
    language = normalize_language(base.language or inline.get("language"))

    return lovable_client.AgentConfig(
        agent_id=base.agent_id or str(body.get("agent_id") or ""),
        token=base.token or str(body.get("voicebot_token") or body.get("token") or ""),
        user_id=base.user_id,
        system_prompt=system_prompt,
        voice_id=voice_id,
        first_message=first_message,
        language=language,
        temperature=base.temperature,
        max_tokens=base.max_tokens,
        speed=base.speed,
        tone=base.tone,
        greeting_audio_pcm=base.greeting_audio_pcm,
        channel_reservation_id=base.channel_reservation_id,
        phone_number_id=base.phone_number_id,
    )


@dataclass
class PreparedGeminiSession:
    call_id: str
    agent_id: str
    voicebot_token: str
    direction: str = "outbound"
    created_at: float = field(default_factory=time.monotonic)
    expires_at: float = 0.0
    agent_config: lovable_client.AgentConfig | None = None
    welcome_text: str = ""
    link: GeminiLiveLink | None = None

    # Pre-rendered greeting audio buffers (8 kHz PCM frames + 8 kHz μ-law bytes)
    ulaw_greeting: bytearray = field(default_factory=bytearray)
    pcm_8k_frames: list[bytes] = field(default_factory=list)
    out_state: Any = None
    out_residual: bytearray = field(default_factory=bytearray)
    transcript_parts: list[str] = field(default_factory=list)

    greeting_started: bool = False
    greeting_done: bool = False
    claimed: bool = False
    closed: bool = False

    ready_event: asyncio.Event = field(default_factory=asyncio.Event)
    first_audio_event: asyncio.Event = field(default_factory=asyncio.Event)
    warm_task: asyncio.Task | None = None
    expiry_task: asyncio.Task | None = None

    # Stage timing metrics (ms from prewarm start)
    bundle_ms: float | None = None
    gemini_setup_ms: float | None = None
    greeting_first_byte_ms: float | None = None
    greeting_total_ms: float | None = None

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.expiry_task is not None and not self.expiry_task.done():
            self.expiry_task.cancel()
        if self.warm_task is not None and not self.warm_task.done():
            self.warm_task.cancel()
        if self.link is not None and not self.claimed:
            try:
                await self.link.close()
            except Exception:
                pass
            self.link = None


_by_call_id: dict[str, PreparedGeminiSession] = {}
_by_token: dict[str, str] = {}  # voicebot_token -> call_id
_store_lock = asyncio.Lock()


async def _evict_expired_locked(now: float) -> list[PreparedGeminiSession]:
    expired_ids = [
        cid for cid, entry in _by_call_id.items() if now >= entry.expires_at
    ]
    removed: list[PreparedGeminiSession] = []
    for cid in expired_ids:
        entry = _by_call_id.pop(cid, None)
        if entry is not None:
            if _by_token.get(entry.voicebot_token) == cid:
                _by_token.pop(entry.voicebot_token, None)
            removed.append(entry)
    return removed


async def _warm_prepared_session(
    entry: PreparedGeminiSession,
    body: dict[str, Any],
) -> None:
    started = entry.created_at
    try:
        if entry.agent_config is None:
            resolved = await lovable_client.resolve_agent_bundle(
                token=entry.voicebot_token
            )
            entry.agent_config = _merge_inline_agent_overrides(resolved, body)
        entry.bundle_ms = round((time.monotonic() - started) * 1000, 1)

        if entry.closed or entry.claimed:
            return

        entry.welcome_text = build_welcome_text(entry.agent_config)
        instructions = build_gemini_instructions_for_agent(entry.agent_config)

        cached_pcm = entry.agent_config.greeting_audio_pcm
        if cached_pcm:
            entry.pcm_8k_frames = chunk_pcm(cached_pcm)
            entry.ulaw_greeting.extend(
                pcm_bytes_to_ulaw(scale_pcm_volume(cached_pcm, config.TTS_VOLUME))
            )
            entry.greeting_started = True
            entry.greeting_done = True
            entry.greeting_first_byte_ms = round((time.monotonic() - started) * 1000, 1)
            entry.greeting_total_ms = entry.greeting_first_byte_ms
            entry.first_audio_event.set()

        async def _on_audio(pcm_24k: bytes) -> None:
            if entry.closed or entry.claimed or not pcm_24k:
                return
            if entry.greeting_first_byte_ms is None:
                entry.greeting_first_byte_ms = round(
                    (time.monotonic() - started) * 1000, 1
                )
                entry.first_audio_event.set()
            pcm_8k, entry.out_state = audioop.ratecv(
                pcm_24k, 2, 1, 24000, 8000, entry.out_state
            )
            entry.out_residual.extend(pcm_8k)
            frame = config.BYTES_PER_CHUNK
            while len(entry.out_residual) >= frame:
                chunk = bytes(entry.out_residual[:frame])
                del entry.out_residual[:frame]
                entry.pcm_8k_frames.append(chunk)
                entry.ulaw_greeting.extend(
                    pcm_bytes_to_ulaw(scale_pcm_volume(chunk, config.TTS_VOLUME))
                )

        async def _on_transcript(role: str, text: str) -> None:
            if role == "assistant" and text:
                entry.transcript_parts.append(text)

        async def _on_turn_complete() -> None:
            if entry.out_residual and not entry.claimed:
                chunk = bytes(entry.out_residual)
                entry.out_residual.clear()
                entry.pcm_8k_frames.append(chunk)
                entry.ulaw_greeting.extend(
                    pcm_bytes_to_ulaw(scale_pcm_volume(chunk, config.TTS_VOLUME))
                )
            entry.greeting_done = True
            entry.greeting_total_ms = round((time.monotonic() - started) * 1000, 1)
            entry.first_audio_event.set()
            logger.info(
                "PREWARM event=greeting_ready call_id=%s bundle_ms=%s "
                "gemini_setup_ms=%s greeting_first_byte_ms=%s greeting_total_ms=%s "
                "ulaw_bytes=%s",
                entry.call_id,
                entry.bundle_ms,
                entry.gemini_setup_ms,
                entry.greeting_first_byte_ms,
                entry.greeting_total_ms,
                len(entry.ulaw_greeting),
            )

        link = GeminiLiveLink(
            f"prewarm_{entry.call_id[:12]}",
            on_audio=_on_audio,
            on_transcript=_on_transcript,
            on_turn_complete=_on_turn_complete,
        )
        entry.link = link
        await link.connect(instructions)
        entry.gemini_setup_ms = round((time.monotonic() - started) * 1000, 1)
        entry.ready_event.set()

        if entry.closed or entry.claimed:
            return

        if cached_pcm:
            await link.inject_assistant_greeting_context(entry.welcome_text)
        elif entry.welcome_text.strip():
            entry.greeting_started = True
            await link.say(
                "The phone call just connected. Speak ONLY this opening greeting "
                f"and do NOT call any tools: {entry.welcome_text}"
            )

        logger.info(
            "PREWARM event=session_connected call_id=%s bundle_ms=%s gemini_setup_ms=%s",
            entry.call_id,
            entry.bundle_ms,
            entry.gemini_setup_ms,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        entry.ready_event.set()
        entry.first_audio_event.set()
        logger.warning(
            "PREWARM event=warm_failed call_id=%s error=%s",
            entry.call_id,
            type(exc).__name__,
        )


async def _expire_after_ttl(entry: PreparedGeminiSession, ttl_s: float) -> None:
    try:
        await asyncio.sleep(ttl_s)
    except asyncio.CancelledError:
        return

    should_close = False
    async with _store_lock:
        current = _by_call_id.get(entry.call_id)
        if current is entry and not entry.claimed:
            _by_call_id.pop(entry.call_id, None)
            if _by_token.get(entry.voicebot_token) == entry.call_id:
                _by_token.pop(entry.voicebot_token, None)
            should_close = True

    if should_close:
        await entry.close()
        logger.info(
            "PREWARM event=expired call_id=%s ttl_s=%.0f",
            entry.call_id,
            ttl_s,
        )


async def validate_and_schedule_gemini_prewarm(
    body: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    """
    Validate ``voicebot_token`` and schedule a Gemini Live prepared session.

    Returns ``(http_status, json_body)``.
    """
    call_id = str(body.get("call_id") or "").strip()
    voicebot_token = str(
        body.get("voicebot_token") or body.get("token") or ""
    ).strip()
    agent_id = str(body.get("agent_id") or "").strip()
    direction = str(body.get("direction") or "outbound").strip().lower()

    if not voicebot_token or len(voicebot_token) < 16:
        return 401, {"ok": False, "error": "invalid_token"}

    if not call_id:
        call_id = f"tok_{voicebot_token[:16]}"

    now = time.monotonic()
    ttl_s = float(max(30, config.PREWARM_TTL_SECONDS))

    # Idempotency check first (before any network call)
    to_close: list[PreparedGeminiSession] = []
    async with _store_lock:
        to_close = await _evict_expired_locked(now)
        existing = _by_call_id.get(call_id)
        if existing is not None and not existing.closed:
            logger.info("PREWARM event=duplicate_hit call_id=%s", call_id)
            for item in to_close:
                asyncio.create_task(item.close())
            return 200, {"ok": True}
        if len(_by_call_id) >= config.PREWARM_MAX_SESSIONS:
            for item in to_close:
                asyncio.create_task(item.close())
            logger.warning(
                "PREWARM event=rejected_capacity call_id=%s active=%s max=%s",
                call_id,
                len(_by_call_id),
                config.PREWARM_MAX_SESSIONS,
            )
            return 429, {"ok": False, "error": "too_many_prewarm_sessions"}

    for item in to_close:
        asyncio.create_task(item.close())

    # Validate voicebot_token using the same path as /agent-by-token / WSS auth.
    validate_started = time.monotonic()
    resolved_config: lovable_client.AgentConfig | None = None
    try:
        resolved_config = await asyncio.wait_for(
            lovable_client.resolve_agent_bundle(token=voicebot_token),
            timeout=1.15,
        )
    except lovable_client.LovableClientError:
        logger.warning(
            "PREWARM event=auth_rejected call_id=%s validate_ms=%.0f",
            call_id,
            (time.monotonic() - validate_started) * 1000,
        )
        return 401, {"ok": False, "error": "invalid_token"}
    except asyncio.TimeoutError:
        # Provider lookup exceeded 1.15s budget; only allow if inline agent has prompt
        # and background task will still verify token via resolve_agent_bundle.
        inline_agent = body.get("agent")
        if not isinstance(inline_agent, dict) or not inline_agent.get("system_prompt"):
            return 401, {"ok": False, "error": "token_validation_timeout"}

    merged_config: lovable_client.AgentConfig | None = None
    if resolved_config is not None:
        merged_config = _merge_inline_agent_overrides(resolved_config, body)

    async with _store_lock:
        existing = _by_call_id.get(call_id)
        if existing is not None and not existing.closed:
            return 200, {"ok": True}
        if len(_by_call_id) >= config.PREWARM_MAX_SESSIONS:
            return 429, {"ok": False, "error": "too_many_prewarm_sessions"}

        entry = PreparedGeminiSession(
            call_id=call_id,
            agent_id=agent_id or (merged_config.agent_id if merged_config else ""),
            voicebot_token=voicebot_token,
            direction=direction,
            created_at=validate_started,
            expires_at=time.monotonic() + ttl_s,
            agent_config=merged_config,
            bundle_ms=(
                round((time.monotonic() - validate_started) * 1000, 1)
                if merged_config is not None
                else None
            ),
        )
        _by_call_id[call_id] = entry
        _by_token[voicebot_token] = call_id
        entry.warm_task = asyncio.create_task(_warm_prepared_session(entry, body))
        entry.expiry_task = asyncio.create_task(_expire_after_ttl(entry, ttl_s))

    logger.info(
        "PREWARM event=queued call_id=%s validate_ms=%.0f active_prewarms=%s",
        call_id,
        (time.monotonic() - validate_started) * 1000,
        len(_by_call_id),
    )
    return 200, {"ok": True}


async def claim_prepared_gemini_session(
    *,
    call_id: str | None = None,
    token: str | None = None,
) -> PreparedGeminiSession | None:
    """
    Look up and atomically claim a prepared Gemini Live session by ``call_id``
    (or fallback ``voicebot_token`` if ``call_id`` was omitted from the WS URL).
    """
    now = time.monotonic()
    to_close: list[PreparedGeminiSession] = []
    entry: PreparedGeminiSession | None = None

    async with _store_lock:
        to_close = await _evict_expired_locked(now)
        lookup_id = (call_id or "").strip()
        if lookup_id and lookup_id in _by_call_id:
            entry = _by_call_id.pop(lookup_id, None)
        elif token and token.strip() in _by_token:
            mapped_id = _by_token.pop(token.strip(), None)
            if mapped_id:
                entry = _by_call_id.pop(mapped_id, None)

        if entry is not None:
            entry.claimed = True
            if _by_token.get(entry.voicebot_token) == entry.call_id:
                _by_token.pop(entry.voicebot_token, None)
            if entry.expiry_task is not None and not entry.expiry_task.done():
                entry.expiry_task.cancel()

    for item in to_close:
        asyncio.create_task(item.close())

    if entry is not None:
        logger.info(
            "PREWARM event=claimed call_id=%s age_ms=%.0f greeting_done=%s ulaw_bytes=%s",
            entry.call_id,
            (now - entry.created_at) * 1000,
            entry.greeting_done,
            len(entry.ulaw_greeting),
        )
    return entry


async def remove_prepared_gemini_session(
    *,
    call_id: str | None = None,
    token: str | None = None,
) -> None:
    """Delete and close any unclaimed prepared session for this call/token."""
    entry: PreparedGeminiSession | None = None
    async with _store_lock:
        cid = (call_id or "").strip()
        if cid and cid in _by_call_id:
            entry = _by_call_id.pop(cid, None)
        elif token and token.strip() in _by_token:
            mapped = _by_token.pop(token.strip(), None)
            if mapped:
                entry = _by_call_id.pop(mapped, None)
        if entry is not None and _by_token.get(entry.voicebot_token) == entry.call_id:
            _by_token.pop(entry.voicebot_token, None)

    if entry is not None:
        await entry.close()
