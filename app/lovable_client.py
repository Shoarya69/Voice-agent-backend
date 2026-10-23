"""
Lovable control plane client.

Responsibilities, all intentionally kept out of the audio/voice pipeline
(app/voice_session.py) entirely:

  1. fetch_agent(token)         - resolve agent config from a per-call token
                                   (WSS URL .../voicebot/<token> or ?token=).
  2. fetch_agent_by_number(n)   - resolve agent config from a caller mobile
                                   number (WSS URL .../voicebot/number/<n>).
                                   Both return the same AgentConfig shape and
                                   are cached in-memory for a few minutes.
  3. post_call_log(...)         - fire-and-forget a call summary/transcript
                                   back to Lovable once a call ends.
  4. post_exotel_status(...)    - relay Exotel's Status Callback fields
                                   (CallStatus/CallDuration/RecordingUrl/...)
                                   to Lovable's voice-status webhook.
  5. post_exotel_conversation(..) - relay Exotel's conversation applet fields
                                   to Lovable's voice-conversation webhook.
  6. resolve_agent_bundle(...)  - singleflight agent + greeting resolution
                                   (dedupes concurrent prewarm + WSS prefetch).
  7. prewarm_agent_bundle(...)  - warm caches during ring / before WSS connect.

All of these use a short-timeout, non-blocking httpx.AsyncClient so a
Lovable outage or slow response can never stall or crash the voicebot.
Nothing here ever sends an `agent_id` - Lovable always resolves the agent
itself, either from the token (agent-by-token) or from the caller's phone
number via `phone_numbers` (voice-inbound, Exotel -> Lovable directly).

Agent-config lookups (read path):
  - Inbound by number: Supabase RPC ``get_inbound_agent_bundle`` (single path)
  - Outbound by token: Supabase RPC ``get_agent_bundle_by_token`` (Moontech HTTP fallback)
  - Greeting PCM: Moontech HTTP ``agent-with-greeting`` when not cached

The `voice-inbound` telephony hook (Exotel -> Lovable, TwiML) is separate
and is not called from this service.
"""

import asyncio
import base64
import binascii
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from urllib.parse import quote

import httpx

from app import config
from app import supabase_agent_provider
from app.audio_utils import pcm_duration_ms

logger = logging.getLogger(__name__)

_AGENT_LOOKUP_PATH = "/api/public/voicebot/agent-by-token"
_AGENT_WITH_GREETING_PATH = "/api/public/voicebot/agent-with-greeting"
_CALL_LOG_PATH = "/api/public/voicebot/call-log"

# Exotel-facing hooks on the Lovable side. These never take an agent_id -
# Lovable resolves the agent from the caller's phone number.
# NOTE: voice-inbound is intentionally NOT called from this service - see
# module docstring. Only status/conversation are relayed here.
_VOICE_STATUS_PATH = "/api/public/hooks/voice-status"
_VOICE_CONVERSATION_PATH = "/api/public/hooks/voice-conversation"

# 2.0s proved too tight in production: the *first* HTTPS request to a
# Cloudflare-fronted app (fresh TLS handshake, cold edge function, etc.) can
# take a bit longer than a warmed-up curl connection, even though the
# endpoint itself responds fine. One retry with a slightly longer timeout
# comfortably covers that without meaningfully slowing down a real outage
# detection (still fails fast, just not on a single borderline-slow request).
_AGENT_FETCH_TIMEOUT = 4.0
_AGENT_FETCH_RETRY_TIMEOUT = 6.0
# call-log waits for Lovable to finish DB insert + Gemini analysis (3-8s typical).
_CALL_LOG_TIMEOUT = 30.0
_EXOTEL_WEBHOOK_TIMEOUT = 5.0

_CACHE_TTL_SECONDS = 5 * 60

_client: httpx.AsyncClient | None = None
_client_lock = asyncio.Lock()

# token -> (fetched_at_monotonic, AgentConfig)
_agent_cache: dict[str, tuple[float, "AgentConfig"]] = {}
_cache_lock = asyncio.Lock()

# mobile number -> (fetched_at_monotonic, AgentConfig)
_agent_by_number_cache: dict[str, tuple[float, "AgentConfig"]] = {}
_number_cache_lock = asyncio.Lock()

# agent_id -> (fetched_at_monotonic, AgentConfig with optional pre-stored greeting PCM)
_agent_with_greeting_cache: dict[str, tuple[float, "AgentConfig"]] = {}
_greeting_cache_lock = asyncio.Lock()

# greeting_audio_hash -> validated 8 kHz 16-bit mono PCM bytes (reused across calls)
_greeting_pcm_by_hash: dict[str, bytes] = {}
# voicebot_token -> greeting_audio_hash
_token_greeting_hash: dict[str, str] = {}

# Durable spool for failed call logs when portal is temporarily unreachable
_FAILED_CALL_LOGS_PATH = os.path.join("logs", "failed_call_logs.jsonl")
_call_log_spool_lock = asyncio.Lock()
_CALL_LOG_MAX_RETRIES = 3
_CALL_LOG_ATTEMPT_TIMEOUT = 10.0

# In-flight bundle resolutions (prewarm + WSS prefetch share one MoontechPro fetch).
_inflight_bundles: dict[str, asyncio.Task] = {}
_inflight_lock = asyncio.Lock()


class LovableClientError(Exception):
    """Raised when an agent-config lookup against Lovable fails."""


@dataclass
class AgentConfig:
    agent_id: str
    token: str
    user_id: str = ""
    system_prompt: str = ""
    voice_id: str = ""
    first_message: str = ""
    language: str = ""
    temperature: float = 0.6
    max_tokens: int = 45
    speed: float = 1.0
    tone: str = ""
    greeting_audio_pcm: bytes | None = None
    greeting_audio_hash: str = ""
    closing_message: str = ""
    gemini_voice: str = ""
    channel_reservation_id: str = ""
    phone_number_id: str = ""

    def gemini_voice_name(self, default_voice: str) -> str:
        """Portal ``engine.voice``, else legacy voice id, else bootstrap default."""
        return (self.gemini_voice or self.voice_id or default_voice or "").strip()

    def as_overrides(self) -> dict:
        """Fields consumed by app.config's per-call override mechanism."""
        return {
            "system_prompt": self.system_prompt,
            "voice_id": self.voice_id,
            "first_message": self.first_message,
            "language": self.language,
            "temperature": self.temperature,
            # max_tokens omitted — platform OPENAI_MAX_TOKENS is not overridable per agent.
        }


def _token_prefix(token: str) -> str:
    return (token or "")[:6]


def _number_suffix(number: str) -> str:
    return f"...{number[-4:]}" if number else ""


def normalize_phone_number(number: str) -> str:
    """
    Canonical cache key for Indian mobiles — MoontechPro matches on last 10 digits.
    Ensures prewarm (Exotel To=07971451588) hits the same cache as WSS
    (.../number/9107971451588).
    """
    digits = re.sub(r"\D", "", number or "")
    if len(digits) >= 10:
        return digits[-10:]
    return digits


def _number_cache_keys(number: str) -> list[str]:
    """Alias keys so lookups succeed regardless of Exotel/WSS formatting."""
    keys: list[str] = []
    normalized = normalize_phone_number(number)
    raw_digits = re.sub(r"\D", "", number or "")
    for candidate in (normalized, raw_digits, number.strip() if number else ""):
        if candidate and candidate not in keys:
            keys.append(candidate)
    return keys or [number]


def _normalize_greeting_pcm(pcm: bytes | None, *, agent_id: str = "") -> bytes | None:
    """
    Accept greeting PCM only if it is valid 8 kHz 16-bit mono PCM and fits limits.
    Strips WAV RIFF header if present and aligns to 2-byte sample boundaries.
    """
    if not pcm:
        return None
    if len(pcm) > 44 and pcm[:4] == b"RIFF" and pcm[8:12] == b"WAVE":
        # Strip standard WAV header up to the 'data' chunk
        data_idx = pcm.find(b"data")
        if 12 <= data_idx <= 128 and data_idx + 8 <= len(pcm):
            pcm = pcm[data_idx + 8 :]
        else:
            pcm = pcm[44:]
    if len(pcm) % 2 != 0:
        pcm = pcm[:-1]
    if len(pcm) < 320:
        return None
    duration_ms = pcm_duration_ms(pcm)
    max_ms = config.MAX_GREETING_AUDIO_MS
    if duration_ms > max_ms:
        logger.warning(
            "Greeting PCM rejected agent_id=%s duration_ms=%.0f max_ms=%s bytes=%s "
            "— will use live TTS greeting instead",
            agent_id[:8] if agent_id else "",
            duration_ms,
            max_ms,
            len(pcm),
        )
        return None
    return pcm


def _decode_greeting_pcm(
    greeting: dict,
    *,
    agent_id: str = "",
    token: str = "",
) -> tuple[bytes | None, str]:
    """
    Decode pre-generated greeting audio (raw PCM 8kHz/16-bit/mono) from base64
    and cache by hash so subsequent calls reuse the decoded PCM bytes.
    """
    if not isinstance(greeting, dict):
        return None, ""

    audio_hash = str(
        greeting.get("hash") or greeting.get("greeting_audio_hash") or ""
    ).strip()
    if audio_hash and audio_hash in _greeting_pcm_by_hash:
        if token:
            _token_greeting_hash[token] = audio_hash
        return _greeting_pcm_by_hash[audio_hash], audio_hash

    audio_b64 = (
        greeting.get("audio_base64")
        or greeting.get("greeting_audio_base64")
        or ""
    )
    if not audio_b64.strip():
        return None, audio_hash
    try:
        raw_pcm = base64.b64decode(audio_b64, validate=True)
    except (ValueError, binascii.Error):
        try:
            raw_pcm = base64.b64decode(audio_b64)
        except Exception:
            logger.warning("Invalid greeting.audio_base64 payload for agent greeting")
            return None, audio_hash

    pcm = _normalize_greeting_pcm(raw_pcm, agent_id=agent_id)
    if pcm is not None:
        if not audio_hash:
            audio_hash = hashlib.sha256(pcm).hexdigest()[:24]
        _greeting_pcm_by_hash[audio_hash] = pcm
        if token:
            _token_greeting_hash[token] = audio_hash
    return pcm, audio_hash


async def get_agent_greeting_pcm(voicebot_token: str) -> bytes | None:
    """
    Retrieve cached or portal-provided 8 kHz PCM greeting for ``voicebot_token``,
    keyed by greeting hash so it is never re-decoded or re-synthesized per call.
    """
    tok = (voicebot_token or "").strip()
    if not tok:
        return None

    known_hash = _token_greeting_hash.get(tok)
    if known_hash and known_hash in _greeting_pcm_by_hash:
        return _greeting_pcm_by_hash[known_hash]

    cached_cfg = _agent_cache.get(tok)
    if cached_cfg is not None and cached_cfg[1].greeting_audio_pcm:
        return cached_cfg[1].greeting_audio_pcm

    greeting_obj = await supabase_agent_provider.fetch_agent_greeting_pcm(tok)
    if greeting_obj:
        pcm, _ = _decode_greeting_pcm(greeting_obj, token=tok)
        if pcm is not None:
            return pcm
    return None


def _parse_agent_config_payload(payload: dict, token: str = "") -> AgentConfig:
    greeting = payload.get("greeting") or {}
    engine = payload.get("engine") if isinstance(payload.get("engine"), dict) else {}
    resolved_token = payload.get("token") or payload.get("voicebot_token") or token
    greeting_pcm, greeting_hash = _decode_greeting_pcm(
        greeting,
        agent_id=payload.get("agent_id", ""),
        token=resolved_token,
    )
    first_message = payload.get("first_message") or greeting.get("text") or ""
    language = str(engine.get("language") or payload.get("language") or "").strip()
    gemini_voice = str(engine.get("voice") or "").strip()
    voice_id = str(payload.get("voice_id") or gemini_voice or "").strip()
    try:
        speed = float(engine.get("speed") if engine.get("speed") is not None else payload.get("speed", 1.0))
    except (TypeError, ValueError):
        speed = 1.0

    return AgentConfig(
        agent_id=payload["agent_id"],
        token=resolved_token,
        user_id=payload.get("user_id", ""),
        system_prompt=payload.get("system_prompt") or "",
        voice_id=voice_id,
        first_message=first_message,
        language=language,
        temperature=float(payload.get("temperature", 0.6)),
        max_tokens=int(payload.get("max_tokens", 45)),
        speed=speed,
        tone=payload.get("tone") or "",
        greeting_audio_pcm=greeting_pcm,
        greeting_audio_hash=greeting_hash,
        closing_message=str(payload.get("closing_message") or "").strip(),
        gemini_voice=gemini_voice,
        channel_reservation_id=(
            payload.get("channel_reservation_id")
            or payload.get("channel_id")
            or payload.get("reservation_id")
            or ""
        ),
        phone_number_id=str(payload.get("phone_number_id") or ""),
    )


def _auth_headers() -> dict:
    headers: dict[str, str] = {}
    if config.LOVABLE_API_SECRET:
        headers["Authorization"] = f"Bearer {config.LOVABLE_API_SECRET}"
    return headers


async def _fetch_agent_payload_via_supabase(fetch_fn, *, log_label: str) -> dict | None:
    """
    Try Supabase read; on anon-key RLS failure fall back to Moontech HTTP.
    Returns payload dict or None if caller should use HTTP.
    """
    if not supabase_agent_provider.is_configured():
        return None
    try:
        return await fetch_fn()
    except supabase_agent_provider.SupabaseAgentError as exc:
        if supabase_agent_provider.using_anon_key() and config.LOVABLE_APP_URL:
            logger.warning(
                "Supabase %s failed with anon key (%s) — falling back to Moontech HTTP",
                log_label,
                exc,
            )
            return None
        raise LovableClientError(str(exc)) from exc


async def _get_json_with_retry(
    url: str,
    *,
    params: dict | None,
    log_key: str,
    headers: dict | None = None,
) -> dict:
    client = await _get_client()

    response = None
    last_exc: Exception | None = None
    for attempt, timeout in enumerate((_AGENT_FETCH_TIMEOUT, _AGENT_FETCH_RETRY_TIMEOUT)):
        try:
            response = await client.get(url, params=params, headers=headers or {}, timeout=timeout)
            response.raise_for_status()
            last_exc = None
            break
        except httpx.HTTPStatusError as exc:
            logger.error(
                "Lovable agent lookup failed %s status=%s",
                log_key,
                exc.response.status_code,
            )
            raise LovableClientError(f"agent lookup failed: HTTP {exc.response.status_code}") from exc
        except httpx.HTTPError as exc:
            last_exc = exc
            detail = str(exc) or repr(exc.__cause__) or repr(exc)
            logger.warning(
                "Lovable agent lookup attempt %s/2 failed %s timeout=%.1fs error_type=%s error=%s",
                attempt + 1,
                log_key,
                timeout,
                type(exc).__name__,
                detail,
            )

    if last_exc is not None:
        detail = str(last_exc) or repr(last_exc.__cause__) or repr(last_exc)
        logger.error(
            "Lovable agent lookup connection error %s url=%s error_type=%s error=%s",
            log_key,
            url,
            type(last_exc).__name__,
            detail,
        )
        raise LovableClientError(
            f"agent lookup connection error ({type(last_exc).__name__}): {detail}"
        ) from last_exc

    try:
        return response.json()
    except ValueError as exc:
        logger.error("Lovable agent lookup returned non-JSON payload %s error=%s", log_key, exc)
        raise LovableClientError(f"malformed agent config payload: {exc}") from exc


async def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        async with _client_lock:
            if _client is None or _client.is_closed:
                from app.http_limits import default_limits, default_timeout

                _client = httpx.AsyncClient(
                    limits=default_limits(),
                    timeout=default_timeout(30.0),
                )
    return _client


async def close() -> None:
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()


async def fetch_agent(token: str) -> AgentConfig:
    """
    Resolve the agent config for `token`, using a 5-minute in-memory cache
    keyed by token so hot agents don't hit Lovable on every call. Raises
    LovableClientError on any failure (missing token, network error, or a
    non-2xx / malformed response) - callers must handle the fallback.
    """
    if not token:
        raise LovableClientError("missing token")
    if not supabase_agent_provider.is_configured() and not config.LOVABLE_APP_URL:
        raise LovableClientError("LOVABLE_APP_URL is not configured")

    cached = _agent_cache.get(token)
    if cached is not None and (time.monotonic() - cached[0]) < _CACHE_TTL_SECONDS:
        return cached[1]

    agent_config = await _fetch_agent_from_lovable(token)

    async with _cache_lock:
        cached = _agent_cache.get(token)
        if cached is not None and (time.monotonic() - cached[0]) < _CACHE_TTL_SECONDS:
            return cached[1]
        _agent_cache[token] = (time.monotonic(), agent_config)
    return agent_config


async def fetch_agent_by_number(number: str) -> AgentConfig:
    """
    Resolve inbound agent config via Supabase RPC ``get_inbound_agent_bundle``.
    Cached per number for 5 minutes.
    """
    if not number:
        raise LovableClientError("missing mobile number")
    if not supabase_agent_provider.is_configured():
        raise LovableClientError(
            "Supabase not configured — inbound calls require get_inbound_agent_bundle RPC"
        )

    cached = _agent_by_number_cache.get(number)
    if cached is not None and (time.monotonic() - cached[0]) < _CACHE_TTL_SECONDS:
        return cached[1]

    for cache_key in _number_cache_keys(number):
        cached = _agent_by_number_cache.get(cache_key)
        if cached is not None and (time.monotonic() - cached[0]) < _CACHE_TTL_SECONDS:
            return cached[1]

    agent_config = await _fetch_agent_by_number_from_lovable(number)

    async with _number_cache_lock:
        stored_at = time.monotonic()
        for cache_key in _number_cache_keys(number):
            _agent_by_number_cache[cache_key] = (stored_at, agent_config)
    return agent_config


async def fetch_agent_with_greeting(agent_id: str) -> AgentConfig:
    """
    Fetch agent config plus pre-generated greeting audio (raw PCM 8kHz) from
    GET /api/public/voicebot/agent-with-greeting/{agentId}. Cached per
    agent_id for 5 minutes. Raises LovableClientError on failure.
    """
    if not agent_id:
        raise LovableClientError("missing agent_id")
    if not supabase_agent_provider.is_configured() and not config.LOVABLE_APP_URL:
        raise LovableClientError("LOVABLE_APP_URL is not configured")

    cached = _agent_with_greeting_cache.get(agent_id)
    if cached is not None and (time.monotonic() - cached[0]) < _CACHE_TTL_SECONDS:
        return cached[1]

    agent_config = await _fetch_agent_with_greeting_from_lovable(agent_id)

    async with _greeting_cache_lock:
        cached = _agent_with_greeting_cache.get(agent_id)
        if cached is not None and (time.monotonic() - cached[0]) < _CACHE_TTL_SECONDS:
            return cached[1]
        _agent_with_greeting_cache[agent_id] = (time.monotonic(), agent_config)
    return agent_config


def _bundle_inflight_key(*, token: str | None = None, number: str | None = None) -> str:
    if token:
        return f"bundle:token:{token}"
    if number:
        return f"bundle:number:{normalize_phone_number(number) or number}"
    raise LovableClientError("missing token or number")


async def _resolve_agent_bundle_impl(
    *,
    token: str | None = None,
    number: str | None = None,
) -> AgentConfig:
    """Fetch agent config + optional pre-stored greeting."""
    started = time.perf_counter()

    if token:
        base = await fetch_agent(token)
    elif number:
        base = await fetch_agent_by_number(number)
    else:
        raise LovableClientError("missing token or number")

    if not base.greeting_audio_pcm and base.token:
        # Check in-memory hash cache first; only query RPC if not an inbound number lookup
        known_hash = _token_greeting_hash.get(base.token)
        if known_hash and known_hash in _greeting_pcm_by_hash:
            base.greeting_audio_pcm = _greeting_pcm_by_hash[known_hash]
            base.greeting_audio_hash = known_hash
        elif not number:
            pcm = await get_agent_greeting_pcm(base.token)
            if pcm is not None:
                base.greeting_audio_pcm = pcm
                base.greeting_audio_hash = _token_greeting_hash.get(base.token, "")

    if base.greeting_audio_pcm:
        logger.info(
            "LATENCY bundle agent_id=%s greeting=cached_pcm hash=%s bytes=%s ms=%.0f",
            base.agent_id,
            base.greeting_audio_hash[:12] if base.greeting_audio_hash else "embedded",
            len(base.greeting_audio_pcm),
            (time.perf_counter() - started) * 1000,
        )
        return base

    # Inbound/outbound RPC bundles already include greeting text (+ optional PCM).
    # Skip Moontech agent-with-greeting HTTP (~2–3 s); use live TTS when PCM missing/rejected.
    if number and not token:
        logger.info(
            "LATENCY bundle agent_id=%s greeting=%s ms=%.0f",
            base.agent_id,
            "pcm" if base.greeting_audio_pcm else "live_tts",
            (time.perf_counter() - started) * 1000,
        )
        return base

    # Token / outbound: get_agent_bundle_by_token RPC already returns prompt, voice,
    # greeting text, and optional greeting audio — skip extra Moontech HTTP (~2.5s).
    logger.info(
        "LATENCY bundle agent_id=%s greeting=%s ms=%.0f",
        base.agent_id,
        "pcm" if base.greeting_audio_pcm else "live_tts",
        (time.perf_counter() - started) * 1000,
    )
    return base


async def resolve_agent_bundle(
    *,
    token: str | None = None,
    number: str | None = None,
) -> AgentConfig:
    """
    Resolve full agent bundle with singleflight deduplication.

    Concurrent prewarm (HTTP / Exotel status) and WSS prefetch await the same
    in-flight task instead of issuing duplicate MoontechPro REST calls.
    """
    key = _bundle_inflight_key(token=token, number=number)

    async with _inflight_lock:
        task = _inflight_bundles.get(key)
        if task is not None and task.done():
            _inflight_bundles.pop(key, None)
            task = None
        if task is None:
            task = asyncio.create_task(
                _resolve_agent_bundle_impl(token=token, number=number)
            )
            _inflight_bundles[key] = task

    return await task


async def prewarm_agent_bundle(
    *,
    token: str | None = None,
    number: str | None = None,
    source: str = "unknown",
) -> bool:
    """
    Warm MoontechPro caches during ring / before WSS connect. Never raises.
    Intended to be scheduled with ``asyncio.create_task(...)``.
    """
    label = _token_prefix(token) if token else _number_suffix(number or "")
    try:
        await resolve_agent_bundle(token=token, number=number)
        logger.info("🔥 Prewarm complete source=%s key=%s", source, label)
        return True
    except LovableClientError as exc:
        logger.warning("🔥 Prewarm failed source=%s key=%s error=%s", source, label, exc)
        return False


async def _fetch_agent_with_greeting_from_lovable(agent_id: str) -> AgentConfig:
    payload = await _fetch_agent_payload_via_supabase(
        lambda: supabase_agent_provider.fetch_agent_payload_with_greeting(agent_id),
        log_label="agent-with-greeting",
    )
    if payload is not None:
        return _parse_agent_config_payload(payload)

    encoded = quote(agent_id, safe="")
    url = f"{config.LOVABLE_APP_URL.rstrip('/')}{_AGENT_WITH_GREETING_PATH}/{encoded}"
    try:
        payload = await _get_json_with_retry(
            url,
            params=None,
            log_key=f"agent_id={agent_id[:8]}...",
            headers=_auth_headers(),
        )
        return _parse_agent_config_payload(payload)
    except (KeyError, TypeError, ValueError) as exc:
        logger.error(
            "Lovable agent-with-greeting returned a malformed payload agent_id=%s error=%s",
            agent_id,
            exc,
        )
        raise LovableClientError(f"malformed agent config payload: {exc}") from exc


async def _fetch_agent_from_lovable(token: str) -> AgentConfig:
    payload = await _fetch_agent_payload_via_supabase(
        lambda: supabase_agent_provider.fetch_agent_payload_by_token(token),
        log_label="agent-by-token",
    )
    if payload is not None:
        return _parse_agent_config_payload(payload, token=token)

    url = f"{config.LOVABLE_APP_URL.rstrip('/')}{_AGENT_LOOKUP_PATH}"
    try:
        payload = await _get_json_with_retry(
            url,
            params={"token": token},
            log_key=f"token={_token_prefix(token)}...",
            headers=_auth_headers(),
        )
        return _parse_agent_config_payload(payload, token=token)
    except (KeyError, TypeError, ValueError) as exc:
        logger.error(
            "Lovable agent lookup returned a malformed payload token=%s... error=%s",
            _token_prefix(token),
            exc,
        )
        raise LovableClientError(f"malformed agent config payload: {exc}") from exc


async def _fetch_agent_by_number_from_lovable(number: str) -> AgentConfig:
    """Inbound agent resolution — Supabase RPC only (no REST / Moontech HTTP)."""
    try:
        payload = await supabase_agent_provider.fetch_agent_payload_by_number(number)
    except supabase_agent_provider.SupabaseAgentError as exc:
        raise LovableClientError(str(exc)) from exc
    try:
        return _parse_agent_config_payload(payload)
    except (KeyError, TypeError, ValueError) as exc:
        logger.error(
            "Inbound RPC bundle malformed for number=...%s error=%s",
            _number_suffix(number),
            exc,
        )
        raise LovableClientError(f"malformed agent config payload: {exc}") from exc


async def _spool_failed_call_log(payload: dict) -> None:
    def _write_line(path: str, text: str) -> None:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(text)

    try:
        os.makedirs(os.path.dirname(_FAILED_CALL_LOGS_PATH), exist_ok=True)
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        async with _call_log_spool_lock:
            await asyncio.to_thread(_write_line, _FAILED_CALL_LOGS_PATH, line)
        logger.warning(
            "Call-log spooled to disk for durable retry call_sid=%s path=%s",
            payload.get("provider_call_sid") or payload.get("call_sid"),
            _FAILED_CALL_LOGS_PATH,
        )
    except Exception as exc:
        logger.error("Failed to spool call-log to disk: %s", exc)


async def post_call_log(payload: dict) -> bool:
    """
    POST a call summary/transcript to Lovable at /api/public/voicebot/call-log
    with up to 3 retries and exponential backoff (1s, 2s, 4s).
    Preserves failed payloads in a durable disk spool if all retries fail.
    Never raises.
    """
    call_sid = payload.get("provider_call_sid") or payload.get("call_sid") or ""
    if not config.LOVABLE_APP_URL:
        logger.warning(
            "Skipping call-log post call_sid=%s: LOVABLE_APP_URL is not configured",
            call_sid,
        )
        return False

    url = f"{config.LOVABLE_APP_URL.rstrip('/')}{_CALL_LOG_PATH}"
    headers = {"Content-Type": "application/json"}
    if config.LOVABLE_API_SECRET:
        headers["Authorization"] = f"Bearer {config.LOVABLE_API_SECRET}"

    last_reason = "unknown"
    for attempt in range(1, _CALL_LOG_MAX_RETRIES + 1):
        try:
            client = await _get_client()
            response = await client.post(
                url,
                json=payload,
                headers=headers,
                timeout=_CALL_LOG_ATTEMPT_TIMEOUT,
            )
            if response.status_code < 400:
                logger.info(
                    "Lovable call-log posted call_sid=%s provider_call_sid=%s "
                    "agent_id=%s status=%s attempt=%s/%s",
                    payload.get("call_sid"),
                    payload.get("provider_call_sid"),
                    payload.get("agent_id"),
                    response.status_code,
                    attempt,
                    _CALL_LOG_MAX_RETRIES,
                )
                return True

            last_reason = f"HTTP {response.status_code}: {response.text[:160]}"
            logger.warning(
                "Lovable call-log attempt %s/%s failed call_sid=%s status=%s reason=%s",
                attempt,
                _CALL_LOG_MAX_RETRIES,
                call_sid,
                response.status_code,
                response.text[:160],
            )
            # Do not retry permanent 4xx client errors except 408/429
            if 400 <= response.status_code < 500 and response.status_code not in (408, 429):
                break
        except Exception as exc:  # noqa: BLE001
            detail = str(exc) or repr(getattr(exc, "__cause__", None)) or repr(exc)
            last_reason = f"{type(exc).__name__}: {detail}"
            logger.warning(
                "Lovable call-log attempt %s/%s error call_sid=%s reason=%s",
                attempt,
                _CALL_LOG_MAX_RETRIES,
                call_sid,
                last_reason,
            )

        if attempt < _CALL_LOG_MAX_RETRIES:
            backoff_s = float(2 ** (attempt - 1))
            await asyncio.sleep(backoff_s)

    logger.error(
        "Lovable call-log delivery failed after %s attempts call_sid=%s final_reason=%s",
        _CALL_LOG_MAX_RETRIES,
        call_sid,
        last_reason,
    )
    await _spool_failed_call_log(payload)
    return False


async def _post_exotel_webhook(path: str, number: str, fields: dict) -> None:
    """
    Fire-and-forget: relay Exotel's raw form fields (CallSid, From, To,
    CallStatus, CallDuration, RecordingUrl, ...) to a Lovable hook as a
    form-encoded POST, identified only by `provider=exotel&number=<number>`
    in the query string - never an agent_id. Never raises.
    """
    if not config.LOVABLE_APP_URL:
        logger.warning("Skipping Exotel webhook relay to %s: LOVABLE_APP_URL is not configured", path)
        return
    if not number:
        logger.warning("Skipping Exotel webhook relay to %s: missing caller number", path)
        return

    url = f"{config.LOVABLE_APP_URL.rstrip('/')}{path}"
    params = {"provider": "exotel", "number": number}
    headers = {}
    if config.LOVABLE_API_SECRET:
        headers["Authorization"] = f"Bearer {config.LOVABLE_API_SECRET}"

    try:
        client = await _get_client()
        response = await client.post(
            url,
            params=params,
            data=fields,
            headers=headers,
            timeout=_EXOTEL_WEBHOOK_TIMEOUT,
        )
        if response.status_code >= 400:
            logger.error(
                "Lovable Exotel webhook relay failed path=%s number=...%s status=%s body=%s",
                path,
                number[-4:],
                response.status_code,
                response.text[:200],
            )
        else:
            logger.info(
                "Lovable Exotel webhook relayed path=%s number=...%s call_sid=%s",
                path,
                number[-4:],
                fields.get("CallSid"),
            )
    except Exception as exc:  # noqa: BLE001 - a Lovable outage must never crash the voicebot
        detail = str(exc) or repr(getattr(exc, "__cause__", None)) or repr(exc)
        logger.error(
            "Lovable Exotel webhook relay error path=%s number=...%s url=%s error_type=%s error=%s",
            path,
            number[-4:] if number else "",
            url,
            type(exc).__name__,
            detail,
        )


async def post_exotel_status(number: str, fields: dict) -> None:
    """Relay an Exotel Status Callback (CallStatus/CallDuration/RecordingUrl/...) to Lovable."""
    await _post_exotel_webhook(_VOICE_STATUS_PATH, number, fields)


async def post_exotel_conversation(number: str, fields: dict) -> None:
    """Relay an Exotel conversation applet callback to Lovable."""
    await _post_exotel_webhook(_VOICE_CONVERSATION_PATH, number, fields)
