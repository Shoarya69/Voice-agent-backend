"""
Portal bootstrap for the Gemini Live engine.

The Moontech admin (Admin → AI Keys) is the source of truth for the Gemini
API key and live model. This module caches that config in memory and refreshes
it on a TTL. A failed refresh keeps the last good config. The key is never logged.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

import httpx

from app import config

logger = logging.getLogger(__name__)

_BOOTSTRAP_PATH = "/api/public/voicebot/bootstrap"
_DEFAULT_TTL_SECONDS = 60
_FETCH_TIMEOUT = 8.0

_cache: "EngineConfig | None" = None
_expires_at = 0.0
_lock = asyncio.Lock()
_refresh_task: asyncio.Task | None = None
_pool_key: str = ""


class GeminiAuthError(Exception):
    """Gemini rejected the API key. Safe to retry once after a bootstrap refresh."""


class GeminiVoiceError(Exception):
    """Gemini rejected the requested voice. Retry once with the portal default voice."""


# Gemini Live prebuilt voices. Anything else (for example an ElevenLabs voice id)
# is not sent to Gemini; the portal default voice is used instead.
_GEMINI_PREBUILT_VOICES = frozenset(
    {
        "achernar",
        "achird",
        "algenib",
        "algieba",
        "alnilam",
        "aoede",
        "autonoe",
        "callirrhoe",
        "charon",
        "despina",
        "enceladus",
        "erinome",
        "fenrir",
        "gacrux",
        "iapetus",
        "kore",
        "laomedeia",
        "leda",
        "orus",
        "puck",
        "pulcherrima",
        "rasalgethi",
        "sadachbia",
        "sadaltager",
        "schedar",
        "sulafat",
        "umbriel",
        "vindemiatrix",
        "zephyr",
        "zubenelgenubi",
    }
)


def is_gemini_prebuilt_voice(name: str) -> bool:
    return (name or "").strip().lower() in _GEMINI_PREBUILT_VOICES


def gemini_voice_or_default(requested: str, default_voice: str) -> str:
    """Use a Gemini prebuilt voice. ElevenLabs ids fall back to the portal default."""
    fallback = (default_voice or "Aoede").strip() or "Aoede"
    name = (requested or "").strip()
    if is_gemini_prebuilt_voice(name):
        return name
    if name and name != fallback:
        logger.info(
            "Gemini voice %s is not a Live prebuilt voice; using default %s",
            name[:8],
            fallback,
        )
    return fallback


def is_gemini_voice_rejection(reason: str, code: int | None = None) -> bool:
    text = (reason or "").lower()
    if "voice" in text and ("not available" in text or "invalid" in text or "api_name" in text):
        return True
    return code == 1007 and "voice" in text


class EngineConfigError(Exception):
    """Bootstrap did not return a usable Gemini engine config."""


@dataclass(frozen=True)
class EngineConfig:
    engine: str
    gemini_api_key: str
    gemini_live_model: str
    gemini_default_voice: str
    ttl_seconds: int
    fetched_at: float

    def masked_key(self) -> str:
        return mask_secret(self.gemini_api_key)


def mask_secret(value: str) -> str:
    raw = (value or "").strip()
    if len(raw) < 8:
        return "••••"
    return f"{raw[:4]}••••{raw[-4:]}"


def shared_secret() -> str:
    return (config.VOICEBOT_SHARED_SECRET or config.LOVABLE_API_SECRET or "").strip()


def current_engine_config() -> EngineConfig | None:
    """Last good config, including one that is past TTL. Never performs I/O."""
    return _cache


def pooled_gemini_key() -> str:
    """Key the warm socket pool was filled with. Empty before the first fill."""
    return _pool_key


def note_pooled_gemini_key(api_key: str) -> None:
    global _pool_key
    _pool_key = api_key or ""


def is_gemini_auth_failure(exc: BaseException) -> bool:
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    try:
        code_int = int(code) if code is not None else None
    except (TypeError, ValueError):
        code_int = None
    if code_int in (401, 403, 1008):
        return True
    text = f"{type(exc).__name__} {exc}".lower()
    return any(
        token in text
        for token in (
            "401",
            "403",
            "api key",
            "api_key",
            "unauth",
            "permission denied",
            "invalid authentication",
        )
    )


async def get_engine_config(*, force: bool = False) -> EngineConfig:
    """
    Return the cached portal engine config.

    ``force=True`` fetches immediately (used once after a Gemini 401/403).
    A failed fetch keeps the previous config when one exists.
    """
    global _cache, _expires_at
    now = time.monotonic()
    if not force and _cache is not None and now < _expires_at:
        return _cache

    async with _lock:
        now = time.monotonic()
        if not force and _cache is not None and now < _expires_at:
            return _cache
        try:
            fresh = await _fetch_bootstrap()
        except Exception as exc:
            if _cache is not None:
                logger.warning(
                    "Gemini bootstrap refresh failed; keeping last config model=%s voice=%s key=%s error=%s",
                    _cache.gemini_live_model,
                    _cache.gemini_default_voice,
                    _cache.masked_key(),
                    type(exc).__name__,
                )
                return _cache
            raise EngineConfigError(str(exc)) from exc

        key_changed = _cache is None or _cache.gemini_api_key != fresh.gemini_api_key
        model_changed = _cache is None or _cache.gemini_live_model != fresh.gemini_live_model
        _cache = fresh
        _expires_at = time.monotonic() + max(15, fresh.ttl_seconds)
        logger.info(
            "Gemini bootstrap ready engine=%s model=%s voice=%s key=%s ttl_s=%s key_changed=%s",
            fresh.engine,
            fresh.gemini_live_model,
            fresh.gemini_default_voice,
            fresh.masked_key(),
            fresh.ttl_seconds,
            key_changed,
        )
        if key_changed or model_changed:
            _schedule_pool_refill()
        return fresh


def start_engine_config_refresh() -> None:
    """Background TTL refresh. Safe to call more than once."""
    global _refresh_task
    if _refresh_task is not None and not _refresh_task.done():
        return
    _refresh_task = asyncio.create_task(_refresh_loop())


async def _refresh_loop() -> None:
    while True:
        ttl = _cache.ttl_seconds if _cache is not None else _DEFAULT_TTL_SECONDS
        await asyncio.sleep(max(15, ttl))
        try:
            await get_engine_config(force=True)
        except Exception as exc:
            logger.warning("Gemini bootstrap background refresh skipped: %s", type(exc).__name__)


def _schedule_pool_refill() -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    async def _refill() -> None:
        try:
            from app.streaming import warm_pool

            await warm_pool.refill_gemini_pool_for_new_key()
        except Exception as exc:
            logger.warning("Gemini pool refill after bootstrap change skipped: %s", exc)

    loop.create_task(_refill())


async def _fetch_bootstrap() -> EngineConfig:
    base = (config.LOVABLE_APP_URL or "").rstrip("/")
    secret = shared_secret()
    if not base:
        raise EngineConfigError("LOVABLE_APP_URL is not configured")
    if not secret:
        raise EngineConfigError("VOICEBOT_SHARED_SECRET is not configured")

    url = f"{base}{_BOOTSTRAP_PATH}"
    headers = {"Authorization": f"Bearer {secret}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT) as client:
        response = await client.post(url, headers=headers, json={})
        if response.status_code in (401, 403):
            raise EngineConfigError(f"bootstrap unauthorized HTTP {response.status_code}")
        response.raise_for_status()
        payload = response.json()

    if not isinstance(payload, dict):
        raise EngineConfigError("bootstrap payload was not an object")

    engine = str(payload.get("engine") or "gemini").strip().lower() or "gemini"
    api_key = str(payload.get("gemini_api_key") or "").strip()
    model = str(payload.get("gemini_live_model") or "").strip()
    voice = str(payload.get("gemini_default_voice") or "Kore").strip() or "Kore"
    try:
        ttl = int(payload.get("config_ttl_seconds") or _DEFAULT_TTL_SECONDS)
    except (TypeError, ValueError):
        ttl = _DEFAULT_TTL_SECONDS

    if engine != "gemini":
        logger.warning("Bootstrap engine=%s; voice server still uses Gemini Live", engine)
    if not api_key or not model:
        raise EngineConfigError("bootstrap missing gemini_api_key or gemini_live_model")
    if model.startswith("models/"):
        model = model[len("models/") :]

    return EngineConfig(
        engine="gemini",
        gemini_api_key=api_key,
        gemini_live_model=model,
        gemini_default_voice=voice,
        ttl_seconds=max(15, ttl),
        fetched_at=time.time(),
    )


def reset_engine_config_for_tests() -> None:
    """Clear the in-memory cache. Tests only."""
    global _cache, _expires_at, _pool_key, _refresh_task
    _cache = None
    _expires_at = 0.0
    _pool_key = ""
    if _refresh_task is not None and not _refresh_task.done():
        _refresh_task.cancel()
    _refresh_task = None
