"""Process-level warm connections so the first syllable on a new call is faster."""

from __future__ import annotations

import asyncio
import logging

from app import config

logger = logging.getLogger(__name__)

_warm_task: asyncio.Task | None = None


async def _prime_elevenlabs_tts_once() -> None:
    if config.VOICE_PIPELINE == "openai_realtime" or not config.ELEVENLABS_API_KEY:
        return
    if not config.STREAMING_PIPELINE or config.TTS_PROVIDER != "elevenlabs_ws":
        return
    from app.streaming.providers.elevenlabs_tts_ws import ElevenLabsWebSocketTTSProvider

    provider = ElevenLabsWebSocketTTSProvider("_startup_warm_")
    try:
        await provider.connect()
        logger.info("✅ ElevenLabs TTS connection primed (TLS + WS handshake)")
    except Exception as exc:
        logger.warning("ElevenLabs TTS prime skipped: %s", exc)
    finally:
        try:
            await provider.close()
        except Exception:
            pass


async def _maintain_loop() -> None:
    """Keep periodic handshakes so idle servers stay hot (optional background)."""
    interval = max(60, config.VOBIZ_WARM_POOL_REFRESH_SEC)
    while True:
        await asyncio.sleep(interval)
        await _prime_elevenlabs_tts_once()


def start_background_warm_pool() -> None:
    global _warm_task
    if not config.VOBIZ_WARM_POOL_ENABLED:
        return
    if _warm_task is not None and not _warm_task.done():
        return
    _warm_task = asyncio.create_task(_maintain_loop())


async def warm_on_process_start() -> None:
    """Run once at boot before accepting calls."""
    if not config.VOBIZ_WARM_POOL_ENABLED:
        return
    await _prime_elevenlabs_tts_once()
    start_background_warm_pool()
