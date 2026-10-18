"""
Transport connection pool (no .env numbers/tokens).

At boot: Supabase client, HTTP pools, idle ElevenLabs STT/TTS websockets.
Per call: borrow sockets from pool; agent config always from Supabase RPC
(vobiz/answer prewarm + WSS prefetch use token/number from the live API).
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from app import config

if TYPE_CHECKING:
    from app.streaming.providers.elevenlabs_stt import ElevenLabsStreamingSTTProvider
    from app.streaming.providers.elevenlabs_tts_ws import ElevenLabsWebSocketTTSProvider

logger = logging.getLogger(__name__)

_maintain_task: asyncio.Task | None = None
_tts_pool: asyncio.Queue[ElevenLabsWebSocketTTSProvider] = asyncio.Queue()
_stt_pool: asyncio.Queue[ElevenLabsStreamingSTTProvider] = asyncio.Queue()
_gemini_ws_pool: asyncio.Queue = asyncio.Queue()
_pool_lock = asyncio.Lock()


async def _prime_supabase() -> bool:
    if not config.SUPABASE_URL:
        return False
    try:
        from app import supabase_agent_provider

        if not supabase_agent_provider.is_configured():
            return False
        await supabase_agent_provider._get_supabase()
        return True
    except Exception as exc:
        logger.warning("Supabase warm skipped: %s", exc)
        return False


async def _prime_http_clients() -> None:
    try:
        from app import lovable_client

        await lovable_client._get_client()
    except Exception as exc:
        logger.warning("Moontech HTTP client warm skipped: %s", exc)
    try:
        from app.elevenlabs_service import elevenlabs_service

        await elevenlabs_service._get_client()
    except Exception as exc:
        logger.debug("ElevenLabs HTTP warm skipped: %s", exc)
    try:
        from app.openai_service import openai_brain

        if openai_brain._client is not None:
            await openai_brain.warm()
    except Exception:
        pass


async def _create_pooled_tts(slot: int) -> ElevenLabsWebSocketTTSProvider | None:
    from app.streaming.providers.elevenlabs_tts_ws import ElevenLabsWebSocketTTSProvider

    provider = ElevenLabsWebSocketTTSProvider(f"_tts_pool_{slot}_")
    try:
        await provider.connect()
        return provider
    except Exception as exc:
        logger.warning("TTS pool slot %s failed: %s", slot, exc)
        try:
            await provider.close()
        except Exception:
            pass
        return None


async def _create_pooled_stt(slot: int) -> ElevenLabsStreamingSTTProvider | None:
    from app.streaming.providers.elevenlabs_stt import ElevenLabsStreamingSTTProvider

    provider = ElevenLabsStreamingSTTProvider(f"_stt_pool_{slot}_")
    try:
        await provider.connect()
        if not await provider.wait_session_ready(8.0):
            logger.warning("STT pool slot %s session_started timeout", slot)
            await provider.close()
            return None
        return provider
    except Exception as exc:
        logger.warning("STT pool slot %s failed: %s", slot, exc)
        try:
            await provider.close()
        except Exception:
            pass
        return None


async def _fill_tts_pool() -> int:
    size = max(0, config.VOBIZ_TTS_POOL_SIZE)
    if size == 0 or not config.ELEVENLABS_API_KEY:
        return _tts_pool.qsize()
    if config.TTS_PROVIDER not in ("elevenlabs_ws", ""):
        return _tts_pool.qsize()
    async with _pool_lock:
        while _tts_pool.qsize() < size:
            provider = await _create_pooled_tts(_tts_pool.qsize())
            if provider is None:
                break
            await _tts_pool.put(provider)
    return _tts_pool.qsize()


async def _fill_stt_pool() -> int:
    size = max(0, config.VOBIZ_STT_POOL_SIZE)
    if size == 0 or not config.ELEVENLABS_API_KEY:
        return _stt_pool.qsize()
    if config.STT_PROVIDER not in ("elevenlabs", ""):
        return _stt_pool.qsize()
    async with _pool_lock:
        while _stt_pool.qsize() < size:
            provider = await _create_pooled_stt(_stt_pool.qsize())
            if provider is None:
                break
            await _stt_pool.put(provider)
    return _stt_pool.qsize()


async def borrow_tts(connection_id: str) -> ElevenLabsWebSocketTTSProvider | None:
    if config.VOBIZ_TTS_POOL_SIZE <= 0:
        return None
    try:
        provider = _tts_pool.get_nowait()
    except asyncio.QueueEmpty:
        return None
    provider._connection_id = connection_id
    provider._prepare_for_reuse()
    asyncio.create_task(_top_up_tts_pool())
    return provider


async def borrow_stt(connection_id: str) -> ElevenLabsStreamingSTTProvider | None:
    if config.VOBIZ_STT_POOL_SIZE <= 0:
        return None
    try:
        provider = _stt_pool.get_nowait()
    except asyncio.QueueEmpty:
        return None
    provider._connection_id = connection_id
    provider._prepare_for_reuse()
    if provider._ws is None or not provider._session_started.is_set():
        try:
            await provider.connect()
            await provider.wait_session_ready(8.0)
        except Exception as exc:
            logger.warning("STT pool borrow reconnect failed conn=%s: %s", connection_id, exc)
            asyncio.create_task(_top_up_stt_pool())
            return None
    asyncio.create_task(_top_up_stt_pool())
    return provider


async def _top_up_tts_pool() -> None:
    async with _pool_lock:
        need = config.VOBIZ_TTS_POOL_SIZE - _tts_pool.qsize()
        for _ in range(max(0, need)):
            provider = await _create_pooled_tts(_tts_pool.qsize())
            if provider is not None:
                await _tts_pool.put(provider)


async def _top_up_stt_pool() -> None:
    async with _pool_lock:
        need = config.VOBIZ_STT_POOL_SIZE - _stt_pool.qsize()
        for _ in range(max(0, need)):
            provider = await _create_pooled_stt(_stt_pool.qsize())
            if provider is not None:
                await _stt_pool.put(provider)


async def release_tts(provider: ElevenLabsWebSocketTTSProvider) -> None:
    """Close the call socket. A fresh pool socket is opened so idle streams don't time out."""
    try:
        await provider.close()
    except Exception:
        pass
    asyncio.create_task(_top_up_tts_pool())


async def release_stt(provider: ElevenLabsStreamingSTTProvider) -> None:
    """STT sessions are call-scoped; recycle by closing and refilling the pool."""
    try:
        await provider.close()
    except Exception:
        pass
    asyncio.create_task(_top_up_stt_pool())


async def _open_gemini_socket():
    if not config.GEMINI_API_KEY:
        return None
    import websockets

    url = f"{config.GEMINI_LIVE_WS_URL}?key={config.GEMINI_API_KEY}"
    try:
        return await websockets.connect(
            url,
            ping_interval=20,
            ping_timeout=10,
            max_size=16 * 1024 * 1024,
        )
    except Exception as exc:
        logger.warning("Gemini Live socket warm failed: %s", exc)
        return None


async def _fill_gemini_pool() -> int:
    if config.VOICE_PIPELINE != "gemini_live" or not config.GEMINI_API_KEY:
        return _gemini_ws_pool.qsize()
    target = max(1, config.VOBIZ_TTS_POOL_SIZE)
    async with _pool_lock:
        while _gemini_ws_pool.qsize() < target:
            ws = await _open_gemini_socket()
            if ws is None:
                break
            await _gemini_ws_pool.put(ws)
    return _gemini_ws_pool.qsize()


async def borrow_gemini_socket():
    while not _gemini_ws_pool.empty():
        try:
            ws = _gemini_ws_pool.get_nowait()
        except asyncio.QueueEmpty:
            break
        if getattr(ws, "close_code", None) is None:
            asyncio.create_task(_fill_gemini_pool())
            return ws
        try:
            await ws.close()
        except Exception:
            pass
    asyncio.create_task(_fill_gemini_pool())
    return None


async def _warm_inbound_gemini_agents() -> None:
    """Pre-fetch inbound agent bundles and pre-synthesize greeting PCM at startup."""
    raw_nums = getattr(config, "PREWARM_INBOUND_NUMBERS", "") or ""
    numbers = [n.strip() for n in raw_nums.split(",") if n.strip()]
    if not numbers:
        return
    try:
        from app import lovable_client
        from app.telephony.vobiz.gemini_session import (
            _synthesized_greeting_pcm_cache,
            save_synthesized_greeting_pcm,
        )
        from app.voice_pipeline.gemini_live.link import GeminiLiveLink
        from app.voice_pipeline.gemini_live.prewarm import (
            build_gemini_instructions_for_agent,
            build_welcome_text,
        )

        try:
            import audioop
        except ImportError:
            audioop = None  # type: ignore[assignment]

        for num in numbers:
            try:
                agent_cfg = await lovable_client.resolve_agent_bundle(number=num)
            except Exception as exc:
                logger.debug("Startup inbound bundle warm skipped for ...%s: %s", num[-4:], exc)
                continue

            welcome = build_welcome_text(agent_cfg)
            if (
                not welcome
                or agent_cfg.greeting_audio_pcm
                or welcome in _synthesized_greeting_pcm_cache
                or audioop is None
            ):
                continue

            # Pre-synthesize the greeting once using Gemini Live so the first call has 0ms TTS delay
            pcm_buf = bytearray()
            state = [None]
            done_ev = asyncio.Event()

            async def _on_audio(pcm_24k: bytes) -> None:
                if pcm_24k:
                    pcm_8k, state[0] = audioop.ratecv(
                        pcm_24k, 2, 1, 24000, 8000, state[0]
                    )
                    pcm_buf.extend(pcm_8k)

            async def _on_done() -> None:
                done_ev.set()

            link = GeminiLiveLink(
                "startup_greeting_warm",
                on_audio=_on_audio,
                on_turn_complete=_on_done,
            )
            try:
                await link.connect(build_gemini_instructions_for_agent(agent_cfg))
                await link.say(
                    "The phone call just connected. Speak ONLY this opening greeting "
                    f"and do NOT call any tools: {welcome}"
                )
                await asyncio.wait_for(done_ev.wait(), timeout=6.0)
                if len(pcm_buf) >= 3200:
                    save_synthesized_greeting_pcm(welcome, bytes(pcm_buf))
                    logger.info(
                        "🎵 Startup pre-synthesized Gemini greeting agent_id=%s bytes=%s",
                        agent_cfg.agent_id,
                        len(pcm_buf),
                    )
            except Exception as exc:
                logger.debug("Startup greeting synth skipped: %s", exc)
            finally:
                await link.close()
    except Exception as exc:
        logger.debug("Inbound Gemini prewarm skipped: %s", exc)


async def _maintain_loop() -> None:
    interval = max(60, config.VOBIZ_WARM_POOL_REFRESH_SEC)
    while True:
        await asyncio.sleep(interval)
        await _prime_supabase()
        if config.VOICE_PIPELINE == "gemini_live":
            await _fill_gemini_pool()
            asyncio.create_task(_warm_inbound_gemini_agents())
        else:
            await _fill_tts_pool()
            await _fill_stt_pool()


def start_background_warm_pool() -> None:
    global _maintain_task
    if not config.VOBIZ_WARM_POOL_ENABLED:
        return
    if _maintain_task is not None and not _maintain_task.done():
        return
    _maintain_task = asyncio.create_task(_maintain_loop())


async def warm_on_process_start() -> None:
    if not config.VOBIZ_WARM_POOL_ENABLED:
        return

    supabase_ok = await _prime_supabase()
    if config.VOICE_PIPELINE == "gemini_live":
        try:
            from app import lovable_client

            await lovable_client._get_client()
        except Exception:
            pass
        gemini_slots = await _fill_gemini_pool()
        asyncio.create_task(_warm_inbound_gemini_agents())
        logger.info(
            "🔥 Gemini transport pool ready supabase=%s gemini_ws_slots=%s",
            supabase_ok,
            gemini_slots,
        )
        start_background_warm_pool()
        return

    await _prime_http_clients()
    tts_slots = await _fill_tts_pool()
    stt_slots = await _fill_stt_pool()

    logger.info(
        "🔥 Transport pool ready supabase=%s tts_slots=%s stt_slots=%s "
        "(agent config per call via Supabase RPC + /vobiz/answer prewarm)",
        supabase_ok,
        tts_slots,
        stt_slots,
    )
    start_background_warm_pool()
