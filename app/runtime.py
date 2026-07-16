"""
Shared runtime resources for high-concurrency voice sessions.

Centralizes admission control, outbound API concurrency limits, and the VAD
thread pool so every module uses the same tuned caps from config.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

from app import config

# CPU-bound VAD/RMS work is offloaded here so the asyncio event loop stays responsive.
vad_executor = ThreadPoolExecutor(
    max_workers=config.VAD_WORKER_THREADS,
    thread_name_prefix="vad",
)

# Hard cap on simultaneous WSS voice sessions for this process.
session_semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_SESSIONS)

# Limit parallel outbound calls to ElevenLabs (STT/TTS are the heaviest APIs).
stt_semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_STT)
tts_semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_TTS)

_active_sessions = 0
_active_sessions_lock = asyncio.Lock()


async def increment_active_sessions() -> int:
    global _active_sessions
    async with _active_sessions_lock:
        _active_sessions += 1
        return _active_sessions


async def decrement_active_sessions() -> int:
    global _active_sessions
    async with _active_sessions_lock:
        _active_sessions = max(0, _active_sessions - 1)
        return _active_sessions


async def active_sessions_count() -> int:
    async with _active_sessions_lock:
        return _active_sessions


async def try_acquire_session() -> bool:
    """
    Admit a new WebSocket call without blocking when at capacity.

    Do NOT use ``asyncio.wait_for(..., timeout=0)`` on ``Semaphore.acquire()`` —
    timeout=0 cancels before the event loop schedules acquire, rejecting every
    call even when slots are free.
    """
    if session_semaphore.locked():
        return False
    await session_semaphore.acquire()
    return True


def release_session_slot() -> None:
    """Return one session slot acquired via try_acquire_session()."""
    session_semaphore.release()


def shutdown_runtime() -> None:
    vad_executor.shutdown(wait=False, cancel_futures=True)
