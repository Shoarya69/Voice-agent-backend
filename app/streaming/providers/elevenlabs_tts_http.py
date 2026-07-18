"""
HTTP streaming TTS fallback — one request per clause (rollback path).

Uses the existing ElevenLabs REST stream endpoint from ``elevenlabs_service``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

from app.elevenlabs_service import ElevenLabsError, elevenlabs_service
from app.streaming.providers.base import StreamingTTSProvider

logger = logging.getLogger(__name__)

_UTTERANCE_END = object()


class ElevenLabsHttpTTSProvider(StreamingTTSProvider):
    """Clause-at-a-time HTTP TTS (legacy optimized path)."""

    def __init__(self, connection_id: str) -> None:
        self._connection_id = connection_id
        self._audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._pending_text: asyncio.Queue[str | None] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None
        self._closed = False
        self._utterance_done = asyncio.Event()
        self._utterance_audio: asyncio.Queue[bytes | None] = asyncio.Queue()

    async def connect(self) -> None:
        self._worker_task = asyncio.create_task(self._worker())

    async def begin_utterance(self) -> None:
        while not self._utterance_audio.empty():
            try:
                self._utterance_audio.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._utterance_done.clear()

    async def send_text(self, text: str) -> None:
        chunk = (text or "").strip()
        if chunk:
            await self._pending_text.put(chunk)

    async def flush_utterance(self) -> None:
        self._utterance_done.clear()
        await self._pending_text.put(_UTTERANCE_END)
        try:
            await asyncio.wait_for(self._utterance_done.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning(
                "HTTP TTS utterance flush timed out conn=%s",
                self._connection_id,
            )

    async def iter_utterance_audio(self) -> AsyncIterator[bytes]:
        while True:
            try:
                chunk = await asyncio.wait_for(self._utterance_audio.get(), timeout=0.25)
            except asyncio.TimeoutError:
                if self._utterance_done.is_set() and self._utterance_audio.empty():
                    break
                continue
            if chunk is None:
                break
            yield chunk

    async def audio_chunks(self) -> AsyncIterator[bytes]:
        async for chunk in self.iter_utterance_audio():
            yield chunk

    async def close(self) -> None:
        self._closed = True
        await self._pending_text.put(None)
        if self._worker_task is not None:
            await self._worker_task
        await self._audio_queue.put(None)

    async def _worker(self) -> None:
        try:
            while not self._closed:
                text = await self._pending_text.get()
                if text is None:
                    break
                if text is _UTTERANCE_END:
                    self._utterance_done.set()
                    continue
                try:
                    async for raw in elevenlabs_service.stream_text_to_speech(text):
                        if raw:
                            await self._utterance_audio.put(raw)
                except ElevenLabsError as exc:
                    logger.error(
                        "HTTP TTS error conn=%s: %s",
                        self._connection_id,
                        exc,
                    )
        finally:
            self._utterance_done.set()
            await self._utterance_audio.put(None)
