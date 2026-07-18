"""
Deepgram live STT fallback (optional).

Requires ``deepgram-sdk`` and ``DEEPGRAM_API_KEY`` when ``STT_PROVIDER=deepgram``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator

from app import config
from app.streaming.providers.base import STTEvent, STTEventKind, StreamingSTTProvider

logger = logging.getLogger(__name__)


class DeepgramStreamingSTTProvider(StreamingSTTProvider):
    def __init__(self, connection_id: str) -> None:
        self._connection_id = connection_id
        self._event_queue: asyncio.Queue[STTEvent | None] = asyncio.Queue()
        self._connection = None
        self._client = None
        self._closed = False

    async def connect(self) -> None:
        if not config.DEEPGRAM_API_KEY:
            raise RuntimeError("DEEPGRAM_API_KEY is not configured")

        try:
            from deepgram import AsyncDeepgramClient
            from deepgram.core.events import EventType
        except ImportError as exc:
            raise RuntimeError(
                "deepgram-sdk is required for STT_PROVIDER=deepgram"
            ) from exc

        self._EventType = EventType
        self._client = AsyncDeepgramClient(api_key=config.DEEPGRAM_API_KEY)
        self._connection = await self._client.listen.v1.connect(
            model=config.DEEPGRAM_STT_MODEL,
            encoding="linear16",
            sample_rate=str(config.SAMPLE_RATE),
            channels="1",
            punctuate="true",
            interim_results="true",
            endpointing=str(config.STREAMING_STT_ENDPOINTING_MS),
            smart_format="true",
        )

        async def on_message(message, **_kwargs):
            await self._handle_message(message)

        self._connection.on(EventType.MESSAGE, on_message)
        await self._connection.start_listening()
        logger.debug("Deepgram streaming STT connected for %s", self._connection_id)

    async def send_audio(self, pcm: bytes, *, commit: bool = False) -> None:
        if not pcm or self._connection is None or self._closed:
            return
        if commit:
            try:
                await self._connection.send(json.dumps({"type": "Finalize"}))
            except Exception as exc:
                await self._push_error(str(exc))
            return
        try:
            await self._connection.send(pcm)
        except Exception as exc:
            await self._push_error(str(exc))

    async def events(self) -> AsyncIterator[STTEvent]:
        while True:
            event = await self._event_queue.get()
            if event is None:
                break
            yield event

    async def close(self) -> None:
        self._closed = True
        if self._connection is not None:
            try:
                await self._connection.finish()
            except Exception:
                pass
            self._connection = None
        await self._event_queue.put(None)

    async def _push_error(self, message: str) -> None:
        await self._event_queue.put(
            STTEvent(kind=STTEventKind.ERROR, error=message)
        )

    async def _handle_message(self, message) -> None:
        msg_type = getattr(message, "type", None) or getattr(message, "event", "")
        if msg_type != "Results":
            return

        channel = getattr(message, "channel", None)
        if channel is None:
            return
        alternatives = getattr(channel, "alternatives", None) or []
        if not alternatives:
            return
        alt = alternatives[0]
        text = (getattr(alt, "transcript", None) or "").strip()
        if not text:
            return

        is_final = bool(getattr(message, "is_final", False) or getattr(message, "speech_final", False))
        kind = STTEventKind.COMMITTED if is_final else STTEventKind.PARTIAL
        await self._event_queue.put(STTEvent(kind=kind, text=text))
