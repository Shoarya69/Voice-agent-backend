"""
ElevenLabs Scribe v2 Realtime STT over WebSocket.

API: wss://api.elevenlabs.io/v1/speech-to-text/realtime
Docs: https://elevenlabs.io/docs/api-reference/speech-to-text/v-1-speech-to-text-realtime
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import AsyncIterator
from urllib.parse import urlencode

import websockets
import websockets.exceptions

from app import config
from app.streaming.providers.base import STTEvent, STTEventKind, StreamingSTTProvider

logger = logging.getLogger(__name__)


class ElevenLabsStreamingSTTProvider(StreamingSTTProvider):
    def __init__(self, connection_id: str) -> None:
        self._connection_id = connection_id
        self._ws = None
        self._event_queue: asyncio.Queue[STTEvent | None] = asyncio.Queue()
        self._reader_task: asyncio.Task | None = None
        self._closed = False
        self._session_started = asyncio.Event()
        self._connect_lock = asyncio.Lock()
        self._chunks_sent = 0
        self._reconnects = 0

    def _build_uri(self) -> str:
        base = config.ELEVENLABS_BASE_URL.rstrip("/").replace("https://", "wss://")
        params = {
            "model_id": config.STREAMING_STT_MODEL,
            "audio_format": f"pcm_{config.SAMPLE_RATE}",
            "commit_strategy": "vad",
            "vad_silence_threshold_secs": config.STREAMING_STT_VAD_SILENCE_SECS,
            "vad_threshold": config.STREAMING_STT_VAD_THRESHOLD,
            "min_speech_duration_ms": config.STREAMING_STT_MIN_SPEECH_MS,
            "min_silence_duration_ms": config.STREAMING_STT_MIN_SILENCE_MS,
        }
        if config.ELEVENLABS_STT_LANGUAGE:
            params["language_code"] = config.ELEVENLABS_STT_LANGUAGE
        return f"{base}/v1/speech-to-text/realtime?{urlencode(params)}"

    async def connect(self) -> None:
        await self._open_connection()

    async def ensure_live(self) -> None:
        """Reconnect if the STT websocket died (e.g. idle during long greeting TTS)."""
        if self._closed:
            return
        if self._ws is not None and self._session_started.is_set():
            return
        async with self._connect_lock:
            if self._ws is not None and self._session_started.is_set():
                return
            await self._open_connection()

    async def _open_connection(self) -> None:
        if not config.ELEVENLABS_API_KEY:
            raise RuntimeError("ELEVENLABS_API_KEY is not configured")

        if self._reader_task is not None and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        self._session_started.clear()
        uri = self._build_uri()
        self._ws = await websockets.connect(
            uri,
            additional_headers={"xi-api-key": config.ELEVENLABS_API_KEY},
            ping_interval=20,
            ping_timeout=10,
            max_size=8 * 1024 * 1024,
        )
        self._reader_task = asyncio.create_task(self._read_loop())
        try:
            await asyncio.wait_for(self._session_started.wait(), timeout=8.0)
        except asyncio.TimeoutError as exc:
            raise RuntimeError("ElevenLabs STT session_started timeout") from exc

        if self._reconnects:
            logger.info(
                "ElevenLabs STT reconnected conn=%s count=%s",
                self._connection_id,
                self._reconnects,
            )
        else:
            logger.info(
                "ElevenLabs STT session ready conn=%s model=%s",
                self._connection_id,
                config.STREAMING_STT_MODEL,
            )

    async def send_audio(self, pcm: bytes, *, commit: bool = False) -> None:
        if not pcm or self._closed:
            return
        await self.ensure_live()
        if self._ws is None:
            return

        payload = {
            "message_type": "input_audio_chunk",
            "audio_base_64": base64.b64encode(pcm).decode("ascii"),
            "commit": commit,
            "sample_rate": config.SAMPLE_RATE,
        }
        try:
            await self._ws.send(json.dumps(payload))
            self._chunks_sent += 1
            if self._chunks_sent in (1, 50, 200) or self._chunks_sent % 500 == 0:
                logger.info(
                    "STT audio forwarded conn=%s chunks=%s commit=%s",
                    self._connection_id,
                    self._chunks_sent,
                    commit,
                )
        except websockets.exceptions.ConnectionClosed as exc:
            logger.warning(
                "STT websocket send failed conn=%s: %s — reconnecting",
                self._connection_id,
                exc,
            )
            self._ws = None
            self._session_started.clear()
            self._reconnects += 1
            await self.ensure_live()
            if self._ws is not None:
                await self._ws.send(json.dumps(payload))

    async def events(self) -> AsyncIterator[STTEvent]:
        while True:
            event = await self._event_queue.get()
            if event is None:
                break
            yield event

    async def close(self) -> None:
        self._closed = True
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        await self._event_queue.put(None)

    async def _push_error(self, message: str) -> None:
        logger.error("STT error conn=%s: %s", self._connection_id, message)
        await self._event_queue.put(
            STTEvent(kind=STTEventKind.ERROR, error=message)
        )

    async def _read_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                await self._handle_message(data)
        except asyncio.CancelledError:
            raise
        except websockets.exceptions.ConnectionClosed:
            logger.warning(
                "ElevenLabs STT reader disconnected conn=%s closed=%s",
                self._connection_id,
                self._closed,
            )
        except Exception as exc:
            logger.error(
                "ElevenLabs streaming STT reader error for %s: %s",
                self._connection_id,
                exc,
            )
            await self._push_error(str(exc))
        finally:
            self._ws = None
            self._session_started.clear()
            if self._closed:
                await self._event_queue.put(None)

    async def _handle_message(self, data: dict) -> None:
        msg_type = data.get("message_type") or data.get("type") or ""

        if msg_type == "session_started":
            self._session_started.set()
            logger.info(
                "ElevenLabs STT session_started conn=%s session_id=%s",
                self._connection_id,
                data.get("session_id"),
            )
            return

        if msg_type == "partial_transcript":
            text = (data.get("text") or "").strip()
            if text:
                await self._event_queue.put(
                    STTEvent(kind=STTEventKind.PARTIAL, text=text)
                )
                logger.info(
                    "STT partial conn=%s text=%r",
                    self._connection_id,
                    text[:100],
                )
            return

        if msg_type in ("committed_transcript", "committed_transcript_with_timestamps"):
            text = (data.get("text") or "").strip()
            if not text:
                return
            language_code = data.get("language_code") or ""
            language_probability = data.get("language_probability")
            logger.info(
                "STT committed conn=%s text=%r",
                self._connection_id,
                text[:120],
            )
            await self._event_queue.put(
                STTEvent(
                    kind=STTEventKind.COMMITTED,
                    text=text,
                    language_code=language_code,
                    language_probability=language_probability,
                )
            )
            return

        if msg_type in ("error", "auth_error", "quota_exceeded", "rate_limited"):
            error = data.get("error") or data.get("message") or msg_type
            await self._push_error(str(error))
