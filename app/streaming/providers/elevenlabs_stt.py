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
        if not config.ELEVENLABS_API_KEY:
            raise RuntimeError("ELEVENLABS_API_KEY is not configured")

        uri = self._build_uri()
        self._ws = await websockets.connect(
            uri,
            additional_headers={"xi-api-key": config.ELEVENLABS_API_KEY},
            ping_interval=20,
            ping_timeout=10,
            max_size=8 * 1024 * 1024,
        )
        self._reader_task = asyncio.create_task(self._read_loop())
        logger.debug("ElevenLabs streaming STT connected for %s", self._connection_id)

    async def send_audio(self, pcm: bytes, *, commit: bool = False) -> None:
        if not pcm or self._ws is None or self._closed:
            return
        payload = {
            "message_type": "input_audio_chunk",
            "audio_base_64": base64.b64encode(pcm).decode("ascii"),
            "commit": commit,
            "sample_rate": config.SAMPLE_RATE,
        }
        try:
            await self._ws.send(json.dumps(payload))
        except websockets.exceptions.ConnectionClosed as exc:
            await self._push_error(f"STT websocket closed: {exc}")

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
            pass
        except Exception as exc:
            logger.error(
                "ElevenLabs streaming STT reader error for %s: %s",
                self._connection_id,
                exc,
            )
            await self._push_error(str(exc))
        finally:
            await self._event_queue.put(None)

    async def _handle_message(self, data: dict) -> None:
        msg_type = data.get("message_type") or data.get("type") or ""

        if msg_type == "session_started":
            logger.debug(
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
            return

        if msg_type in ("committed_transcript", "committed_transcript_with_timestamps"):
            text = (data.get("text") or "").strip()
            if not text:
                return
            language_code = data.get("language_code") or ""
            language_probability = data.get("language_probability")
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
