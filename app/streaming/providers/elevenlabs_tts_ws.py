"""
ElevenLabs WebSocket TTS (stream-input) with a persistent per-call connection.

API: wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input
Docs: https://elevenlabs.io/docs/eleven-api/guides/how-to/websockets/realtime-tts
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from typing import AsyncIterator
from urllib.parse import urlencode

import websockets
import websockets.exceptions

from app import config
from app.streaming.providers.base import StreamingTTSProvider

logger = logging.getLogger(__name__)


class ElevenLabsWebSocketTTSProvider(StreamingTTSProvider):
    def __init__(self, connection_id: str) -> None:
        self._connection_id = connection_id
        self._ws = None
        self._audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._reader_task: asyncio.Task | None = None
        self._keepalive_task: asyncio.Task | None = None
        self._connected = False
        self._closed = False
        self._reconnects = 0
        self._utterance_open = False
        self._utterance_done = asyncio.Event()
        self._utterance_audio: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._flush_sent = False

    def _build_uri(self) -> str:
        base = config.ELEVENLABS_BASE_URL.rstrip("/").replace("https://", "wss://")
        params = {
            "model_id": config.ELEVENLABS_TTS_MODEL,
            "output_format": f"pcm_{config.SAMPLE_RATE}",
            "inactivity_timeout": config.STREAMING_TTS_WS_INACTIVITY_TIMEOUT,
        }
        return (
            f"{base}/v1/text-to-speech/{config.ELEVENLABS_VOICE_ID}/stream-input?"
            f"{urlencode(params)}"
        )

    async def connect(self) -> None:
        await self._open_socket()

    async def _open_socket(self) -> None:
        if not config.ELEVENLABS_API_KEY:
            raise RuntimeError("ELEVENLABS_API_KEY is not configured")

        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

        uri = self._build_uri()
        self._ws = await websockets.connect(
            uri,
            additional_headers={"xi-api-key": config.ELEVENLABS_API_KEY},
            ping_interval=20,
            ping_timeout=10,
            max_size=16 * 1024 * 1024,
        )
        await self._ws.send(
            json.dumps(
                {
                    "text": " ",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75,
                        "style": 0.0,
                        "use_speaker_boost": True,
                    },
                    "generation_config": {
                        "chunk_length_schedule": config.STREAMING_TTS_CHUNK_SCHEDULE,
                    },
                }
            )
        )
        self._connected = True
        self._utterance_open = False
        self._reader_task = asyncio.create_task(self._read_loop())
        if self._keepalive_task is None or self._keepalive_task.done():
            self._keepalive_task = asyncio.create_task(self._keepalive_loop())
        if self._reconnects:
            logger.info(
                "ElevenLabs TTS websocket reconnected conn=%s count=%s",
                self._connection_id,
                self._reconnects,
            )

    async def begin_utterance(self) -> None:
        if self._closed:
            return
        while not self._utterance_audio.empty():
            try:
                self._utterance_audio.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._utterance_done.clear()
        self._flush_sent = False
        if not self._connected or self._ws is None:
            await self._open_socket()
            self._reconnects += 1
        self._utterance_open = True

    async def send_text(self, text: str) -> None:
        chunk = (text or "").strip()
        if not chunk or self._closed:
            return
        if not self._utterance_open:
            await self.begin_utterance()
        if not chunk.endswith(" "):
            chunk = f"{chunk} "
        started = time.perf_counter()
        label = chunk.strip()[:40]
        try:
            assert self._ws is not None
            await self._ws.send(json.dumps({"text": chunk}))
        except websockets.exceptions.ConnectionClosed:
            await self._recover_and_send(chunk)
            started = time.perf_counter()
        finally:
            from app.latency_trace import active_trace

            trace = active_trace()
            if trace is not None:
                trace.record_tts_ws_send(label, (time.perf_counter() - started) * 1000)

    async def flush_utterance(self) -> None:
        if self._closed or not self._utterance_open or self._ws is None:
            self._utterance_open = False
            self._utterance_done.set()
            return
        self._utterance_open = False
        self._flush_sent = True
        try:
            await self._ws.send(json.dumps({"text": ""}))
        except websockets.exceptions.ConnectionClosed:
            self._connected = False
        try:
            await asyncio.wait_for(self._utterance_done.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning(
                "TTS utterance flush timed out conn=%s",
                self._connection_id,
            )
        finally:
            from app.latency_trace import active_trace

            trace = active_trace()
            if trace is not None:
                trace.mark_tts_flush()
            self._flush_sent = False

    async def iter_utterance_audio(self) -> AsyncIterator[bytes]:
        from app.latency_trace import active_trace

        while True:
            wait_started = time.perf_counter()
            try:
                chunk = await asyncio.wait_for(self._utterance_audio.get(), timeout=0.25)
            except asyncio.TimeoutError:
                wait_ms = (time.perf_counter() - wait_started) * 1000
                trace = active_trace()
                if trace is not None:
                    trace.record_pcm_underflow(wait_ms)
                if self._utterance_done.is_set() and self._utterance_audio.empty():
                    break
                continue
            if chunk is None:
                break
            trace = active_trace()
            if trace is not None:
                trace.record_tts_chunk(len(chunk))
            yield chunk

    async def audio_chunks(self) -> AsyncIterator[bytes]:
        async for chunk in self.iter_utterance_audio():
            yield chunk

    async def close(self) -> None:
        self._closed = True
        if self._keepalive_task is not None:
            self._keepalive_task.cancel()
        if self._reader_task is not None:
            self._reader_task.cancel()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        await self._utterance_audio.put(None)

    @property
    def reconnect_count(self) -> int:
        return self._reconnects

    async def _recover_and_send(self, text: str) -> None:
        self._connected = False
        self._reconnects += 1
        await self._open_socket()
        self._utterance_open = True
        assert self._ws is not None
        await self._ws.send(json.dumps({"text": text}))

    async def _keepalive_loop(self) -> None:
        try:
            while not self._closed:
                await asyncio.sleep(15)
                if self._closed or self._ws is None or self._utterance_open:
                    continue
                try:
                    await self._ws.send(json.dumps({"text": " "}))
                except websockets.exceptions.ConnectionClosed:
                    self._connected = False
        except asyncio.CancelledError:
            pass

    async def _read_loop(self) -> None:
        assert self._ws is not None
        ws = self._ws
        try:
            async for raw in ws:
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                audio_b64 = data.get("audio")
                if audio_b64:
                    pcm = base64.b64decode(audio_b64)
                    await self._utterance_audio.put(pcm)
                if data.get("isFinal"):
                    from app.latency_trace import active_trace

                    if self._flush_sent:
                        self._utterance_done.set()
                    else:
                        trace = active_trace()
                        if trace is not None:
                            trace.record_tts_premature_is_final()
                        logger.debug(
                            "TTS isFinal between clauses (ignored) conn=%s",
                            self._connection_id,
                        )
        except asyncio.CancelledError:
            raise
        except websockets.exceptions.ConnectionClosed:
            self._connected = False
        except Exception as exc:
            logger.error(
                "ElevenLabs TTS websocket reader error conn=%s: %s",
                self._connection_id,
                exc,
            )
