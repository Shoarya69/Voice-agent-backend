"""
OpenAI Realtime pipeline — speech-to-speech via the Realtime WebSocket API.

Exotel (8 kHz PCM) ↔ resample ↔ OpenAI Realtime (24 kHz PCM) ↔ Exotel

Agent config, call history, and CRM fields use the shared ``VoicePipelineSession``
base — same as the compound pipeline.
"""

from __future__ import annotations

import asyncio
import audioop
import base64
import json
import logging
from typing import ClassVar

import websockets
import websockets.exceptions

from app import config
from app.audio_utils import chunk_pcm, decode_payload
from app.openai_service import openai_brain
from app.voice_pipeline.base import VoicePipelineSession

logger = logging.getLogger(__name__)

# OpenAI Realtime expects 24 kHz PCM16 mono for pcm16 format.
_REALTIME_INPUT_RATE = 24_000
_REALTIME_OUTPUT_RATE = 24_000


class OpenAIRealtimePipelineSession(VoicePipelineSession):
    """OpenAI Realtime API provider (single-model speech-to-speech)."""

    pipeline_name: ClassVar[str] = "openai_realtime"

    def __init__(self, connection_id: str, websocket) -> None:
        super().__init__(connection_id, websocket)
        self._realtime_ws = None
        self._reader_task: asyncio.Task | None = None
        self._response_task: asyncio.Task | None = None
        self._realtime_connected = False
        self._realtime_connecting = False
        self._playback_task: asyncio.Task | None = None
        self.is_speaking = False
        self._is_playing_welcome = False
        self._resample_state_in = None
        self._resample_state_out = None
        self._assistant_transcript_buffer = ""
        self._output_frame_queue: asyncio.Queue = asyncio.Queue()
        # Greeting de-duplication: PCM path records history once + injects Realtime item;
        # live response.create path waits for response.done transcript only.
        self._awaiting_greeting_transcript = False
        self._greeting_text_recorded: str | None = None

    # ------------------------------------------------------------------
    # Exotel event handlers (VoicePipelineSession)
    # ------------------------------------------------------------------
    async def add_audio_chunk(self, media_data: dict) -> None:
        if self._closed or not self._realtime_connected:
            return
        if self.is_speaking or self._is_playing_welcome:
            if not config.ENABLE_BARGE_IN:
                return

        payload = media_data.get("payload", "")
        pcm_8k = decode_payload(payload)
        if not pcm_8k:
            return

        pcm_24k, self._resample_state_in = audioop.ratecv(
            pcm_8k,
            config.SAMPLE_WIDTH,
            config.CHANNELS,
            config.SAMPLE_RATE,
            _REALTIME_INPUT_RATE,
            self._resample_state_in,
        )
        audio_b64 = base64.b64encode(pcm_24k).decode("ascii")
        await self._send_realtime(
            {
                "type": "input_audio_buffer.append",
                "audio": audio_b64,
            }
        )

    async def speak_welcome(self) -> None:
        if not config.ENABLE_WELCOME_MESSAGE or not config.WELCOME_MESSAGE:
            return

        welcome = config.WELCOME_MESSAGE.strip()
        greeting_pcm = getattr(self, "greeting_audio_pcm", None)

        if greeting_pcm:
            self._greeting_text_recorded = welcome
            self._append_history_message("assistant", welcome)
            self._response_task = asyncio.create_task(
                self._play_pcm_greeting(greeting_pcm, welcome)
            )
            return

        await self._ensure_realtime_connected()
        self._awaiting_greeting_transcript = True
        await self._send_realtime(
            {
                "type": "response.create",
                "response": {
                    "modalities": ["text", "audio"],
                    "instructions": f"Say exactly this greeting, nothing else: {welcome}",
                },
            }
        )

    async def handle_clear(self) -> None:
        await self.send_exotel_clear()
        if self._playback_task and not self._playback_task.done():
            self._playback_task.cancel()
        await self._send_realtime({"type": "response.cancel"})
        self.is_speaking = False

    async def close(self) -> None:
        self._closed = True
        prefetch = getattr(self, "_agent_prefetch_task", None)
        if prefetch is not None and not prefetch.done():
            prefetch.cancel()
        for task in (self._reader_task, self._playback_task, self._response_task):
            if task is not None and not task.done():
                task.cancel()
        if self._realtime_ws is not None:
            try:
                await self._realtime_ws.close()
            except Exception:
                pass
            self._realtime_ws = None

    # ------------------------------------------------------------------
    # Realtime connection lifecycle
    # ------------------------------------------------------------------
    async def _ensure_realtime_connected(self) -> None:
        if self._realtime_connected or self._realtime_connecting:
            if self._realtime_connecting:
                for _ in range(50):
                    if self._realtime_connected:
                        return
                    await asyncio.sleep(0.05)
            return

        if not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not configured")

        self._realtime_connecting = True
        url = (
            f"{config.OPENAI_REALTIME_BASE_URL.rstrip('/')}"
            f"/realtime?model={config.OPENAI_REALTIME_MODEL}"
        )
        headers = {
            "Authorization": f"Bearer {config.OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1",
        }

        try:
            self._realtime_ws = await websockets.connect(
                url,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=10,
            )
            self._reader_task = asyncio.create_task(self._read_realtime_events())
            await self._configure_realtime_session()
            self._realtime_connected = True
            logger.info(
                "OpenAI Realtime connected for %s model=%s",
                self.connection_id,
                config.OPENAI_REALTIME_MODEL,
            )
        finally:
            self._realtime_connecting = False

    async def _configure_realtime_session(self) -> None:
        instructions = openai_brain._messages(self.history)[0]["content"]

        await self._send_realtime(
            {
                "type": "session.update",
                "session": {
                    "modalities": ["text", "audio"],
                    "instructions": instructions,
                    "voice": config.OPENAI_REALTIME_VOICE,
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "input_audio_transcription": {"model": "whisper-1"},
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": config.OPENAI_REALTIME_VAD_THRESHOLD,
                        "prefix_padding_ms": config.OPENAI_REALTIME_PREFIX_PADDING_MS,
                        "silence_duration_ms": config.OPENAI_REALTIME_SILENCE_DURATION_MS,
                    },
                    "temperature": config.OPENAI_TEMPERATURE,
                    "max_response_output_tokens": config.OPENAI_MAX_TOKENS,
                },
            }
        )

    async def _inject_assistant_message_into_realtime(self, text: str) -> None:
        """
        Tell the Realtime model about assistant speech it did not generate
        (e.g. pre-recorded PCM greeting played directly to Exotel).
        """
        if not self._realtime_connected or not text.strip():
            return
        await self._send_realtime(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": text.strip()}],
                },
            }
        )
        logger.info(
            "Injected assistant greeting into Realtime conversation for %s",
            self.connection_id,
        )

    async def _send_realtime(self, event: dict) -> None:
        if self._realtime_ws is None:
            return
        try:
            await self._realtime_ws.send(json.dumps(event))
        except websockets.exceptions.ConnectionClosed as exc:
            logger.warning(
                "OpenAI Realtime connection closed for %s: %s",
                self.connection_id,
                exc,
            )
            self._realtime_connected = False

    async def _read_realtime_events(self) -> None:
        assert self._realtime_ws is not None
        try:
            async for raw in self._realtime_ws:
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                await self._handle_realtime_event(event)
        except asyncio.CancelledError:
            raise
        except websockets.exceptions.ConnectionClosed:
            logger.info("OpenAI Realtime reader ended for %s", self.connection_id)
        except Exception as exc:
            logger.error(
                "OpenAI Realtime reader error for %s: %s",
                self.connection_id,
                exc,
            )

    def _is_duplicate_greeting_transcript(self, text: str) -> bool:
        recorded = (self._greeting_text_recorded or "").strip()
        normalized = text.strip()
        if not recorded or not normalized:
            return False
        if normalized == recorded:
            return True
        return recorded in normalized or normalized in recorded

    async def _record_assistant_transcript(self, raw_text: str) -> None:
        text = self._strip_call_end_marker(raw_text)
        if not text:
            self._schedule_close_after_playback_if_needed()
            return

        if self._awaiting_greeting_transcript:
            self._awaiting_greeting_transcript = False
            self._greeting_text_recorded = text
            self._append_history_message("assistant", text)
            logger.info("🧠 Assistant (%s) reply: %s", self.connection_id, text)
            return

        if self._is_duplicate_greeting_transcript(text):
            logger.debug(
                "Skipping duplicate greeting transcript for %s",
                self.connection_id,
            )
            return

        self._append_history_message("assistant", text)
        logger.info("🧠 Assistant (%s) reply: %s", self.connection_id, text)

        if self._should_end_call:
            logger.info(
                "☎️ Closing call for %s because LLM requested end-call",
                self.connection_id,
            )

    async def _handle_realtime_event(self, event: dict) -> None:
        event_type = event.get("type", "")

        if event_type == "response.audio.delta":
            delta_b64 = event.get("delta") or ""
            if not delta_b64:
                return
            pcm_24k = base64.b64decode(delta_b64)
            await self._enqueue_output_pcm(pcm_24k, sample_rate=_REALTIME_OUTPUT_RATE)

        elif event_type == "response.audio_transcript.delta":
            self._assistant_transcript_buffer += event.get("delta") or ""

        elif event_type == "response.done":
            text = self._assistant_transcript_buffer.strip()
            self._assistant_transcript_buffer = ""
            if text:
                await self._record_assistant_transcript(text)

        elif event_type == "conversation.item.input_audio_transcription.completed":
            text = (event.get("transcript") or "").strip()
            if text:
                self._append_history_message("user", text)
                logger.info("ACCEPTED - stt %s text=%r", self.connection_id, text)

        elif event_type == "error":
            logger.error(
                "OpenAI Realtime error for %s: %s",
                self.connection_id,
                event.get("error"),
            )

    async def _enqueue_output_pcm(self, pcm: bytes, *, sample_rate: int) -> None:
        pcm_8k, self._resample_state_out = audioop.ratecv(
            pcm,
            config.SAMPLE_WIDTH,
            config.CHANNELS,
            sample_rate,
            config.SAMPLE_RATE,
            self._resample_state_out,
        )
        if self._playback_task is None or self._playback_task.done():
            self._playback_task = asyncio.create_task(self._playback_worker())
        for frame in chunk_pcm(pcm_8k):
            await self._output_frame_queue.put(frame)

    async def _playback_worker(self) -> None:
        loop = asyncio.get_running_loop()
        next_send_time: float | None = None
        self.is_speaking = True
        try:
            while not self._closed:
                try:
                    frame = await asyncio.wait_for(
                        self._output_frame_queue.get(), timeout=0.15
                    )
                except asyncio.TimeoutError:
                    if self._output_frame_queue.empty():
                        break
                    continue
                now = loop.time()
                if next_send_time is None:
                    next_send_time = now
                delay = next_send_time - now
                if delay > 0:
                    await asyncio.sleep(delay)
                if not await self.send_pcm_frame(frame):
                    break
                next_send_time += config.CHUNK_DURATION_MS / 1000
        finally:
            self.is_speaking = False
            self._schedule_close_after_playback_if_needed()

    async def _play_pcm_greeting(self, greeting_pcm: bytes, welcome_text: str) -> None:
        self._is_playing_welcome = True
        try:
            await self._ensure_realtime_connected()
            await self._inject_assistant_message_into_realtime(welcome_text)

            frame_queue: asyncio.Queue = asyncio.Queue()
            for frame in chunk_pcm(greeting_pcm):
                await frame_queue.put(frame)
            await frame_queue.put(None)
            loop = asyncio.get_running_loop()
            next_send_time: float | None = None
            self.is_speaking = True
            while True:
                frame = await frame_queue.get()
                if frame is None:
                    break
                now = loop.time()
                if next_send_time is None:
                    next_send_time = now
                delay = next_send_time - now
                if delay > 0:
                    await asyncio.sleep(delay)
                if not await self.send_pcm_frame(frame):
                    break
                next_send_time += config.CHUNK_DURATION_MS / 1000
        except asyncio.CancelledError:
            raise
        except websockets.exceptions.ConnectionClosed:
            logger.info(
                "Greeting stopped — WebSocket closed for %s",
                self.connection_id,
            )
        finally:
            self.is_speaking = False
            self._is_playing_welcome = False

    async def on_agent_ready(self) -> None:
        await self._ensure_realtime_connected()
