"""Vobiz media session: μ-law I/O with the compound VoiceSession pipeline."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import ClassVar

from app import config
from app.audio_utils import (
    decode_payload,
    encode_payload,
    pcm_bytes_to_ulaw,
    scale_pcm_volume,
    ulaw_bytes_to_pcm,
)
from app.telephony.vobiz.stream import VobizStream
from app.voice_pipeline.compound.session import CompoundPipelineSession

logger = logging.getLogger(__name__)

_SENTENCE_END = re.compile(r"(?<=[.!?।])\s+")


class VobizCompoundSession(CompoundPipelineSession):
    """ElevenLabs + OpenAI pipeline with Vobiz playAudio outbound and μ-law inbound."""

    pipeline_name: ClassVar[str] = "compound_vobiz"

    def __init__(self, connection_id: str, websocket, vobiz: VobizStream) -> None:
        super().__init__(connection_id, websocket)
        self.telephony_provider = "vobiz"
        self._vobiz = vobiz
        self._ulaw_out_buffer = bytearray()
        self._playback_turn = 0
        self._finish_playback_task: asyncio.Task | None = None

    async def add_audio_chunk(self, media_data: dict) -> None:
        payload = media_data.get("payload", "")
        ulaw = decode_payload(payload)
        if not ulaw:
            return
        pcm = ulaw_bytes_to_pcm(ulaw)
        await super().add_audio_chunk({"payload": encode_payload(pcm)})

    async def send_pcm_frame(self, pcm_frame: bytes) -> bool:
        if self._closed or self._vobiz.closed:
            return False
        pcm_frame = scale_pcm_volume(pcm_frame, config.TTS_VOLUME)
        self._ulaw_out_buffer.extend(pcm_bytes_to_ulaw(pcm_frame))
        chunk = max(160, config.VOBIZ_ULAW_PLAY_CHUNK)
        while len(self._ulaw_out_buffer) >= chunk:
            block = bytes(self._ulaw_out_buffer[:chunk])
            del self._ulaw_out_buffer[:chunk]
            await self._vobiz.play_ulaw(block)
        return True

    async def flush_playback_output(self) -> None:
        await self._flush_ulaw_out_buffer()

    async def _flush_ulaw_out_buffer(self) -> None:
        if self._ulaw_out_buffer:
            await self._vobiz.play_ulaw(bytes(self._ulaw_out_buffer))
            self._ulaw_out_buffer.clear()

    async def _send_pcm_frame(self, pcm_frame: bytes) -> bool:
        return await self.send_pcm_frame(pcm_frame)

    async def _send_paced_frame(self, frame, timeline=None, playback_started=False):
        """Vobiz playAudio is paced by the carrier — avoid Exotel 20 ms clock gaps."""
        if not config.VOBIZ_SKIP_PLAYBACK_PACING:
            return await super()._send_paced_frame(frame, timeline, playback_started)

        if self._closed or not self._websocket_is_open():
            return False
        send_started = time.perf_counter()
        if not await self._send_pcm_frame(frame):
            return False
        send_latency_ms = (time.perf_counter() - send_started) * 1000

        if config.ENABLE_LATENCY_TRACE:
            from app.latency_trace import active_trace

            trace = active_trace()
            if trace is not None:
                trace.record_exotel_send(pace_delay_ms=0.0, send_latency_ms=send_latency_ms)
        if timeline is not None:
            timeline.mark_once("first_packet_sent_ms")
            timeline.mark("last_packet_sent_ms")
        return True

    def _arm_echo_guard(self) -> None:
        super()._arm_echo_guard()
        if self._finish_playback_task is not None and not self._finish_playback_task.done():
            self._finish_playback_task.cancel()
        self._finish_playback_task = asyncio.create_task(self._debounced_finish_playback())

    async def _debounced_finish_playback(self) -> None:
        try:
            await asyncio.sleep(0.08)
            await self._vobiz_finish_playback()
        except asyncio.CancelledError:
            raise

    async def _vobiz_finish_playback(self) -> None:
        if self._vobiz.closed:
            return
        try:
            await self._flush_ulaw_out_buffer()
            self._playback_turn += 1
            await self._vobiz.checkpoint(f"utt-{self._playback_turn}")
        except Exception as exc:
            logger.debug(
                "Vobiz playback finish skipped for %s: %s",
                self.connection_id,
                exc,
            )

    async def _send_clear(self) -> None:
        self._ulaw_out_buffer.clear()
        await self._vobiz.clear()

    async def handle_clear(self) -> None:
        await super().handle_clear()

    @staticmethod
    def _first_sentence(text: str) -> str:
        text = (text or "").strip()
        if not text:
            return text
        parts = _SENTENCE_END.split(text, maxsplit=1)
        first = parts[0].strip()
        if first and first[-1] not in ".!?।":
            first = first + "."
        return first or text

    def _welcome_text(self) -> str:
        text = (config.WELCOME_MESSAGE or "").strip()
        if not text:
            agent = getattr(self, "agent_config", None)
            if agent and (agent.first_message or "").strip():
                text = agent.first_message.strip()
            else:
                text = "Namaste, main aapki madad ke liye hoon."
        if config.VOBIZ_GREETING_FIRST_SENTENCE:
            text = self._first_sentence(text)
        words = text.split()
        if len(words) > config.MAX_WELCOME_WORDS:
            text = " ".join(words[: config.MAX_WELCOME_WORDS]).strip()
        return text

    async def warm_playback_pipeline(self) -> None:
        """Pre-open ElevenLabs TTS while waiting for Vobiz ``start`` (saves ~300–800 ms)."""
        if not self._streaming.enabled:
            return
        started = time.perf_counter()
        try:
            await self._streaming.ensure_tts_for_greeting()
            logger.info(
                "Vobiz TTS pre-warmed conn=%s ms=%.0f",
                self.connection_id,
                (time.perf_counter() - started) * 1000,
            )
        except Exception as exc:
            logger.warning(
                "Vobiz TTS pre-warm failed conn=%s: %s",
                self.connection_id,
                exc,
            )

    async def on_agent_ready(self) -> None:
        """TTS first; STT in background so greeting audio starts within ~1 s."""
        if not self._streaming.enabled:
            return
        try:
            await self._streaming.ensure_tts_for_greeting()
        except Exception as exc:
            logger.error(
                "Vobiz TTS connect failed for %s: %s",
                self.connection_id,
                exc,
            )
        asyncio.create_task(self._streaming.ensure_stt_if_needed())

    async def speak_welcome(self) -> None:
        """
        Vobiz often drops the call if no playAudio within a few seconds.
        Must await greeting TTS/PCM before handling more events.
        """
        if not config.ENABLE_WELCOME_MESSAGE:
            await self._speak_minimal_vobiz_greeting(self._welcome_text())
            return

        welcome = self._welcome_text()
        if not welcome.strip():
            await self._speak_minimal_vobiz_greeting(
                "Namaste, main aapki madad ke liye hoon."
            )
            return

        self.history.append({"role": "assistant", "content": welcome})
        t0 = time.perf_counter()
        self._set_response_task(self._speak_welcome())

        task = self._response_task
        if task is None:
            await self._speak_minimal_vobiz_greeting(welcome)
            return
        try:
            await task
            logger.info(
                "Vobiz greeting done conn=%s ms=%.0f words=%s",
                self.connection_id,
                (time.perf_counter() - t0) * 1000,
                len(welcome.split()),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                "Vobiz greeting task failed conn=%s: %s",
                self.connection_id,
                exc,
            )
            await self._speak_minimal_vobiz_greeting(welcome)
        finally:
            await self.flush_playback_output()

    async def _speak_minimal_vobiz_greeting(self, text: str) -> None:
        """Last-resort audio so Vobiz does not tear down a silent stream."""
        self._is_playing_welcome = True
        try:
            if self._streaming.enabled:
                await self._streaming.ensure_tts_for_greeting()
                await self._streaming.speak_text(self._first_sentence(text)[:200])
            else:
                sentence_queue: asyncio.Queue = asyncio.Queue()
                await sentence_queue.put(text[:200])
                await sentence_queue.put(None)
                await self._speak_from_sentence_queue(sentence_queue)
        finally:
            self._is_playing_welcome = False
            await self.flush_playback_output()
            self._arm_echo_guard()

    async def close(self) -> None:
        if self._finish_playback_task is not None:
            self._finish_playback_task.cancel()
        self._ulaw_out_buffer.clear()
        self._vobiz.closed = True
        await super().close()
