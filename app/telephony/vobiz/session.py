"""Vobiz media session: μ-law I/O with the compound VoiceSession pipeline."""

from __future__ import annotations

import asyncio
import logging
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


class VobizCompoundSession(CompoundPipelineSession):
    """ElevenLabs + OpenAI pipeline with Vobiz playAudio outbound and μ-law inbound."""

    pipeline_name: ClassVar[str] = "compound_vobiz"

    def __init__(self, connection_id: str, websocket, vobiz: VobizStream) -> None:
        super().__init__(connection_id, websocket)
        self.telephony_provider = "vobiz"
        self._vobiz = vobiz
        self._ulaw_out_buffer = bytearray()
        self._playback_turn = 0

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

    async def _flush_ulaw_out_buffer(self) -> None:
        if self._ulaw_out_buffer:
            await self._vobiz.play_ulaw(bytes(self._ulaw_out_buffer))
            self._ulaw_out_buffer.clear()

    async def _send_pcm_frame(self, pcm_frame: bytes) -> bool:
        return await self.send_pcm_frame(pcm_frame)

    def _arm_echo_guard(self) -> None:
        super()._arm_echo_guard()
        asyncio.create_task(self._vobiz_finish_playback())

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

    def _welcome_text(self) -> str:
        text = (config.WELCOME_MESSAGE or "").strip()
        if text:
            words = text.split()
            if len(words) <= config.MAX_WELCOME_WORDS:
                return text
            return " ".join(words[: config.MAX_WELCOME_WORDS]).strip()
        agent = getattr(self, "agent_config", None)
        if agent and (agent.first_message or "").strip():
            return agent.first_message.strip()
        return "Namaste, main aapki madad ke liye hoon."

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
        self._set_response_task(self._speak_welcome())

        task = self._response_task
        if task is None:
            await self._speak_minimal_vobiz_greeting(welcome)
            return
        try:
            await task
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                "Vobiz greeting task failed conn=%s: %s",
                self.connection_id,
                exc,
            )
            await self._speak_minimal_vobiz_greeting(welcome)

    async def _speak_minimal_vobiz_greeting(self, text: str) -> None:
        """Last-resort audio so Vobiz does not tear down an silent stream."""
        self._is_playing_welcome = True
        try:
            if self._streaming.enabled:
                await self._streaming.ensure_connected()
                await self._streaming.speak_text(text[:200])
            else:
                sentence_queue: asyncio.Queue = asyncio.Queue()
                await sentence_queue.put(text[:200])
                await sentence_queue.put(None)
                await self._speak_from_sentence_queue(sentence_queue)
        finally:
            self._is_playing_welcome = False
            self._arm_echo_guard()

    async def close(self) -> None:
        self._ulaw_out_buffer.clear()
        self._vobiz.closed = True
        await super().close()
