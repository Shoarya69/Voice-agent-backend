"""
Exotel session for Gemini Live.

8 kHz phone PCM ↔ 16 kHz in / 24 kHz out ↔ Gemini Native Audio.
"""

from __future__ import annotations

import audioop
import logging
from typing import ClassVar

from app import config
from app.audio_utils import decode_payload
from app.openai_service import openai_brain
from app.voice_pipeline.base import VoicePipelineSession
from app.voice_pipeline.gemini_live.link import GeminiLiveLink

logger = logging.getLogger(__name__)


class GeminiLivePipelineSession(VoicePipelineSession):
    """Gemini Live speech-to-speech. No ElevenLabs, no ChatGPT."""

    pipeline_name: ClassVar[str] = "gemini_live"

    def __init__(self, connection_id: str, websocket) -> None:
        super().__init__(connection_id, websocket)
        self.is_speaking = False
        self._is_playing_welcome = False
        self._in_state = None
        self._out_state = None
        self._out_residual = bytearray()
        self._link = GeminiLiveLink(
            connection_id,
            on_audio=self._on_model_audio,
            on_transcript=self._on_transcript,
            on_interrupt=self._on_interrupt,
        )

    async def on_agent_ready(self) -> None:
        await self._ensure_link()

    async def add_audio_chunk(self, media_data: dict) -> None:
        if self._closed or not self._link.ready:
            return
        if (self.is_speaking or self._is_playing_welcome) and not config.ENABLE_BARGE_IN:
            return
        pcm_8k = decode_payload(media_data.get("payload", ""))
        if not pcm_8k:
            return
        pcm_16k, self._in_state = audioop.ratecv(
            pcm_8k, 2, 1, 8000, 16000, self._in_state
        )
        await self._link.send_pcm16k(pcm_16k)

    async def speak_welcome(self) -> None:
        if not config.ENABLE_WELCOME_MESSAGE:
            return
        welcome = (config.WELCOME_MESSAGE or "").strip()
        if not welcome:
            return
        await self._ensure_link()
        self._is_playing_welcome = True
        try:
            await self._link.say(
                "The phone call just connected. Say this greeting in a natural voice, "
                f"then stop and wait: {welcome}"
            )
            await self._link.wait_turn(12)
        finally:
            self._is_playing_welcome = False
            self.is_speaking = False

    async def handle_clear(self) -> None:
        await self._on_interrupt()

    async def close(self) -> None:
        self._closed = True
        await self._link.close()

    async def _ensure_link(self) -> None:
        if self._link.ready:
            return
        instructions = openai_brain._messages([])[0]["content"]
        await self._link.connect(instructions)

    async def _on_model_audio(self, pcm_24k: bytes) -> None:
        if self._closed or not pcm_24k:
            return
        self.is_speaking = True
        pcm_8k, self._out_state = audioop.ratecv(
            pcm_24k, 2, 1, 24000, 8000, self._out_state
        )
        self._out_residual.extend(pcm_8k)
        frame = config.BYTES_PER_CHUNK
        while len(self._out_residual) >= frame:
            chunk = bytes(self._out_residual[:frame])
            del self._out_residual[:frame]
            if not await self.send_pcm_frame(chunk):
                return

    async def _on_transcript(self, role: str, text: str) -> None:
        cleaned = self._strip_call_end_marker(text) if role == "assistant" else text
        self._append_history_message(role, cleaned)
        logger.info("Gemini Live %s (%s): %s", role, self.connection_id, cleaned[:160])
        if role == "assistant":
            self._schedule_close_after_playback_if_needed()

    async def _on_interrupt(self) -> None:
        self._out_residual.clear()
        self._out_state = None
        self.is_speaking = False
        await self.send_exotel_clear()
