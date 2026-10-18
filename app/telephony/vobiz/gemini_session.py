"""
Vobiz call on Gemini Native Multimodal Live.

Phone μ-law 8 kHz ↔ Gemini Live audio. No ElevenLabs, no ChatGPT.
Playback uses the same paced playAudio path as the compound Vobiz session.
"""

from __future__ import annotations

import audioop
import logging
from typing import ClassVar

from app import config
from app.audio_utils import (
    decode_payload,
    pcm_bytes_to_ulaw,
    scale_pcm_volume,
    ulaw_bytes_to_pcm,
)
from app.openai_service import openai_brain
from app.telephony.vobiz.session import VobizCompoundSession
from app.voice_pipeline.gemini_live.link import GeminiLiveLink

logger = logging.getLogger(__name__)


class VobizGeminiLiveSession(VobizCompoundSession):
    """Gemini Live on a Vobiz media stream."""

    pipeline_name: ClassVar[str] = "gemini_live"

    def __init__(self, connection_id: str, websocket, vobiz) -> None:
        super().__init__(connection_id, websocket, vobiz)
        self._in_state = None
        self._out_state = None
        self._out_residual = bytearray()
        self._greeting_started = False
        self._link = GeminiLiveLink(
            connection_id,
            on_audio=self._on_model_audio,
            on_transcript=self._on_transcript,
            on_interrupt=self._on_interrupt,
            on_turn_complete=self._on_turn_complete,
        )

    async def preconnect_socket(self) -> None:
        """Open Gemini socket in parallel with Supabase agent lookup."""
        try:
            await self._link.open_socket()
        except Exception as exc:
            logger.debug("Gemini early socket open skipped conn=%s: %s", self.connection_id, exc)

    async def warm_playback_pipeline(self) -> None:
        """Connect Gemini and start synthesizing greeting BEFORE Vobiz `start` arrives."""
        try:
            await self._ensure_link()
            await self._start_greeting_if_needed()
        except Exception as exc:
            logger.warning("Gemini Live prewarm failed conn=%s: %s", self.connection_id, exc)

    async def on_agent_ready(self) -> None:
        await self._ensure_link()
        if self._ulaw_out_buffer and self._vobiz.stream_id:
            await self._drain_ulaw(force=False)

    async def add_audio_chunk(self, media_data: dict) -> None:
        if self._closed or not self._link.ready:
            return
        if (self.is_speaking or self._is_playing_welcome) and not config.ENABLE_BARGE_IN:
            return
        ulaw = decode_payload(media_data.get("payload", ""))
        if not ulaw:
            return
        pcm_8k = ulaw_bytes_to_pcm(ulaw)
        pcm_16k, self._in_state = audioop.ratecv(
            pcm_8k, 2, 1, 8000, 16000, self._in_state
        )
        await self._link.send_pcm16k(pcm_16k)

    async def _start_greeting_if_needed(self) -> None:
        if self._greeting_started:
            return
        welcome = self._welcome_text()
        if not welcome.strip():
            return
        self._greeting_started = True
        self.history.append({"role": "assistant", "content": welcome})
        self._is_playing_welcome = True
        self.is_speaking = True
        await self._link.say(
            "The phone call just connected. Say this greeting naturally, "
            f"then wait for the caller: {welcome}"
        )

    async def speak_welcome(self) -> None:
        if self._greeting_started:
            if self._ulaw_out_buffer and self._vobiz.stream_id:
                await self._drain_ulaw(force=False)
            return
        try:
            await self._ensure_link()
            await self._start_greeting_if_needed()
        except Exception as exc:
            logger.error(
                "Gemini Live connect failed conn=%s: %s",
                self.connection_id,
                exc,
            )

    async def handle_clear(self) -> None:
        await self._on_interrupt()

    async def close(self) -> None:
        await self._link.close()
        await super().close()

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
            if not self._vobiz.stream_id:
                # Hold pre-synthesized greeting until Vobiz sends `start` with streamId.
                self._ulaw_out_buffer.extend(
                    pcm_bytes_to_ulaw(scale_pcm_volume(chunk, config.TTS_VOLUME))
                )
                continue
            if not await self._send_paced_frame(chunk):
                return

    async def _on_transcript(self, role: str, text: str) -> None:
        cleaned = self._strip_call_end_marker(text) if role == "assistant" else text
        if role == "assistant" and self.history and self.history[-1]["role"] == "assistant":
            if self.history[-1]["content"] == cleaned:
                return
        self._append_history_message(role, cleaned)
        logger.info("Gemini Live %s (%s): %s", role, self.connection_id, cleaned[:160])

    async def _on_turn_complete(self) -> None:
        if self._out_residual:
            await self._send_paced_frame(bytes(self._out_residual))
            self._out_residual.clear()
        await self.flush_playback_output()
        self._is_playing_welcome = False
        self.is_speaking = False
        self._arm_echo_guard()

    async def _on_interrupt(self) -> None:
        self._out_residual.clear()
        self._out_state = None
        self._ulaw_out_buffer.clear()
        self._play_origin = None
        self._play_sent_ms = 0.0
        self._is_playing_welcome = False
        self.is_speaking = False
        await self._send_clear()
