"""
Vobiz call on Gemini Native Multimodal Live.

Phone μ-law 8 kHz ↔ Gemini Live audio. No ElevenLabs, no ChatGPT.
Playback uses the same paced playAudio path as the compound Vobiz session.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, ClassVar

try:
    import audioop
except ImportError:  # Python 3.13+
    audioop = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from app.voice_pipeline.gemini_live.prewarm import PreparedGeminiSession

from app import config
from app.audio_utils import (
    decode_payload,
    pcm_bytes_to_ulaw,
    scale_pcm_volume,
    ulaw_bytes_to_pcm,
)
from app.openai_service import openai_brain
from app.telephony.vobiz.session import VobizCompoundSession
from app.voice_pipeline.gemini_live.call_end import (
    assistant_sounds_like_closing,
    user_requested_hangup,
)
from app.voice_pipeline.gemini_live.link import GeminiLiveLink
from app.voice_pipeline.tools import gemini_live_system_instructions

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
        self._prewarm_greeting_done = False
        self._prewarm_call_id: str | None = None
        self._ws_connected_at = time.monotonic()
        self._first_audio_logged = False
        self._caller_wants_hangup = False
        self._last_model_audio_at = 0.0
        self._user_turns = 0
        self._link = GeminiLiveLink(
            connection_id,
            on_audio=self._on_model_audio,
            on_transcript=self._on_transcript,
            on_interrupt=self._on_interrupt,
            on_turn_complete=self._on_turn_complete,
            on_end_call=self._on_end_call_tool,
        )

    async def attach_prepared_session(self, entry: "PreparedGeminiSession") -> bool:
        """Attach a pre-warmed Gemini Live session and pre-rendered greeting."""
        self._prewarm_call_id = entry.call_id
        if not entry.ready_event.is_set():
            try:
                await asyncio.wait_for(entry.ready_event.wait(), timeout=2.5)
            except asyncio.TimeoutError:
                pass
        if entry.agent_config is None or entry.link is None or not entry.link.ready:
            return False

        self.agent_id = entry.agent_config.agent_id
        self.agent_config = entry.agent_config
        self.greeting_audio_pcm = entry.agent_config.greeting_audio_pcm
        self.lovable_token = entry.voicebot_token
        self.call_direction = entry.direction or "outbound"

        if entry.greeting_started and not entry.first_audio_event.is_set():
            try:
                await asyncio.wait_for(entry.first_audio_event.wait(), timeout=0.25)
            except asyncio.TimeoutError:
                pass

        old_link = self._link
        self._link = entry.link
        self._out_state = entry.out_state
        self._out_residual = bytearray(entry.out_residual)
        if entry.ulaw_greeting:
            self._ulaw_out_buffer.extend(entry.ulaw_greeting)
        self._greeting_started = entry.greeting_started
        self._prewarm_greeting_done = entry.greeting_done
        if entry.welcome_text:
            self.history.append({"role": "assistant", "content": entry.welcome_text})

        self._link.rebind_handlers(
            on_audio=self._on_model_audio,
            on_transcript=self._on_transcript,
            on_interrupt=self._on_interrupt,
            on_turn_complete=self._on_turn_complete,
            on_end_call=self._on_end_call_tool,
        )
        if not entry.greeting_done and entry.greeting_started:
            self._is_playing_welcome = True
            self.is_speaking = True
            self._last_model_audio_at = time.monotonic()

        asyncio.create_task(old_link.close())
        return True

    async def _finish_prewarm_greeting_playback(self) -> None:
        self._is_playing_welcome = True
        self.is_speaking = True
        self._last_model_audio_at = time.monotonic()
        try:
            await self.flush_playback_output()
        finally:
            self._is_playing_welcome = False
            self.is_speaking = False
            self._arm_echo_guard()

    def _log_prewarm_first_audio_if_needed(self) -> None:
        if self._first_audio_logged or not self._prewarm_call_id:
            return
        self._first_audio_logged = True
        logger.info(
            "PREWARM event=first_audio_sent call_id=%s pickup_to_first_audio_ms=%.0f",
            self._prewarm_call_id,
            (time.monotonic() - self._ws_connected_at) * 1000,
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
            self._log_prewarm_first_audio_if_needed()

    async def add_audio_chunk(self, media_data: dict) -> None:
        if self._closed or not self._link.ready:
            return
        # Never stay deaf if Gemini drops turnComplete — release after 0.6s of no model audio.
        if (self.is_speaking or self._is_playing_welcome) and self._last_model_audio_at > 0:
            if (time.monotonic() - self._last_model_audio_at) > 0.6:
                self.is_speaking = False
                self._is_playing_welcome = False
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
        self._last_model_audio_at = time.monotonic()
        await self._link.say(
            "The phone call just connected. Speak ONLY this opening greeting "
            f"and do NOT call any tools: {welcome}"
        )

    async def speak_welcome(self) -> None:
        if self._greeting_started:
            if self._ulaw_out_buffer and self._vobiz.stream_id:
                await self._drain_ulaw(force=False)
                self._log_prewarm_first_audio_if_needed()
                if self._prewarm_greeting_done:
                    asyncio.create_task(self._finish_prewarm_greeting_playback())
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
        base = openai_brain._messages([])[0]["content"]
        await self._link.connect(gemini_live_system_instructions(base))

    async def _on_model_audio(self, pcm_24k: bytes) -> None:
        if self._closed or not pcm_24k:
            return
        self.is_speaking = True
        self._last_model_audio_at = time.monotonic()
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
            self._log_prewarm_first_audio_if_needed()

    async def _on_end_call_tool(self) -> None:
        # Ignore premature end_call during opening greeting before caller has spoken.
        if self._user_turns == 0 and not self._caller_wants_hangup:
            logger.warning(
                "Ignoring premature Gemini end_call on greeting conn=%s",
                self.connection_id,
            )
            return
        self._should_end_call = True
        logger.info("Gemini Live end_call tool conn=%s", self.connection_id)

    async def _on_transcript(self, role: str, text: str) -> None:
        if role == "user":
            self._user_turns += 1
            if user_requested_hangup(text):
                self._caller_wants_hangup = True
        cleaned = self._strip_call_end_marker(text) if role == "assistant" else text
        if role == "assistant" and self.history and self.history[-1]["role"] == "assistant":
            if self.history[-1]["content"] == cleaned:
                return
        self._append_history_message(role, cleaned)
        logger.info("Gemini Live %s (%s): %s", role, self.connection_id, cleaned[:160])
        if role == "assistant" and self._caller_wants_hangup and assistant_sounds_like_closing(
            cleaned
        ):
            self._should_end_call = True

    async def _await_outbound_playback(self) -> None:
        await self.flush_playback_output()
        await self._vobiz_finish_playback()

    async def _on_turn_complete(self) -> None:
        if self._out_residual:
            await self._send_paced_frame(bytes(self._out_residual))
            self._out_residual.clear()
        await self.flush_playback_output()
        self._is_playing_welcome = False
        self.is_speaking = False
        self._arm_echo_guard()
        if (
            not self._should_end_call
            and self._caller_wants_hangup
            and self.history
            and self.history[-1]["role"] == "assistant"
        ):
            self._should_end_call = True
        self._schedule_close_after_playback_if_needed()

    async def _on_interrupt(self) -> None:
        self._out_residual.clear()
        self._out_state = None
        self._ulaw_out_buffer.clear()
        self._play_origin = None
        self._play_sent_ms = 0.0
        self._is_playing_welcome = False
        self.is_speaking = False
        await self._send_clear()
