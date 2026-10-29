"""
Vobiz call on Gemini Native Multimodal Live.

Phone μ-law 8 kHz ↔ Gemini Live audio. No ElevenLabs, no ChatGPT.
Playback uses the same paced playAudio path as the compound Vobiz session.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
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
    resample_24k_to_8k_clean,
    scale_pcm_volume,
    ulaw_bytes_to_pcm,
)
from app.telephony.vobiz.session import VobizCompoundSession
from app.voice_pipeline.gemini_live.barge_in import GeminiBargeInController
from app.voice_pipeline.gemini_live.call_end import (
    assistant_sounds_like_closing,
    user_requested_hangup,
)
from app.voice_pipeline.gemini_live.link import GeminiLiveLink, connect_session

logger = logging.getLogger(__name__)

_NOISE_TRANSCRIPT = (
    "<noise>",
    "[noise]",
    "(noise)",
    "noise",
    "<silence>",
    "[silence]",
    "<unk>",
    "unk",
)


def _is_noise_transcript(text: str) -> bool:
    cleaned = (text or "").strip().lower()
    if not cleaned:
        return True
    return cleaned in _NOISE_TRANSCRIPT

# Cache synthesized 8 kHz PCM greetings by welcome text (in RAM + logs/ disk cache)
# so even the first call after a container restart plays the greeting in <200ms.
_GREETING_DISK_CACHE_PATH = os.path.join("logs", "gemini_greeting_cache_v2.json")
_synthesized_greeting_pcm_cache: dict[str, bytes] = {}


def _load_greeting_disk_cache() -> None:
    try:
        if not os.path.exists(_GREETING_DISK_CACHE_PATH):
            return
        with open(_GREETING_DISK_CACHE_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            for k, b64 in data.items():
                if isinstance(k, str) and isinstance(b64, str) and b64:
                    _synthesized_greeting_pcm_cache[k] = base64.b64decode(b64)
        if _synthesized_greeting_pcm_cache:
            logger.info(
                "🎵 Loaded %s cached Gemini greetings from disk",
                len(_synthesized_greeting_pcm_cache),
            )
    except Exception as exc:
        logger.debug("Greeting disk cache load skipped: %s", exc)


def greeting_cache_key(agent_id: str, greeting_hash: str, welcome_text: str) -> str:
    """Per-agent key so one customer's greeting is never played for another."""
    ident = (agent_id or "").strip() or "unknown"
    tag = (greeting_hash or "").strip() or (welcome_text or "").strip()
    return f"{ident}|{tag}"


def cached_greeting_pcm(agent_id: str, greeting_hash: str, welcome_text: str) -> bytes | None:
    return _synthesized_greeting_pcm_cache.get(
        greeting_cache_key(agent_id, greeting_hash, welcome_text)
    )


def save_synthesized_greeting_pcm(
    welcome_text: str,
    pcm_8k: bytes,
    *,
    agent_id: str = "",
    greeting_hash: str = "",
) -> None:
    if not welcome_text or not pcm_8k or len(pcm_8k) < 3200:
        return
    _synthesized_greeting_pcm_cache[
        greeting_cache_key(agent_id, greeting_hash, welcome_text)
    ] = pcm_8k
    try:
        os.makedirs(os.path.dirname(_GREETING_DISK_CACHE_PATH), exist_ok=True)
        serialized = {
            k: base64.b64encode(v).decode("ascii")
            for k, v in list(_synthesized_greeting_pcm_cache.items())[-25:]
        }
        with open(_GREETING_DISK_CACHE_PATH, "w", encoding="utf-8") as fh:
            json.dump(serialized, fh)
    except Exception as exc:
        logger.debug("Greeting disk cache write skipped: %s", exc)


_load_greeting_disk_cache()


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
        self.accept_ms: float = 0.0
        self.session_ready_ms: float | None = None
        self.first_audio_out_ms: float | None = None
        self._first_audio_logged = False
        self._caller_wants_hangup = False
        self._last_model_audio_at = 0.0
        self._last_user_transcript_at = 0.0
        self._turn_first_audio_logged = True
        self._user_turns = 0
        self._echo_guard_mono_until = 0.0
        self._playback_active_until_mono = 0.0
        self._barge_in_passthrough_until = 0.0
        self._capturing_greeting_pcm = False
        self._captured_greeting_pcm = bytearray()
        self._captured_greeting_key = ""
        self._drop_model_audio_until_next_turn = False
        self._call_error_stage = ""
        self._engine = None
        self._greeting_playback_task: asyncio.Task | None = None
        self._barge_in = GeminiBargeInController()
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
        if getattr(entry, "engine", None) is not None:
            self._engine = entry.engine
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
            self._echo_guard_mono_until = time.monotonic() + 0.45
            self._barge_in.on_assistant_turn_complete()
            self._arm_echo_guard()

    def _record_first_audio_out(self) -> None:
        if self.first_audio_out_ms is None:
            self.first_audio_out_ms = round(
                (time.monotonic() - self._ws_connected_at) * 1000, 1
            )
            logger.info(
                "CALL_METRICS event=first_audio_out session_id=%s provider_call_sid=%s "
                "caller=%s accept_ms=%.1f session_ready_ms=%s first_audio_out_ms=%.1f",
                self.connection_id,
                self.call_sid or "pending",
                (self.caller_from or "")[-4:] if self.caller_from else "unknown",
                self.accept_ms,
                self.session_ready_ms,
                self.first_audio_out_ms,
            )
        if self._first_audio_logged or not self._prewarm_call_id:
            return
        self._first_audio_logged = True
        logger.info(
            "PREWARM event=first_audio_sent call_id=%s pickup_to_first_audio_ms=%.0f",
            self._prewarm_call_id,
            (time.monotonic() - self._ws_connected_at) * 1000,
        )

    async def _connect_and_inject_greeting_context(self, welcome: str) -> None:
        """Connect Gemini Live in background while cached PCM greeting plays."""
        try:
            await self._ensure_link()
            await self._link.inject_assistant_greeting_context(welcome)
        except Exception as exc:
            logger.warning(
                "Background Gemini Live connect during cached greeting failed conn=%s: %s",
                self.connection_id,
                exc,
            )

    async def preconnect_socket(self) -> None:
        """Open Gemini socket in parallel with Supabase agent lookup."""
        try:
            await self._link.open_socket()
        except Exception as exc:
            logger.debug("Gemini early socket open skipped conn=%s: %s", self.connection_id, exc)

    async def warm_playback_pipeline(self) -> None:
        """Queue cached PCM greeting or start Gemini synthesis BEFORE Vobiz `start` arrives."""
        try:
            welcome = self._welcome_text()
            cached = cached_greeting_pcm(
                getattr(self, "agent_id", "") or "",
                getattr(getattr(self, "agent_config", None), "greeting_audio_hash", "") or "",
                welcome,
            )
            if not self.greeting_audio_pcm and cached:
                self.greeting_audio_pcm = cached
            if self.greeting_audio_pcm and not self._greeting_started:
                self._greeting_started = True
                self._prewarm_greeting_done = True
                self.history.append({"role": "assistant", "content": welcome})
                self._barge_in.on_assistant_transcript(welcome)
                self._ulaw_out_buffer.extend(
                    pcm_bytes_to_ulaw(
                        scale_pcm_volume(self.greeting_audio_pcm, config.TTS_VOLUME)
                    )
                )
                asyncio.create_task(self._connect_and_inject_greeting_context(welcome))
                return
            await self._ensure_link()
            await self._start_greeting_if_needed()
        except Exception as exc:
            logger.warning("Gemini Live prewarm failed conn=%s: %s", self.connection_id, exc)

    async def _wait_play_lead(self) -> None:
        """Keep ~260 ms buffered ahead on Vobiz to prevent mid-word jitter/underruns."""
        if self._play_origin is None:
            return
        elapsed_ms = (time.monotonic() - self._play_origin) * 1000
        ahead_ms = self._play_sent_ms - elapsed_ms
        if ahead_ms > 260:
            await asyncio.sleep((ahead_ms - 260) / 1000)

    async def _drain_ulaw(self, *, force: bool) -> None:
        while self._ulaw_out_buffer and not self._vobiz.closed:
            # 640 bytes (80ms) for fast first packet, then 1600 bytes (200ms) smooth frames
            need = 640 if self._play_sent_ms == 0 else 1600
            if len(self._ulaw_out_buffer) < need and not force:
                return
            take = (
                len(self._ulaw_out_buffer)
                if force and len(self._ulaw_out_buffer) < need
                else need
            )
            await self._wait_play_lead()
            block = bytes(self._ulaw_out_buffer[:take])
            del self._ulaw_out_buffer[:take]
            await self._vobiz.play_ulaw(block)
            self._record_first_audio_out()
            now = time.monotonic()
            if self._play_origin is None:
                self._play_origin = now
            self._play_sent_ms += len(block) / 8.0
            # Track exact wall-clock time when Vobiz finishes playing this audio on the phone
            self._playback_active_until_mono = (
                max(now, self._playback_active_until_mono) + (len(block) / 8000.0)
            )
            if force and not self._ulaw_out_buffer:
                return

    async def on_agent_ready(self) -> None:
        if self.session_ready_ms is None:
            self.session_ready_ms = round(
                (time.monotonic() - self._ws_connected_at) * 1000, 1
            )
        welcome = self._welcome_text()
        cached = cached_greeting_pcm(
            getattr(self, "agent_id", "") or "",
            getattr(getattr(self, "agent_config", None), "greeting_audio_hash", "") or "",
            welcome,
        )
        if not self.greeting_audio_pcm and cached:
            self.greeting_audio_pcm = cached
        if self._ulaw_out_buffer and self._vobiz.stream_id:
            await self._drain_ulaw(force=False)
            self._record_first_audio_out()
        if not self.greeting_audio_pcm and not self._greeting_started:
            await self._ensure_link()

    async def add_audio_chunk(self, media_data: dict) -> None:
        if self._closed or not self._link.ready:
            return
        ulaw = decode_payload(media_data.get("payload", ""))
        if not ulaw:
            return
        pcm_8k = ulaw_bytes_to_pcm(ulaw)
        pcm_16k, self._in_state = audioop.ratecv(
            pcm_8k, 2, 1, 8000, 16000, self._in_state
        )

        now = time.monotonic()

        # Release stuck speaking/welcome flags once playback buffer has finished
        if not self._ulaw_out_buffer and now >= (self._playback_active_until_mono + 0.25):
            self.is_speaking = False
            self._is_playing_welcome = False

        # Opening greeting must finish playing on caller's phone + 400ms echo tail.
        greeting_on_phone = (
            self._user_turns == 0
            and (self._is_playing_welcome or now < (self._playback_active_until_mono + 0.4))
        )
        if greeting_on_phone:
            return

        # Once caller barges in, stream their entire utterance uninterrupted for 3.5s
        # so it is never chopped by repeated barge-in triggers.
        if now < self._barge_in_passthrough_until:
            await self._link.send_pcm16k(pcm_16k)
            return

        phone_still_playing = (
            self.is_speaking
            or self._is_playing_welcome
            or now < (self._playback_active_until_mono + 0.25)
        )
        if phone_still_playing:
            # While agent audio is playing on the phone, block phone-line echo from
            # reaching Gemini (prevents self-interruption), BUT
            # allow real caller speech (high RMS) to barge in and silence the agent!
            should_interrupt, frames_to_send = (
                self._barge_in.process_caller_frame_during_playback(
                    pcm_8k, pcm_16k
                )
            )
            if should_interrupt:
                logger.info(
                    "🛑 Caller barge-in detected conn=%s speech_ms=%.0f",
                    self.connection_id,
                    self._barge_in.speech_ms,
                )
                self._barge_in_passthrough_until = now + 3.5
                await self._on_interrupt(local_barge_in=True)
                for chunk_16k in frames_to_send:
                    await self._link.send_pcm16k(chunk_16k)
            return

        self._drop_model_audio_until_next_turn = False
        await self._link.send_pcm16k(pcm_16k)

    async def _start_greeting_if_needed(self) -> None:
        if self._greeting_started:
            return
        welcome = self._welcome_text()
        if not welcome.strip():
            return
        cached = cached_greeting_pcm(
            getattr(self, "agent_id", "") or "",
            getattr(getattr(self, "agent_config", None), "greeting_audio_hash", "") or "",
            welcome,
        )
        if not self.greeting_audio_pcm and cached:
            self.greeting_audio_pcm = cached
        self._greeting_started = True
        self.history.append({"role": "assistant", "content": welcome})
        self._barge_in.on_assistant_transcript(welcome)
        self._barge_in.on_assistant_speaking_start()
        self._is_playing_welcome = True
        self.is_speaking = True
        self._last_model_audio_at = time.monotonic()
        if self.greeting_audio_pcm:
            self._prewarm_greeting_done = True
            self._ulaw_out_buffer.extend(
                pcm_bytes_to_ulaw(
                    scale_pcm_volume(self.greeting_audio_pcm, config.TTS_VOLUME)
                )
            )
            asyncio.create_task(self._connect_and_inject_greeting_context(welcome))
            return
        self._capturing_greeting_pcm = True
        self._captured_greeting_pcm.clear()
        self._captured_greeting_key = welcome
        await self._link.say(
            "The phone call just connected. Speak ONLY this opening greeting "
            f"and do NOT call any tools: {welcome}"
        )

    async def speak_welcome(self) -> None:
        if self._greeting_started and self._ulaw_out_buffer and self._vobiz.stream_id:
            await self._drain_ulaw(force=False)
            self._record_first_audio_out()
            if self._prewarm_greeting_done:
                self._greeting_playback_task = asyncio.create_task(
                    self._finish_prewarm_greeting_playback()
                )
            return

        # Prewarm connected but the model returned no audio. Speak on this call.
        if self._greeting_started and not self._ulaw_out_buffer:
            if not self._prewarm_greeting_done:
                for _ in range(8):
                    if self._ulaw_out_buffer or self.is_speaking:
                        break
                    await asyncio.sleep(0.25)
            if self._ulaw_out_buffer and self._vobiz.stream_id:
                await self._drain_ulaw(force=False)
                self._record_first_audio_out()
                return
            if self.is_speaking:
                return
            logger.warning(
                "Prewarm greeting was empty conn=%s; speaking on the live call",
                self.connection_id,
            )
            self._greeting_started = False
            self._prewarm_greeting_done = False

        if self._greeting_started and not self._prewarm_greeting_done:
            return

        try:
            await self._ensure_link()
            await self._start_greeting_if_needed()
            if self._ulaw_out_buffer and self._vobiz.stream_id:
                await self._drain_ulaw(force=False)
                self._record_first_audio_out()
                if self._prewarm_greeting_done:
                    self._greeting_playback_task = asyncio.create_task(
                        self._finish_prewarm_greeting_playback()
                    )
        except Exception as exc:
            logger.error(
                "Gemini Live connect failed conn=%s: %s",
                self.connection_id,
                exc,
            )

    async def handle_clear(self) -> None:
        await self._on_interrupt()

    async def close(self) -> None:
        if (
            self._greeting_playback_task is not None
            and not self._greeting_playback_task.done()
        ):
            self._greeting_playback_task.cancel()
        await self._link.close()
        await super().close()

    async def _ensure_link(self) -> None:
        if self._link.ready:
            return
        try:
            await connect_session(self._link, self)
        except Exception as exc:
            from app.engine_config import GeminiAuthError

            if isinstance(exc, GeminiAuthError):
                self._call_error_stage = "gemini_auth"
                try:
                    await self.websocket.close(code=1000, reason="gemini auth failed")
                except Exception:
                    pass
            raise

    async def _on_model_audio(self, pcm_24k: bytes) -> None:
        if self._closed or not pcm_24k or self._drop_model_audio_until_next_turn:
            return
        if not self.is_speaking:
            self._barge_in.on_assistant_speaking_start()
        self.is_speaking = True
        self._last_model_audio_at = time.monotonic()
        if not self._turn_first_audio_logged and self._last_user_transcript_at > 0:
            self._turn_first_audio_logged = True
            logger.info(
                "LATENCY_GEMINI_TURN session_id=%s response_start_ms=%.0f",
                self.connection_id,
                (time.monotonic() - self._last_user_transcript_at) * 1000,
            )
        pcm_8k, self._out_state = resample_24k_to_8k_clean(
            pcm_24k, self._out_state
        )
        if not pcm_8k:
            return
        if self._capturing_greeting_pcm and len(self._captured_greeting_pcm) < 128000:
            self._captured_greeting_pcm.extend(pcm_8k)
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
            self._record_first_audio_out()

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
        if role == "user" and _is_noise_transcript(text):
            self._drop_model_audio_until_next_turn = True
            self._out_residual.clear()
            self._ulaw_out_buffer.clear()
            logger.info(
                "Ignoring noise transcript conn=%s text=%s",
                self.connection_id,
                (text or "")[:40],
            )
            return
        if role == "user":
            self._user_turns += 1
            self._last_user_transcript_at = time.monotonic()
            self._turn_first_audio_logged = False
            self._drop_model_audio_until_next_turn = False
            self._barge_in_passthrough_until = 0.0
            if user_requested_hangup(text):
                self._caller_wants_hangup = True
        elif role == "assistant" and self._drop_model_audio_until_next_turn:
            return
        cleaned = self._strip_call_end_marker(text) if role == "assistant" else text
        if role == "assistant":
            self._barge_in.on_assistant_transcript(cleaned)
            if self.history and self.history[-1]["role"] == "assistant":
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
        if self._capturing_greeting_pcm:
            self._capturing_greeting_pcm = False
            if (
                self._captured_greeting_key
                and len(self._captured_greeting_pcm) >= 3200
            ):
                save_synthesized_greeting_pcm(
                    self._captured_greeting_key,
                    bytes(self._captured_greeting_pcm),
                    agent_id=self.agent_id or "",
                    greeting_hash=getattr(self.agent_config, "greeting_audio_hash", "") or "",
                )
            self._captured_greeting_pcm.clear()
        if self._out_residual and not self._drop_model_audio_until_next_turn:
            self._ulaw_out_buffer.extend(
                pcm_bytes_to_ulaw(
                    scale_pcm_volume(bytes(self._out_residual), config.TTS_VOLUME)
                )
            )
            self._out_residual.clear()
        if self._drop_model_audio_until_next_turn:
            self._ulaw_out_buffer.clear()
            self._out_residual.clear()
        elif self._ulaw_out_buffer and self._vobiz.stream_id:
            await self._drain_ulaw(force=True)
        self._play_origin = None
        self._play_sent_ms = 0.0
        self._is_playing_welcome = False
        self.is_speaking = False
        self._barge_in.on_assistant_turn_complete()
        if (
            not self._should_end_call
            and self._caller_wants_hangup
            and self.history
            and self.history[-1]["role"] == "assistant"
        ):
            self._should_end_call = True
        self._schedule_close_after_playback_if_needed()

    async def _on_interrupt(self, local_barge_in: bool = False) -> None:
        # Ignore spurious Gemini server-side interrupts when we already gate echo locally
        if not local_barge_in:
            return
        if (
            self._greeting_playback_task is not None
            and not self._greeting_playback_task.done()
        ):
            self._greeting_playback_task.cancel()
        self._drop_model_audio_until_next_turn = True
        self._capturing_greeting_pcm = False
        self._captured_greeting_pcm.clear()
        self._barge_in.mark_interrupted()
        self._out_residual.clear()
        self._out_state = None
        self._ulaw_out_buffer.clear()
        self._play_origin = None
        self._play_sent_ms = 0.0
        self._playback_active_until_mono = 0.0
        self._is_playing_welcome = False
        self.is_speaking = False
        await self._send_clear()
