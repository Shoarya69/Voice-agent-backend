"""
VoiceSession: per-call state machine that ties together audio buffering,
silence detection, ElevenLabs STT/TTS and the OpenAI brain to hold a real
voice conversation over an Exotel bidirectional media stream.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass

import webrtcvad
import websockets.exceptions

from app import config
from app.audio_utils import (
    chunk_pcm,
    decode_payload,
    iter_pcm_frames,
    pcm_duration_ms,
    pcm_rms,
)
from app.elevenlabs_service import ElevenLabsError, STTResult, elevenlabs_service
from app.openai_service import BrainError, openai_brain
from app.latency_trace import CallLatencySummary
from app.runtime import stt_semaphore, tts_semaphore, vad_executor
from app.streaming.session_controller import StreamingSessionController
from app.turn_timeline import TurnTimeline
from app.voice_pipeline.base import VoicePipelineSession

logger = logging.getLogger(__name__)


@dataclass
class VadDecision:
    rms: float
    noise_floor_rms: float
    dynamic_threshold_rms: float
    vad_is_speech: bool
    accepted_as_speech: bool
    reason: str


def _update_noise_floor_state(
    rms: float,
    vad_is_speech: bool,
    noise_floor_rms: float,
    noise_calibration_ms: int,
) -> tuple[float, int]:
    if vad_is_speech and noise_calibration_ms >= config.NOISE_CALIBRATION_MS:
        return noise_floor_rms, noise_calibration_ms

    alpha = 0.15 if noise_calibration_ms < config.NOISE_CALIBRATION_MS else 0.03
    updated = (1 - alpha) * noise_floor_rms + alpha * rms
    noise_floor_rms = min(config.NOISE_FLOOR_MAX_RMS, max(1.0, updated))
    noise_calibration_ms += config.CHUNK_DURATION_MS
    return noise_floor_rms, noise_calibration_ms


def _compute_vad_decisions(
    vad: webrtcvad.Vad,
    frames: list[bytes],
    noise_floor_rms: float,
    max_rms_seen: float,
    noise_calibration_ms: int,
) -> tuple[list[VadDecision], float, float, int]:
    """CPU-heavy VAD/RMS for a batch of frames (runs in a worker thread)."""
    decisions: list[VadDecision] = []
    nf = noise_floor_rms
    max_rms = max_rms_seen
    cal_ms = noise_calibration_ms

    for frame in frames:
        rms = pcm_rms(frame)
        max_rms = max(max_rms, rms)
        try:
            vad_is_speech = vad.is_speech(frame, config.SAMPLE_RATE)
        except Exception:
            vad_is_speech = False

        dynamic_threshold = max(
            config.DYNAMIC_RMS_MIN,
            nf * config.NOISE_FLOOR_MULTIPLIER,
        )
        accepted = vad_is_speech and rms >= dynamic_threshold

        if not accepted:
            nf, cal_ms = _update_noise_floor_state(rms, vad_is_speech, nf, cal_ms)

        if not vad_is_speech:
            reason = "vad_non_speech"
        elif rms < dynamic_threshold:
            reason = "below_dynamic_threshold"
        else:
            reason = "speech"

        decisions.append(
            VadDecision(
                rms=rms,
                noise_floor_rms=nf,
                dynamic_threshold_rms=dynamic_threshold,
                vad_is_speech=vad_is_speech,
                accepted_as_speech=accepted,
                reason=reason,
            )
        )

    return decisions, nf, max_rms, cal_ms


class VoiceSession(VoicePipelineSession):
    """Manages a single call's audio buffering, turn-taking and AI pipeline."""

    def __init__(self, connection_id: str, websocket):
        super().__init__(connection_id, websocket)

        self._vad = webrtcvad.Vad(config.VAD_MODE)
        self._residual_pcm = bytearray()
        self._audio_buffer = bytearray()
        self._turn_monitor_task: asyncio.Task | None = None
        self._response_task: asyncio.Task | None = None

        self.is_speaking = False
        self._is_playing_welcome = False
        self._media_chunks_received = 0
        self._speech_started = False
        self._last_voice_at: float | None = None
        self._speech_candidate_buffer = bytearray()
        self._speech_candidate_ms = 0
        self._speech_candidate_gap_ms = 0
        self._current_turn_vad_speech_ms = 0
        self._turn_processing = False
        self._max_rms_seen = 0.0
        self._noise_floor_rms = max(1.0, config.DYNAMIC_RMS_MIN / config.NOISE_FLOOR_MULTIPLIER)
        self._noise_calibration_ms = 0
        # Monotonic loop-clock timestamp until which inbound audio is ignored,
        # to avoid the bot's own voice (echoed back by the phone line) from
        # poisoning the noise floor or being mistaken for caller speech.
        self._echo_guard_until: float = 0.0
        self._should_end_call = False
        bytes_per_ms = config.SAMPLE_RATE * config.SAMPLE_WIDTH * config.CHANNELS / 1000
        self._max_turn_pcm_bytes = int(config.MAX_TURN_AUDIO_MS * bytes_per_ms)
        self._streaming = StreamingSessionController(self)
        self._pace_next_send_time: float | None = None
        self._latency_summary = CallLatencySummary(connection_id)

    # ------------------------------------------------------------------
    # Inbound audio handling
    # ------------------------------------------------------------------
    def _append_pcm_capped(self, buffer: bytearray, pcm: bytes) -> None:
        if not pcm:
            return
        buffer.extend(pcm)
        if len(buffer) > self._max_turn_pcm_bytes:
            del buffer[: len(buffer) - self._max_turn_pcm_bytes]

    async def add_audio_chunk(self, media_data: dict):
        if self._closed:
            return

        payload = media_data.get("payload", "")
        pcm = decode_payload(payload)
        if not pcm:
            return

        self._ensure_turn_monitor()
        self._residual_pcm.extend(pcm)

        frames = list(iter_pcm_frames(bytes(self._residual_pcm)))
        usable = len(self._residual_pcm) - (len(self._residual_pcm) % config.BYTES_PER_CHUNK)
        if usable:
            del self._residual_pcm[:usable]

        if not frames:
            return

        loop = asyncio.get_running_loop()
        now = loop.time()

        if self._streaming.enabled:
            await self._handle_streaming_audio(pcm, frames, now)
            return

        if not config.ENABLE_BARGE_IN and (
            self.is_speaking or self._is_playing_welcome or now < self._echo_guard_until
        ):
            return

        decisions, nf, max_rms, cal_ms = await loop.run_in_executor(
            vad_executor,
            _compute_vad_decisions,
            self._vad,
            frames,
            self._noise_floor_rms,
            self._max_rms_seen,
            self._noise_calibration_ms,
        )
        self._noise_floor_rms = nf
        self._max_rms_seen = max_rms
        self._noise_calibration_ms = cal_ms

        for frame, decision in zip(frames, decisions):
            self._apply_vad_decision(frame, decision, now)

    def _apply_vad_decision(self, frame: bytes, decision: VadDecision, now: float):
        self._media_chunks_received += 1
        if config.ENABLE_VAD_FRAME_LOGS and (
            self._media_chunks_received == 1
            or self._media_chunks_received % config.VAD_FRAME_LOG_EVERY == 0
        ):
            logger.info(
                (
                    "VAD_FRAME - %s chunks=%s rms=%.1f max_rms=%.1f "
                    "noise_floor=%.1f threshold=%.1f vad=%s accepted=%s buffered=%s"
                ),
                self.connection_id,
                self._media_chunks_received,
                decision.rms,
                self._max_rms_seen,
                decision.noise_floor_rms,
                decision.dynamic_threshold_rms,
                decision.vad_is_speech,
                decision.accepted_as_speech,
                len(self._audio_buffer),
            )

        if decision.accepted_as_speech:
            self._speech_candidate_gap_ms = 0
            if not self._speech_started:
                self._append_pcm_capped(self._speech_candidate_buffer, frame)
                self._speech_candidate_ms += config.CHUNK_DURATION_MS
                if self._speech_candidate_ms < config.VAD_START_MS:
                    return

                logger.info(
                    (
                        "ACCEPTED - voice_start %s speech_ms=%s rms=%.1f "
                        "noise_floor=%.1f threshold=%.1f vad=%s"
                    ),
                    self.connection_id,
                    self._speech_candidate_ms,
                    decision.rms,
                    decision.noise_floor_rms,
                    decision.dynamic_threshold_rms,
                    decision.vad_is_speech,
                )
                self._audio_buffer.clear()
                self._append_pcm_capped(self._audio_buffer, bytes(self._speech_candidate_buffer))
                self._current_turn_vad_speech_ms = self._speech_candidate_ms
                self._speech_candidate_buffer.clear()
                self._speech_candidate_ms = 0
                self._speech_started = True

                if config.ENABLE_BARGE_IN and not self._is_playing_welcome and (
                    self.is_speaking or (self._response_task and not self._response_task.done())
                ):
                    self._interrupt_response()
                    return

                self._last_voice_at = now
                return

            self._last_voice_at = now
            self._current_turn_vad_speech_ms += config.CHUNK_DURATION_MS
            self._append_pcm_capped(self._audio_buffer, frame)
            return

        if self._speech_started:
            self._append_pcm_capped(self._audio_buffer, frame)
            return

        if self._speech_candidate_ms:
            self._speech_candidate_gap_ms += config.CHUNK_DURATION_MS
            if self._speech_candidate_gap_ms <= config.VAD_CANDIDATE_TOLERANCE_MS:
                self._append_pcm_capped(self._speech_candidate_buffer, frame)
                self._speech_candidate_ms += config.CHUNK_DURATION_MS
                return

            logger.info(
                (
                    "REJECTED - speech_candidate_reset %s reason=%s "
                    "candidate_ms=%s rms=%.1f vad=%s"
                ),
                self.connection_id,
                decision.reason,
                self._speech_candidate_ms,
                decision.rms,
                decision.vad_is_speech,
            )
        self._speech_candidate_buffer.clear()
        self._speech_candidate_ms = 0
        self._speech_candidate_gap_ms = 0

    def _classify_frame(self, frame: bytes) -> VadDecision:
        rms = pcm_rms(frame)
        self._max_rms_seen = max(self._max_rms_seen, rms)
        try:
            vad_is_speech = self._vad.is_speech(frame, config.SAMPLE_RATE)
        except Exception as exc:
            logger.warning("REJECTED - vad_error %s error=%s", self.connection_id, exc)
            vad_is_speech = False

        dynamic_threshold = max(
            config.DYNAMIC_RMS_MIN,
            self._noise_floor_rms * config.NOISE_FLOOR_MULTIPLIER,
        )
        accepted = vad_is_speech and rms >= dynamic_threshold

        if not accepted:
            self._update_noise_floor(rms, vad_is_speech)

        if not vad_is_speech:
            reason = "vad_non_speech"
        elif rms < dynamic_threshold:
            reason = "below_dynamic_threshold"
        else:
            reason = "speech"

        return VadDecision(
            rms=rms,
            noise_floor_rms=self._noise_floor_rms,
            dynamic_threshold_rms=dynamic_threshold,
            vad_is_speech=vad_is_speech,
            accepted_as_speech=accepted,
            reason=reason,
        )

    def _update_noise_floor(self, rms: float, vad_is_speech: bool):
        if vad_is_speech and self._noise_calibration_ms >= config.NOISE_CALIBRATION_MS:
            return

        alpha = 0.15 if self._noise_calibration_ms < config.NOISE_CALIBRATION_MS else 0.03
        updated = (1 - alpha) * self._noise_floor_rms + alpha * rms
        self._noise_floor_rms = min(config.NOISE_FLOOR_MAX_RMS, max(1.0, updated))
        self._noise_calibration_ms += config.CHUNK_DURATION_MS

    def _arm_echo_guard(self):
        """Call right after the bot stops speaking to briefly mute inbound audio."""
        self._echo_guard_until = asyncio.get_running_loop().time() + (
            config.ECHO_GUARD_MS / 1000
        )

    async def on_agent_ready(self) -> None:
        """Pre-connect streaming STT/TTS before greeting so caller audio works immediately after."""
        if self._streaming.enabled:
            try:
                await self._streaming.ensure_connected()
            except Exception as exc:
                logger.error(
                    "Streaming pipeline connect failed for %s: %s",
                    self.connection_id,
                    exc,
                )

    async def _handle_streaming_audio(
        self, pcm: bytes, frames: list[bytes], now: float
    ) -> None:
        # While the bot speaks the greeting, ignore inbound audio (telephony echo).
        # Keep this window short — oversized greeting PCM blocks the caller for 20s+.
        if self._is_playing_welcome:
            return

        if not config.ENABLE_BARGE_IN and (
            self.is_speaking or now < self._echo_guard_until
        ):
            return

        if config.ENABLE_BARGE_IN and (
            self.is_speaking
            or self._is_playing_welcome
            or (self._response_task and not self._response_task.done())
        ):
            loop = asyncio.get_running_loop()
            decisions, nf, max_rms, cal_ms = await loop.run_in_executor(
                vad_executor,
                _compute_vad_decisions,
                self._vad,
                frames,
                self._noise_floor_rms,
                self._max_rms_seen,
                self._noise_calibration_ms,
            )
            self._noise_floor_rms = nf
            self._max_rms_seen = max_rms
            self._noise_calibration_ms = cal_ms
            for frame, decision in zip(frames, decisions):
                if decision.accepted_as_speech:
                    self._interrupt_response()
                    break
            if self.is_speaking or (
                self._response_task and not self._response_task.done()
            ):
                await self._streaming.forward_audio(pcm)
                return

        if now < self._echo_guard_until:
            return

        await self._streaming.forward_audio(pcm)

    def _ensure_turn_monitor(self):
        if self._streaming.enabled:
            return
        if self._turn_monitor_task is None or self._turn_monitor_task.done():
            self._turn_monitor_task = asyncio.create_task(self._monitor_turn_silence())

    async def _monitor_turn_silence(self):
        try:
            while not self._closed:
                await asyncio.sleep(config.TURN_MONITOR_INTERVAL_MS / 1000)
                if (
                    self._speech_started
                    and self._last_voice_at is not None
                    and not self._turn_processing
                    and asyncio.get_running_loop().time() - self._last_voice_at
                    >= config.VAD_END_SILENCE_MS / 1000
                ):
                    await self._handle_turn_end()
        except asyncio.CancelledError:
            pass

    async def _handle_turn_end(self):
        if self._closed or not self._audio_buffer:
            return

        self._turn_processing = True
        pcm_bytes = bytes(self._audio_buffer)
        vad_speech_ms = self._current_turn_vad_speech_ms
        duration_ms = pcm_duration_ms(pcm_bytes)
        self._audio_buffer.clear()
        self._speech_started = False
        self._last_voice_at = None
        self._speech_candidate_buffer.clear()
        self._speech_candidate_ms = 0
        self._speech_candidate_gap_ms = 0
        self._current_turn_vad_speech_ms = 0

        if duration_ms < config.MIN_AUDIO_MS_TO_PROCESS:
            logger.info(
                "REJECTED - noise %s duration_ms=%.0f < min_audio_ms=%s vad_speech_ms=%s",
                self.connection_id,
                duration_ms,
                config.MIN_AUDIO_MS_TO_PROCESS,
                vad_speech_ms,
            )
            self._turn_processing = False
            return

        if vad_speech_ms < config.MIN_VAD_SPEECH_MS:
            logger.info(
                "REJECTED - speech_ms %s vad_speech_ms=%s < min_vad_speech_ms=%s duration_ms=%.0f",
                self.connection_id,
                vad_speech_ms,
                config.MIN_VAD_SPEECH_MS,
                duration_ms,
            )
            self._turn_processing = False
            return

        logger.info(
            "ACCEPTED - sending_to_stt %s duration_ms=%.0f vad_speech_ms=%s",
            self.connection_id,
            duration_ms,
            vad_speech_ms,
        )
        timeline = TurnTimeline(connection_id=self.connection_id)
        # Make sure any previous turn (still "thinking" through STT/LLM/TTS,
        # or even still speaking) is fully stopped before starting a new one.
        # Without this, a caller who keeps talking while the bot is still
        # generating its previous reply could get a second turn kicked off
        # here while the first one's _response_task is still alive - both
        # would eventually reach _pace_and_send_frames and send audio to
        # Exotel concurrently, which is what caused two replies to be heard
        # talking over each other ("voice babbling").
        await self._cancel_response_task()
        self._set_response_task(
            self._run_turn(
                pcm_bytes,
                {
                    "duration_ms": duration_ms,
                    "vad_speech_ms": vad_speech_ms,
                },
                timeline,
            )
        )
        self._turn_processing = False

    async def _run_turn(self, pcm_bytes: bytes, turn_stats: dict, timeline: TurnTimeline):
        producer_task: asyncio.Task | None = None
        try:
            turn_started = time.perf_counter()

            timeline.mark("stt_start_ms")
            stt_started = time.perf_counter()
            stt_result = await self._transcribe(pcm_bytes)
            stt_ms = (time.perf_counter() - stt_started) * 1000
            timeline.mark("stt_complete_ms")
            timeline.stt_ms = round(stt_ms, 1)

            if not stt_result.text:
                logger.info("Empty transcript for %s, ignoring turn", self.connection_id)
                logger.info(
                    "LATENCY - rejected_turn %s reason=empty_transcript stt_ms=%.0f turn_total_ms=%.0f",
                    self.connection_id,
                    stt_ms,
                    (time.perf_counter() - turn_started) * 1000,
                )
                timeline.emit()
                return
            if not self._accept_stt_result(stt_result, turn_stats):
                logger.info(
                    "LATENCY - rejected_turn %s reason=stt_rejected stt_ms=%.0f turn_total_ms=%.0f",
                    self.connection_id,
                    stt_ms,
                    (time.perf_counter() - turn_started) * 1000,
                )
                timeline.emit()
                return

            logger.info(
                (
                    "ACCEPTED - stt %s text=%r language=%s language_probability=%s "
                    "avg_word_confidence=%s stt_ms=%.0f"
                ),
                self.connection_id,
                stt_result.text,
                stt_result.language_code,
                stt_result.language_probability,
                stt_result.avg_word_confidence,
                stt_ms,
            )
            self.history.append({"role": "user", "content": stt_result.text})
            self._trim_history()

            # Stream the LLM reply sentence-by-sentence straight into TTS so
            # the caller starts hearing the first sentence while the model is
            # still generating the rest, instead of waiting for the full
            # reply (this was the main source of the multi-second "time lag").
            timing = {"first_sentence_ms": None}
            sentence_queue: asyncio.Queue = asyncio.Queue()
            producer_task = asyncio.create_task(
                self._produce_llm_sentences(
                    sentence_queue, turn_started, timing, timeline
                )
            )
            tts_stats = await self._speak_from_sentence_queue(sentence_queue, timeline)
            reply_text = self._truncate_reply_words(
                (await producer_task).strip(), config.REPLY_WORD_MAX
            )
            producer_task = None

            if not reply_text:
                turn_total_ms = (time.perf_counter() - turn_started) * 1000
                logger.info(
                    "LATENCY - rejected_turn %s reason=empty_reply stt_ms=%.0f turn_total_ms=%.0f",
                    self.connection_id,
                    stt_ms,
                    turn_total_ms,
                )
                timeline.emit()
                return

            logger.info("🧠 Assistant (%s) reply: %s", self.connection_id, reply_text)
            self.history.append({"role": "assistant", "content": reply_text})
            self._trim_history()

            turn_total_ms = (time.perf_counter() - turn_started) * 1000
            timeline.sentences = tts_stats.get("sentences", 0)
            logger.info(
                (
                    "LATENCY - turn %s stt_ms=%.0f llm_first_sentence_ms=%s "
                    "tts_first_byte_ms=%s tts_total_ms=%s sentences=%s turn_total_ms=%.0f"
                ),
                self.connection_id,
                stt_ms,
                timing.get("first_sentence_ms"),
                tts_stats.get("first_byte_ms"),
                tts_stats.get("total_ms"),
                tts_stats.get("sentences"),
                turn_total_ms,
            )
            timeline.emit()
            if self._should_end_call:
                logger.info("☎️ Closing call for %s because LLM requested end-call", self.connection_id)
                await self._close_after_final_reply()
        except asyncio.CancelledError:
            logger.info("🛑 Turn cancelled for %s (interrupted by caller)", self.connection_id)
            raise
        except websockets.exceptions.ConnectionClosed:
            logger.info(
                "WebSocket closed during turn for %s — stopping playback",
                self.connection_id,
            )
        except Exception as exc:
            logger.error("Error running conversation turn for %s: %s", self.connection_id, exc)
        finally:
            if producer_task and not producer_task.done():
                producer_task.cancel()

    async def _transcribe(self, pcm_bytes: bytes) -> STTResult:
        try:
            async with stt_semaphore:
                return await elevenlabs_service.speech_to_text(pcm_bytes)
        except ElevenLabsError as exc:
            logger.error("STT error for %s: %s", self.connection_id, exc)
            return STTResult(text="")

    @staticmethod
    def _split_next_sentence(buffer: str):
        """
        If `buffer` contains a full sentence (or has grown long enough that we
        should start speaking anyway), return (sentence, remainder). Otherwise
        return None to keep accumulating tokens.
        """
        boundary_chars = ".!?।\n"
        for i, ch in enumerate(buffer):
            if ch in boundary_chars and i >= 2:
                return buffer[: i + 1].strip(), buffer[i + 1 :]
        if len(buffer) >= config.REPLY_WORD_MAX * 6:
            idx = buffer.rfind(" ", 0, config.REPLY_WORD_MAX * 6)
            if idx > 8:
                return buffer[:idx].strip(), buffer[idx + 1 :]
        return None

    @staticmethod
    def _truncate_reply_words(text: str, max_words: int) -> str:
        words = text.split()
        if len(words) <= max_words:
            return text.strip()
        return " ".join(words[:max_words]).strip()

    async def _enqueue_llm_sentence(
        self,
        sentence_queue: asyncio.Queue,
        sentence: str,
        words_sent: list[int],
    ) -> bool:
        """Queue one LLM sentence for TTS, respecting the global word cap."""
        if words_sent[0] >= config.REPLY_WORD_MAX:
            return False
        remaining = config.REPLY_WORD_MAX - words_sent[0]
        trimmed = self._truncate_reply_words(sentence, remaining)
        if not trimmed:
            return False
        words_sent[0] += len(trimmed.split())
        await sentence_queue.put(trimmed)
        return words_sent[0] < config.REPLY_WORD_MAX

    async def _produce_llm_sentences(
        self,
        sentence_queue: asyncio.Queue,
        turn_started: float,
        timing: dict,
        timeline: TurnTimeline | None = None,
    ) -> str:
        """Consume the OpenAI token stream, pushing finished sentences to the queue."""
        buffer = ""
        parts: list[str] = []
        words_sent = [0]
        self._should_end_call = False
        openai_requested = False
        try:
            async for delta in openai_brain.stream_reply(self.history):
                if not openai_requested:
                    openai_requested = True
                    if timeline is not None:
                        timeline.mark_once("openai_request_ms")
                if timeline is not None:
                    timeline.mark_once("openai_first_token_ms")
                if words_sent[0] >= config.REPLY_WORD_MAX:
                    break
                buffer += delta
                parts.append(delta)
                buffer = self._strip_call_end_marker(buffer)
                while True:
                    result = self._split_next_sentence(buffer)
                    if result is None:
                        break
                    sentence, buffer = result
                    if sentence:
                        if timing["first_sentence_ms"] is None:
                            timing["first_sentence_ms"] = round(
                                (time.perf_counter() - turn_started) * 1000
                            )
                            if timeline is not None:
                                timeline.mark_once("llm_first_sentence_ms")
                        if not await self._enqueue_llm_sentence(
                            sentence_queue, sentence, words_sent
                        ):
                            buffer = ""
                            break
                if words_sent[0] >= config.REPLY_WORD_MAX:
                    break
            tail = self._strip_call_end_marker(buffer).strip()
            if tail and words_sent[0] < config.REPLY_WORD_MAX:
                if timing["first_sentence_ms"] is None:
                    timing["first_sentence_ms"] = round(
                        (time.perf_counter() - turn_started) * 1000
                    )
                    if timeline is not None:
                        timeline.mark_once("llm_first_sentence_ms")
                await self._enqueue_llm_sentence(sentence_queue, tail, words_sent)
        except BrainError as exc:
            logger.error("Brain error for %s: %s", self.connection_id, exc)
            if not parts:
                fallback = "Sorry, repeat please?"
                await self._enqueue_llm_sentence(sentence_queue, fallback, words_sent)
                return fallback
        finally:
            await sentence_queue.put(None)
        reply = self._strip_call_end_marker("".join(parts)).strip()
        return self._truncate_reply_words(reply, config.REPLY_WORD_MAX)

    def _strip_call_end_marker(self, text: str) -> str:
        marker = config.CALL_END_MARKER
        if marker in text:
            self._should_end_call = True
            return text.replace(marker, "").strip()
        return text

    async def _close_after_final_reply(self):
        """Close the media stream after the final TTS reply has finished."""
        self._closed = True
        try:
            await self.websocket.close(code=1000, reason="assistant ended call")
        except Exception as exc:
            logger.warning("Error closing websocket for %s: %s", self.connection_id, exc)

    async def _speak_from_sentence_queue(
        self, sentence_queue: asyncio.Queue, timeline: TurnTimeline | None = None
    ) -> dict:
        """
        Drive TTS fetching and real-time playback pacing as two decoupled
        stages connected by a small frame queue (a jitter buffer):

          sentence_queue -> [_produce_audio_frames] -> frame_queue -> [_pace_and_send_frames] -> Exotel

        Decoupling them means TTS network jitter (or the time it takes to
        fetch the next sentence) never bleeds into the outbound send timing,
        which is what was causing the choppy/"cut-cut" audio - the old code
        fetched+sent in lockstep, so any hiccup fetching audio directly
        stalled the 20ms send cadence Exotel expects.
        """
        frame_queue: asyncio.Queue = asyncio.Queue(maxsize=250)  # ~5s of audio, caps memory
        timing = {"first_byte_ms": None, "sentences": 0}
        started = time.perf_counter()
        fetcher_task = asyncio.create_task(
            self._produce_audio_frames(
                sentence_queue, frame_queue, timing, started, timeline
            )
        )
        try:
            pace_stats = await self._pace_and_send_frames(frame_queue, timeline)
        finally:
            if not fetcher_task.done():
                fetcher_task.cancel()
            try:
                await fetcher_task
            except asyncio.CancelledError:
                pass
        return {
            "first_byte_ms": timing["first_byte_ms"],
            "total_ms": pace_stats.get("total_ms"),
            "sentences": timing["sentences"],
        }

    async def _produce_audio_frames(
        self,
        sentence_queue: asyncio.Queue,
        frame_queue: asyncio.Queue,
        timing: dict,
        started: float,
        timeline: TurnTimeline | None = None,
    ):
        """Fetch TTS audio for each queued sentence and split it into fixed-size frames."""
        residual = bytearray()
        try:
            while True:
                sentence = await sentence_queue.get()
                if sentence is None:
                    break
                if timeline is not None:
                    timeline.mark_once("tts_first_request_ms")
                try:
                    async with tts_semaphore:
                        async for raw_chunk in elevenlabs_service.stream_text_to_speech(sentence):
                            if timing["first_byte_ms"] is None:
                                timing["first_byte_ms"] = round(
                                    (time.perf_counter() - started) * 1000
                                )
                                if timeline is not None:
                                    timeline.mark_once("tts_first_byte_ms")
                            residual.extend(raw_chunk)
                            while len(residual) >= config.BYTES_PER_CHUNK:
                                frame = bytes(residual[: config.BYTES_PER_CHUNK])
                                del residual[: config.BYTES_PER_CHUNK]
                                await frame_queue.put(frame)
                except ElevenLabsError as exc:
                    logger.error("TTS error for %s: %s", self.connection_id, exc)
                timing["sentences"] += 1
            if residual:
                for frame in chunk_pcm(bytes(residual)):
                    await frame_queue.put(frame)
        finally:
            await frame_queue.put(None)

    async def _send_paced_frame(
        self,
        frame: bytes,
        timeline: TurnTimeline | None = None,
        playback_started: bool = False,
    ) -> bool:
        if self._closed or not self._websocket_is_open():
            return False
        loop = asyncio.get_running_loop()
        now = loop.time()
        if self._pace_next_send_time is None:
            self._pace_next_send_time = now
        delay = self._pace_next_send_time - now
        pace_delay_ms = max(0.0, delay * 1000)
        if delay > 0:
            await asyncio.sleep(delay)
        send_started = time.perf_counter()
        if not await self._send_pcm_frame(frame):
            return False
        send_latency_ms = (time.perf_counter() - send_started) * 1000
        from app import config as app_config
        from app.latency_trace import active_trace

        if app_config.ENABLE_LATENCY_TRACE:
            trace = active_trace()
            if trace is not None:
                trace.record_exotel_send(
                    pace_delay_ms=pace_delay_ms,
                    send_latency_ms=send_latency_ms,
                )
        if timeline is not None:
            timeline.mark_once("first_packet_sent_ms")
            timeline.mark("last_packet_sent_ms")
        self._pace_next_send_time += config.CHUNK_DURATION_MS / 1000
        return True

    async def _pace_and_send_frames(
        self, frame_queue: asyncio.Queue, timeline: TurnTimeline | None = None
    ) -> dict:
        """
        Send frames to Exotel on a precise 20ms clock. Uses a drift-corrected
        schedule (next_send_time += interval) instead of "sleep 20ms after
        every send", because the send itself takes non-zero time - a flat
        post-send sleep silently pushes real playback to run slower than
        real-time, which starves Exotel's jitter buffer and sounds like
        stuttering/glitching every second or so.
        """
        stats = {"total_ms": 0}
        started = time.perf_counter()
        self.is_speaking = True
        self._pace_next_send_time = None
        playback_started = False
        try:
            while True:
                if self._closed or not self._websocket_is_open():
                    break
                wait_started = time.perf_counter()
                frame = await frame_queue.get()
                if playback_started and timeline is not None:
                    gap_ms = (time.perf_counter() - wait_started) * 1000
                    timeline.record_playback_queue_gap(gap_ms)
                if frame is None:
                    break
                if not await self._send_paced_frame(frame, timeline, playback_started):
                    break
                playback_started = True
        finally:
            self.is_speaking = False
            self._arm_echo_guard()
            self._pace_next_send_time = None
            stats["total_ms"] = round((time.perf_counter() - started) * 1000)
        return stats

    async def speak_welcome(self):
        if not config.ENABLE_WELCOME_MESSAGE or not config.WELCOME_MESSAGE:
            return
        self.history.append({"role": "assistant", "content": config.WELCOME_MESSAGE})
        self._set_response_task(self._speak_welcome())

    async def _speak_welcome(self):
        """Play the greeting without letting initial stream noise cancel it."""
        self._is_playing_welcome = True
        try:
            greeting_pcm = getattr(self, "greeting_audio_pcm", None)
            if greeting_pcm:
                duration_ms = pcm_duration_ms(greeting_pcm)
                logger.info(
                    "🎙️ Playing pre-stored greeting for %s duration_ms=%.0f bytes=%s",
                    self.connection_id,
                    duration_ms,
                    len(greeting_pcm),
                )
                frame_queue: asyncio.Queue = asyncio.Queue()
                for frame in chunk_pcm(greeting_pcm):
                    await frame_queue.put(frame)
                await frame_queue.put(None)
                await self._pace_and_send_frames(frame_queue)
            elif self._streaming.enabled:
                await self._streaming.speak_text(config.WELCOME_MESSAGE)
            else:
                sentence_queue: asyncio.Queue = asyncio.Queue()
                await sentence_queue.put(config.WELCOME_MESSAGE)
                await sentence_queue.put(None)
                await self._speak_from_sentence_queue(sentence_queue)
        except asyncio.CancelledError:
            raise
        except websockets.exceptions.ConnectionClosed:
            logger.info(
                "Greeting stopped — WebSocket closed for %s",
                self.connection_id,
            )
        finally:
            self._is_playing_welcome = False
            self._arm_echo_guard()

    def _log_response_task_result(self, task: asyncio.Task, label: str) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is None:
            return
        if isinstance(exc, websockets.exceptions.ConnectionClosed):
            logger.info(
                "WebSocket closed during %s for %s",
                label,
                self.connection_id,
            )
            return
        logger.error(
            "Unhandled %s task error for %s: %s",
            label,
            self.connection_id,
            exc,
        )

    def _set_response_task(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._response_task = task
        task.add_done_callback(
            lambda done: self._log_response_task_result(done, "response")
        )
        return task

    # ------------------------------------------------------------------
    # Outbound audio streaming
    # ------------------------------------------------------------------
    async def _send_pcm_frame(self, pcm_frame: bytes) -> bool:
        return await self.send_pcm_frame(pcm_frame)

    def _interrupt_response(self):
        """Stop the bot mid-sentence and tell Exotel to clear its playback buffer."""
        asyncio.create_task(self._cancel_response_task())
        asyncio.create_task(self._send_clear())

    async def _cancel_response_task(self):
        """
        Cancel the in-flight response task (if any) and wait for it to fully
        unwind - including its finally blocks that reset `is_speaking` and
        arm the echo guard - before returning. This is the single choke
        point that guarantees at most one reply is ever being generated or
        spoken at a time; skipping the `await` here (i.e. just calling
        `.cancel()` and moving on) leaves a window where the old task is
        still mid-cancellation while a new turn/response starts, which is
        what let two replies' audio get interleaved on the outbound stream.
        """
        task = self._response_task
        if task is None or task.done():
            return
        self._streaming.cancel_active_turn()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as exc:
            logger.error("Error while cancelling previous response for %s: %s", self.connection_id, exc)

    async def _send_clear(self):
        try:
            await self.websocket.send(json.dumps({"event": "clear", "stream_sid": self.stream_sid}))
        except Exception:
            pass

    async def handle_clear(self):
        """Handle an inbound CLEAR event (e.g. caller pressed a barge-in key)."""
        self._audio_buffer.clear()
        self._speech_started = False
        self._last_voice_at = None
        self._speech_candidate_buffer.clear()
        self._speech_candidate_ms = 0
        self._speech_candidate_gap_ms = 0
        self._current_turn_vad_speech_ms = 0
        if self._response_task and not self._response_task.done():
            self._response_task.cancel()
        self.is_speaking = False
        try:
            await self.websocket.send(json.dumps({"event": "clear", "stream_sid": self.stream_sid}))
        except Exception:
            pass

    def _trim_history(self):
        max_messages = config.MAX_CONVERSATION_TURNS * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    def _accept_stt_result(self, stt_result: STTResult, turn_stats: dict) -> bool:
        """Accept/reject STT output using duration and model confidence metadata."""
        if turn_stats.get("vad_speech_ms", 0) < config.MIN_VAD_SPEECH_MS:
            logger.info(
                "REJECTED - stt_pre_duration %s vad_speech_ms=%s < min_vad_speech_ms=%s",
                self.connection_id,
                turn_stats.get("vad_speech_ms"),
                config.MIN_VAD_SPEECH_MS,
            )
            return False

        real_words = [word for word in stt_result.words if word.get("type") == "word"]
        audio_events = [word for word in stt_result.words if word.get("type") == "audio_event"]
        if audio_events and not real_words:
            logger.info(
                "REJECTED - stt_audio_event_only %s events=%s text=%r",
                self.connection_id,
                [event.get("text") for event in audio_events],
                stt_result.text,
            )
            return False

        if (
            stt_result.language_probability is not None
            and stt_result.language_probability < config.STT_MIN_LANGUAGE_PROBABILITY
        ):
            logger.info(
                "REJECTED - stt_language_confidence %s language_probability=%.3f < %.3f text=%r",
                self.connection_id,
                stt_result.language_probability,
                config.STT_MIN_LANGUAGE_PROBABILITY,
                stt_result.text,
            )
            return False

        if (
            stt_result.avg_word_confidence is not None
            and stt_result.avg_word_confidence < config.STT_MIN_AVG_WORD_CONFIDENCE
        ):
            logger.info(
                "REJECTED - stt_confidence %s avg=%.3f < %.3f text=%r",
                self.connection_id,
                stt_result.avg_word_confidence,
                config.STT_MIN_AVG_WORD_CONFIDENCE,
                stt_result.text,
            )
            return False

        return True

    async def close(self):
        self._closed = True
        await self._streaming.close()
        if config.ENABLE_LATENCY_TRACE:
            self._latency_summary.emit_call_summary()
        prefetch = getattr(self, "_agent_prefetch_task", None)
        if prefetch is not None and not prefetch.done():
            prefetch.cancel()
        if self._turn_monitor_task:
            self._turn_monitor_task.cancel()
        if self._response_task and not self._response_task.done():
            self._response_task.cancel()
            try:
                await self._response_task
            except asyncio.CancelledError:
                pass
            except websockets.exceptions.ConnectionClosed:
                pass
            except Exception as exc:
                logger.debug(
                    "Response task ended with error during close for %s: %s",
                    self.connection_id,
                    exc,
                )
