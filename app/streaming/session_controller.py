"""
Streaming pipeline controller — wired into VoiceSession without touching CRM/agent logic.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from app import config
from app.elevenlabs_service import STTResult
from app.latency_trace import (
    CallLatencySummary,
    TurnLatencyTrace,
    active_trace,
    set_active_trace,
)
from app.runtime import stt_semaphore
from app.streaming.factory import StreamingProviders, build_streaming_providers
from app.streaming.providers.base import STTEventKind
from app.streaming.turn_runner import StreamingTurnRunner
from app.turn_timeline import TurnTimeline

if TYPE_CHECKING:
    from app.voice_session import VoiceSession

logger = logging.getLogger(__name__)


class StreamingSessionController:
    def __init__(self, session: VoiceSession) -> None:
        self._session = session
        self._providers: StreamingProviders | None = None
        self._stt_task: asyncio.Task | None = None
        self._connected = False
        self._connect_lock = asyncio.Lock()
        self._turn_in_progress = False
        self._cancel_turn = False

    @property
    def enabled(self) -> bool:
        return config.STREAMING_PIPELINE

    async def ensure_connected(self) -> None:
        if not self.enabled or self._connected:
            return
        async with self._connect_lock:
            if self._connected:
                return
            self._providers = build_streaming_providers(self._session.connection_id)
            async with stt_semaphore:
                await self._providers.stt.connect()
            await self._providers.tts.connect()
            self._stt_task = asyncio.create_task(self._stt_event_loop())
            self._connected = True
            logger.info(
                "Streaming pipeline ready conn=%s stt=%s model=%s tts=%s",
                self._session.connection_id,
                config.STT_PROVIDER,
                config.STREAMING_STT_MODEL,
                config.TTS_PROVIDER,
            )

    async def forward_audio(self, pcm: bytes) -> None:
        if not self.enabled or not pcm or self._session._closed:
            return
        await self.ensure_connected()
        assert self._providers is not None
        await self._ensure_stt_listener()
        stt = self._providers.stt
        if hasattr(stt, "ensure_live"):
            await stt.ensure_live()
        await stt.send_audio(pcm)

    async def _ensure_stt_listener(self) -> None:
        """Restart STT event consumer if it exited after a websocket drop."""
        if self._stt_task is None or self._stt_task.done():
            if self._stt_task is not None and not self._stt_task.cancelled():
                exc = self._stt_task.exception()
                if exc is not None:
                    logger.error(
                        "STT event loop ended conn=%s error=%s — restarting",
                        self._session.connection_id,
                        exc,
                    )
            logger.warning(
                "Restarting STT event loop conn=%s",
                self._session.connection_id,
            )
            self._stt_task = asyncio.create_task(self._stt_event_loop())

    async def refresh_stt_after_playback(self) -> None:
        """Call after bot TTS so STT is live before the caller speaks."""
        if not self.enabled or self._providers is None:
            return
        stt = self._providers.stt
        if hasattr(stt, "ensure_live"):
            await stt.ensure_live()
        await self._ensure_stt_listener()
        logger.info(
            "STT refreshed after playback conn=%s",
            self._session.connection_id,
        )

    async def close(self) -> None:
        if self._stt_task is not None:
            self._stt_task.cancel()
            try:
                await self._stt_task
            except asyncio.CancelledError:
                pass
        if self._providers is not None:
            await self._providers.stt.close()
            await self._providers.tts.close()
            self._providers = None
        self._connected = False

    async def speak_text(self, text: str) -> None:
        """Play arbitrary text (greeting) via streaming TTS."""
        await self.ensure_connected()
        assert self._providers is not None
        timeline = TurnTimeline(connection_id=self._session.connection_id)
        turn_started = time.perf_counter()
        should_end_call = [False]
        runner = StreamingTurnRunner(
            connection_id=self._session.connection_id,
            providers=self._providers,
            send_frame=self._session._send_paced_frame,
            is_cancelled=lambda: self._session._closed,
        )
        self._session.is_speaking = True
        try:
            await self._providers.tts.begin_utterance()
            timeline.mark_once("tts_first_request_ms")
            await self._providers.tts.send_text(text)
            await self._providers.tts.flush_utterance()
            residual = bytearray()
            playback_started = False
            async for raw in self._providers.tts.iter_utterance_audio():
                if self._session._closed:
                    break
                timeline.mark_once("tts_first_byte_ms")
                residual.extend(raw)
                while len(residual) >= config.BYTES_PER_CHUNK:
                    frame = bytes(residual[: config.BYTES_PER_CHUNK])
                    del residual[: config.BYTES_PER_CHUNK]
                    if not await self._session._send_paced_frame(
                        frame, timeline, playback_started
                    ):
                        break
                    playback_started = True
        finally:
            self._session.is_speaking = False
            self._session._arm_echo_guard()
            timeline.emit()
            await self.refresh_stt_after_playback()

    async def _stt_event_loop(self) -> None:
        assert self._providers is not None
        try:
            async for event in self._providers.stt.events():
                if event.kind == STTEventKind.PARTIAL:
                    trace = active_trace()
                    if trace is not None:
                        trace.record_stt_partial()
                    logger.debug(
                        "STT partial conn=%s text=%r",
                        self._session.connection_id,
                        event.text[:80],
                    )
                    continue
                if event.kind == STTEventKind.ERROR:
                    logger.error(
                        "STT streaming error conn=%s: %s",
                        self._session.connection_id,
                        event.error,
                    )
                    continue
                if event.kind != STTEventKind.COMMITTED:
                    continue
                await self._handle_committed_transcript(event.text, event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                "STT event loop failed conn=%s: %s",
                self._session.connection_id,
                exc,
            )

    async def _handle_committed_transcript(self, text: str, event) -> None:
        session = self._session
        if session._closed or session._is_playing_welcome:
            logger.info(
                "STT committed ignored conn=%s playing_welcome=%s text=%r",
                session.connection_id,
                session._is_playing_welcome,
                text[:80],
            )
            return
        if self._turn_in_progress:
            return

        stt_result = STTResult(
            text=text,
            language_code=getattr(event, "language_code", "") or "",
            language_probability=getattr(event, "language_probability", None),
        )
        turn_stats = {"vad_speech_ms": config.MIN_VAD_SPEECH_MS}

        if not text.strip():
            return
        if not session._accept_stt_result(stt_result, turn_stats):
            logger.info(
                "REJECTED - streaming stt conn=%s text=%r",
                session.connection_id,
                text,
            )
            return

        logger.info(
            "ACCEPTED - streaming stt conn=%s text=%r",
            session.connection_id,
            text,
        )

        if config.ENABLE_BARGE_IN and (
            session.is_speaking
            or (session._response_task and not session._response_task.done())
        ):
            session._interrupt_response()
            await session._cancel_response_task()

        await session._cancel_response_task()
        self._cancel_turn = False
        session._set_response_task(self._run_turn(text))

    async def _run_turn(self, user_text: str) -> None:
        session = self._session
        self._turn_in_progress = True
        timeline = TurnTimeline(connection_id=session.connection_id)
        turn_index = len(session._latency_summary.turns) + 1
        trace = TurnLatencyTrace(
            connection_id=session.connection_id,
            turn_index=turn_index,
        )
        trace.mark_stt_committed(user_text)
        trace_token = set_active_trace(trace)
        timeline.mark("stt_start_ms")
        timeline.mark("stt_complete_ms")
        timeline.stt_ms = 0.0
        turn_started = time.perf_counter()
        should_end_call = [False]

        try:
            assert self._providers is not None
            runner = StreamingTurnRunner(
                connection_id=session.connection_id,
                providers=self._providers,
                send_frame=session._send_paced_frame,
                is_cancelled=lambda: self._cancel_turn or session._closed,
            )
            session.is_speaking = True
            reply_text, stats = await runner.run(
                history=session.history,
                user_text=user_text,
                timeline=timeline,
                turn_started=turn_started,
                should_end_call=should_end_call,
            )
            session._should_end_call = should_end_call[0]
            session._trim_history()

            if not reply_text:
                timeline.emit()
                if config.ENABLE_LATENCY_TRACE:
                    trace.emit_turn_summary()
                    session._latency_summary.add_turn(trace)
                return

            logger.info("🧠 Assistant (%s) reply: %s", session.connection_id, reply_text)
            turn_total_ms = (time.perf_counter() - turn_started) * 1000
            timeline.sentences = stats.clauses
            logger.info(
                (
                    "LATENCY - turn %s stt_ms=%.0f llm_first_sentence_ms=%s "
                    "tts_first_byte_ms=%s tts_total_ms=%s clauses=%s turn_total_ms=%.0f"
                ),
                session.connection_id,
                0,
                timeline.llm_first_sentence_ms,
                stats.first_byte_ms,
                stats.total_ms,
                stats.clauses,
                turn_total_ms,
            )
            timeline.emit()
            if config.ENABLE_LATENCY_TRACE:
                trace.emit_turn_summary()
                session._latency_summary.add_turn(trace)

            if session._should_end_call:
                logger.info(
                    "☎️ Closing call for %s because LLM requested end-call",
                    session.connection_id,
                )
                await session._close_after_final_reply()
        except asyncio.CancelledError:
            logger.info(
                "🛑 Streaming turn cancelled for %s",
                session.connection_id,
            )
            raise
        finally:
            set_active_trace(trace_token)
            session.is_speaking = False
            session._arm_echo_guard()
            self._turn_in_progress = False
            self._cancel_turn = False
            await self.refresh_stt_after_playback()

    def cancel_active_turn(self) -> None:
        self._cancel_turn = True
