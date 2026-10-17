"""
Overlapping STT → LLM → TTS turn orchestration for minimum time-to-first-audio.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncIterator, Awaitable, Callable

import websockets.exceptions

from app import config
from app.audio_utils import chunk_pcm
from app.elevenlabs_service import ElevenLabsError, elevenlabs_service
from app.latency_trace import active_trace
from app.runtime import tts_semaphore
from app.streaming.clause_buffer import split_next_clause
from app.streaming.factory import StreamingProviders
from app.streaming.providers.base import TTSStats
from app.streaming.providers.elevenlabs_tts_ws import ElevenLabsWebSocketTTSProvider
from app.streaming.providers.openai_llm import brain_error_type
from app.turn_timeline import TurnTimeline

logger = logging.getLogger(__name__)
BrainError = brain_error_type()

SendFrameFn = Callable[[bytes, TurnTimeline | None, bool], Awaitable[bool]]


class StreamingTurnRunner:
    """
    Runs one assistant turn with maximum overlap:

      LLM token stream → clause buffer → TTS (persistent WS) → PCM frames → Exotel
    """

    def __init__(
        self,
        *,
        connection_id: str,
        providers: StreamingProviders,
        send_frame: SendFrameFn,
        is_cancelled: Callable[[], bool],
    ) -> None:
        self._connection_id = connection_id
        self._providers = providers
        self._send_frame = send_frame
        self._is_cancelled = is_cancelled

    async def run(
        self,
        *,
        history: list[dict],
        user_text: str,
        timeline: TurnTimeline,
        turn_started: float,
        should_end_call: list[bool],
        llm_user_text: str | None = None,
    ) -> tuple[str, TTSStats]:
        history.append({"role": "user", "content": user_text})
        llm_messages = list(history)
        if llm_user_text is not None and llm_messages:
            llm_messages[-1] = {"role": "user", "content": llm_user_text}

        timing = {"first_clause_ms": None, "first_byte_ms": None}
        stats = TTSStats()
        reply_parts: list[str] = []
        words_sent = 0
        word_cap = (
            config.VOBIZ_REPLY_WORD_MAX
            if self._connection_id.startswith("vobiz_")
            else config.REPLY_WORD_MAX
        )
        clause_max_chars = (
            config.VOBIZ_STREAMING_CLAUSE_MAX_CHARS
            if self._connection_id.startswith("vobiz_")
            else None
        )

        use_http_tts = self._connection_id.startswith("vobiz_")
        playback_task: asyncio.Task | None = None
        if not use_http_tts:
            await self._providers.tts.begin_utterance()
            if hasattr(self._providers.tts, "reconnect_count"):
                stats.reconnects = getattr(self._providers.tts, "reconnect_count", 0)
            playback_task = asyncio.create_task(
                self._playback_utterance(timeline, timing, stats)
            )

        llm_buffer = ""
        openai_started = False
        stop_llm = False

        try:
            async with tts_semaphore:
                try:
                    async for delta in self._stream_llm_deltas(llm_messages):
                        if self._is_cancelled() or stop_llm:
                            break
                        if not openai_started:
                            openai_started = True
                            timeline.mark_once("openai_request_ms")
                        timeline.mark_once("openai_first_token_ms")
                        trace = active_trace()
                        if trace is not None:
                            trace.record_llm_token()

                        if words_sent >= word_cap:
                            break

                        llm_buffer += delta
                        reply_parts.append(delta)
                        llm_buffer = self._strip_call_end_marker(
                            llm_buffer, should_end_call
                        )

                        while True:
                            split = split_next_clause(
                                llm_buffer, max_chars=clause_max_chars
                            )
                            if split is None:
                                break
                            clause, llm_buffer = split
                            if not clause:
                                break
                            clause, words_sent = self._trim_clause(
                                clause, words_sent, word_cap
                            )
                            if not clause:
                                break
                            if timing["first_clause_ms"] is None:
                                timing["first_clause_ms"] = round(
                                    (time.perf_counter() - turn_started) * 1000
                                )
                                timeline.mark_once("llm_first_sentence_ms")
                            timeline.mark_once("tts_first_request_ms")
                            trace = active_trace()
                            if trace is not None:
                                trace.record_llm_clause(clause)
                            if use_http_tts:
                                await self._play_http_clause(clause, timeline, timing)
                            else:
                                await self._providers.tts.send_text(clause)
                            stats.clauses += 1
                            if words_sent >= word_cap:
                                llm_buffer = ""
                                stop_llm = True
                                break

                    tail = self._strip_call_end_marker(
                        llm_buffer, should_end_call
                    ).strip()
                    if tail and words_sent < word_cap and not stop_llm:
                        tail, words_sent = self._trim_clause(
                            tail, words_sent, word_cap
                        )
                        if tail:
                            if timing["first_clause_ms"] is None:
                                timing["first_clause_ms"] = round(
                                    (time.perf_counter() - turn_started) * 1000
                                )
                                timeline.mark_once("llm_first_sentence_ms")
                            timeline.mark_once("tts_first_request_ms")
                            trace = active_trace()
                            if trace is not None:
                                trace.record_llm_clause(tail)
                            if use_http_tts:
                                await self._play_http_clause(tail, timeline, timing)
                            else:
                                await self._providers.tts.send_text(tail)
                            stats.clauses += 1
                finally:
                    if stats.clauses > 0 and not use_http_tts:
                        await self._providers.tts.flush_utterance()
        except BrainError as exc:
            logger.error("Brain error for %s: %s", self._connection_id, exc)
            if not reply_parts:
                fallback = "Sorry, repeat please?"
                await self._providers.tts.send_text(fallback)
                await self._providers.tts.flush_utterance()
                reply_parts.append(fallback)
        finally:
            if playback_task is not None:
                try:
                    await asyncio.wait_for(playback_task, timeout=3.0)
                except asyncio.TimeoutError:
                    playback_task.cancel()
                    logger.warning(
                        "TTS playback timed out conn=%s",
                        self._connection_id,
                    )
                except asyncio.CancelledError:
                    pass
            if (
                not use_http_tts
                and timing.get("first_byte_ms") is None
                and reply_parts
            ):
                logger.warning(
                    "TTS produced no audio conn=%s clauses=%s — reconnect + one-shot retry",
                    self._connection_id,
                    stats.clauses,
                )
                retry_text = self._truncate_reply_words(
                    self._strip_call_end_marker(
                        "".join(reply_parts), should_end_call
                    ).strip(),
                    word_cap,
                )
                if retry_text and isinstance(
                    self._providers.tts, ElevenLabsWebSocketTTSProvider
                ):
                    await self._retry_turn_audio(
                        retry_text, timeline, timing, stats, turn_started
                    )
                if timing.get("first_byte_ms") is None and retry_text:
                    await self._play_http_fallback(retry_text, timeline, timing)

        reply_text = self._truncate_reply_words(
            self._strip_call_end_marker("".join(reply_parts), should_end_call).strip(),
            word_cap,
        )
        if reply_text:
            history.append({"role": "assistant", "content": reply_text})

        stats.first_byte_ms = timing.get("first_byte_ms")
        stats.total_ms = round((time.perf_counter() - turn_started) * 1000, 1)
        return reply_text, stats

    async def _stream_llm_deltas(
        self, llm_messages: list[dict]
    ) -> AsyncIterator[str]:
        """Yield LLM tokens; stop on idle/max so TTS flush always runs."""
        is_vobiz = self._connection_id.startswith("vobiz_")
        idle_sec = (
            config.VOBIZ_LLM_STREAM_IDLE_SEC if is_vobiz else 3.0
        )
        max_sec = config.VOBIZ_LLM_STREAM_MAX_SEC if is_vobiz else 25.0
        stream = self._providers.llm.stream_reply(llm_messages)
        started = time.perf_counter()
        got_token = False
        try:
            while True:
                if self._is_cancelled():
                    break
                if time.perf_counter() - started > max_sec:
                    logger.info(
                        "LLM stream max duration conn=%s sec=%.1f",
                        self._connection_id,
                        max_sec,
                    )
                    break
                # First token can take ~1.3s; idle cutoff only after speech has started.
                wait_sec = idle_sec if got_token else max(6.0, idle_sec)
                try:
                    delta = await asyncio.wait_for(
                        stream.__anext__(), timeout=wait_sec
                    )
                except asyncio.TimeoutError:
                    logger.info(
                        "LLM stream idle stop conn=%s idle_sec=%.2f got_token=%s",
                        self._connection_id,
                        wait_sec,
                        got_token,
                    )
                    break
                except StopAsyncIteration:
                    break
                if delta:
                    got_token = True
                    yield delta
        finally:
            close = getattr(stream, "aclose", None)
            if close is not None:
                try:
                    await close()
                except Exception:
                    pass

    async def _play_http_clause(
        self,
        text: str,
        timeline: TurnTimeline,
        timing: dict,
    ) -> None:
        logger.info(
            "HTTP TTS speak conn=%s chars=%s",
            self._connection_id,
            len(text),
        )
        await self._play_http_pcm(text, timeline, timing)

    async def _play_http_fallback(
        self,
        text: str,
        timeline: TurnTimeline,
        timing: dict,
    ) -> None:
        logger.info(
            "HTTP TTS fallback conn=%s chars=%s",
            self._connection_id,
            len(text),
        )
        await self._play_http_pcm(text, timeline, timing)

    async def _play_http_pcm(
        self,
        text: str,
        timeline: TurnTimeline,
        timing: dict,
    ) -> None:
        residual = bytearray()
        playback_started = False
        try:
            async for raw in elevenlabs_service.stream_text_to_speech(text):
                if self._is_cancelled() or not raw:
                    continue
                if timing.get("first_byte_ms") is None:
                    timing["first_byte_ms"] = round(
                        (time.perf_counter() - timeline.anchor) * 1000,
                        1,
                    )
                    timeline.mark_once("tts_first_byte_ms")
                residual.extend(raw)
                while len(residual) >= config.BYTES_PER_CHUNK:
                    frame = bytes(residual[: config.BYTES_PER_CHUNK])
                    del residual[: config.BYTES_PER_CHUNK]
                    if not await self._send_frame(frame, timeline, playback_started):
                        return
                    playback_started = True
            if residual:
                for frame in chunk_pcm(bytes(residual)):
                    if not await self._send_frame(frame, timeline, playback_started):
                        return
        except ElevenLabsError as exc:
            logger.error(
                "HTTP TTS fallback failed conn=%s: %s",
                self._connection_id,
                exc,
            )

    async def _retry_turn_audio(
        self,
        text: str,
        timeline: TurnTimeline,
        timing: dict,
        stats: TTSStats,
        turn_started: float,
    ) -> None:
        tts = self._providers.tts
        assert isinstance(tts, ElevenLabsWebSocketTTSProvider)
        await tts.hard_reconnect()
        playback_task = asyncio.create_task(
            self._playback_utterance(timeline, timing, stats)
        )
        try:
            await tts.begin_utterance()
            timeline.mark_once("tts_first_request_ms")
            await tts.send_text(text)
            await tts.flush_utterance()
        finally:
            try:
                await asyncio.wait_for(playback_task, timeout=12.0)
            except asyncio.TimeoutError:
                playback_task.cancel()
                logger.warning(
                    "TTS retry playback timed out conn=%s",
                    self._connection_id,
                )
            except asyncio.CancelledError:
                pass

    async def _playback_utterance(
        self,
        timeline: TurnTimeline,
        timing: dict,
        stats: TTSStats,
    ) -> None:
        residual = bytearray()
        playback_started = False
        try:
            async for raw_chunk in self._providers.tts.iter_utterance_audio():
                if self._is_cancelled():
                    break
                if timing["first_byte_ms"] is None:
                    timing["first_byte_ms"] = round(
                        (time.perf_counter() - timeline.anchor) * 1000,
                        1,
                    )
                    timeline.mark_once("tts_first_byte_ms")
                residual.extend(raw_chunk)
                trace = active_trace()
                if trace is not None:
                    trace.record_pcm_residual(len(residual))
                while len(residual) >= config.BYTES_PER_CHUNK:
                    frame = bytes(residual[: config.BYTES_PER_CHUNK])
                    del residual[: config.BYTES_PER_CHUNK]
                    if not await self._send_frame(frame, timeline, playback_started):
                        return
                    playback_started = True
            if residual:
                unpadded_len = len(bytes(residual))
                trace = active_trace()
                if (
                    trace is not None
                    and unpadded_len % config.BYTES_PER_CHUNK != 0
                ):
                    trace.record_pcm_silence_pad()
                for frame in chunk_pcm(bytes(residual)):
                    if not await self._send_frame(frame, timeline, playback_started):
                        return
                    playback_started = True
        except websockets.exceptions.ConnectionClosed:
            pass

    @staticmethod
    def _strip_call_end_marker(text: str, should_end_call: list[bool]) -> str:
        marker = config.CALL_END_MARKER
        if marker in text:
            should_end_call[0] = True
            return text.replace(marker, "").strip()
        return text

    @staticmethod
    def _truncate_reply_words(text: str, max_words: int) -> str:
        words = text.split()
        if len(words) <= max_words:
            return text.strip()
        return " ".join(words[:max_words]).strip()

    def _trim_clause(
        self, clause: str, words_sent: int, word_cap: int | None = None
    ) -> tuple[str, int]:
        cap = word_cap if word_cap is not None else config.REPLY_WORD_MAX
        remaining = cap - words_sent
        if remaining <= 0:
            return "", words_sent
        trimmed = self._truncate_reply_words(clause, remaining)
        if not trimmed:
            return "", words_sent
        words_sent += len(trimmed.split())
        return trimmed, words_sent
