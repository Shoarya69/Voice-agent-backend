"""
Overlapping STT → LLM → TTS turn orchestration for minimum time-to-first-audio.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable

import websockets.exceptions

from app import config
from app.audio_utils import chunk_pcm
from app.runtime import tts_semaphore
from app.streaming.clause_buffer import split_next_clause
from app.streaming.factory import StreamingProviders
from app.streaming.providers.base import TTSStats
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
    ) -> tuple[str, TTSStats]:
        history.append({"role": "user", "content": user_text})

        timing = {"first_clause_ms": None, "first_byte_ms": None}
        stats = TTSStats()
        reply_parts: list[str] = []
        words_sent = 0

        await self._providers.tts.begin_utterance()
        if hasattr(self._providers.tts, "reconnect_count"):
            stats.reconnects = getattr(self._providers.tts, "reconnect_count", 0)

        playback_task = asyncio.create_task(
            self._playback_utterance(timeline, timing, stats)
        )

        llm_buffer = ""
        openai_started = False

        try:
            async with tts_semaphore:
                async for delta in self._providers.llm.stream_reply(history):
                    if self._is_cancelled():
                        break
                    if not openai_started:
                        openai_started = True
                        timeline.mark_once("openai_request_ms")
                    timeline.mark_once("openai_first_token_ms")

                    if words_sent >= config.REPLY_WORD_MAX:
                        break

                    llm_buffer += delta
                    reply_parts.append(delta)
                    llm_buffer = self._strip_call_end_marker(llm_buffer, should_end_call)

                    while True:
                        split = split_next_clause(llm_buffer)
                        if split is None:
                            break
                        clause, llm_buffer = split
                        if not clause:
                            break
                        clause, words_sent = self._trim_clause(clause, words_sent)
                        if not clause:
                            break
                        if timing["first_clause_ms"] is None:
                            timing["first_clause_ms"] = round(
                                (time.perf_counter() - turn_started) * 1000
                            )
                            timeline.mark_once("llm_first_sentence_ms")
                        timeline.mark_once("tts_first_request_ms")
                        await self._providers.tts.send_text(clause)
                        stats.clauses += 1
                        if words_sent >= config.REPLY_WORD_MAX:
                            llm_buffer = ""
                            break

                tail = self._strip_call_end_marker(llm_buffer, should_end_call).strip()
                if tail and words_sent < config.REPLY_WORD_MAX:
                    tail, words_sent = self._trim_clause(tail, words_sent)
                    if tail:
                        if timing["first_clause_ms"] is None:
                            timing["first_clause_ms"] = round(
                                (time.perf_counter() - turn_started) * 1000
                            )
                            timeline.mark_once("llm_first_sentence_ms")
                        timeline.mark_once("tts_first_request_ms")
                        await self._providers.tts.send_text(tail)
                        stats.clauses += 1

                await self._providers.tts.flush_utterance()
        except BrainError as exc:
            logger.error("Brain error for %s: %s", self._connection_id, exc)
            if not reply_parts:
                fallback = "Sorry, repeat please?"
                await self._providers.tts.send_text(fallback)
                await self._providers.tts.flush_utterance()
                reply_parts.append(fallback)
        finally:
            try:
                await playback_task
            except asyncio.CancelledError:
                pass

        reply_text = self._truncate_reply_words(
            self._strip_call_end_marker("".join(reply_parts), should_end_call).strip(),
            config.REPLY_WORD_MAX,
        )
        if reply_text:
            history.append({"role": "assistant", "content": reply_text})

        stats.first_byte_ms = timing.get("first_byte_ms")
        stats.total_ms = round((time.perf_counter() - turn_started) * 1000, 1)
        return reply_text, stats

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

    def _trim_clause(self, clause: str, words_sent: int) -> tuple[str, int]:
        remaining = config.REPLY_WORD_MAX - words_sent
        if remaining <= 0:
            return "", words_sent
        trimmed = self._truncate_reply_words(clause, remaining)
        if not trimmed:
            return "", words_sent
        words_sent += len(trimmed.split())
        return trimmed, words_sent
