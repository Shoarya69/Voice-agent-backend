"""
Per-turn latency instrumentation for the Compound voice pipeline.

All timestamps are milliseconds relative to ``anchor`` (user end-of-speech).
Zero behavioral impact — logging only.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TurnTimeline:
    """Structured latency markers for one caller turn."""

    connection_id: str
    anchor: float = field(default_factory=time.perf_counter)

    user_end_of_speech_ms: float = 0.0
    stt_start_ms: float | None = None
    stt_complete_ms: float | None = None
    openai_request_ms: float | None = None
    openai_first_token_ms: float | None = None
    llm_first_sentence_ms: float | None = None
    tts_first_request_ms: float | None = None
    tts_first_byte_ms: float | None = None
    first_packet_sent_ms: float | None = None
    last_packet_sent_ms: float | None = None

    stt_ms: float | None = None
    sentences: int = 0
    playback_queue_gaps_ms: list[float] = field(default_factory=list)
    playback_queue_gap_total_ms: float = 0.0

    def mark(self, name: str, at: float | None = None) -> float:
        """Record ``name`` as ms since anchor; returns the value."""
        value = round(((at or time.perf_counter()) - self.anchor) * 1000, 1)
        setattr(self, name, value)
        return value

    def mark_once(self, name: str, at: float | None = None) -> float | None:
        """Set ``name`` only if it has not been set yet."""
        if getattr(self, name) is not None:
            return getattr(self, name)
        return self.mark(name, at)

    def record_playback_queue_gap(self, gap_ms: float) -> None:
        """Playback waited for audio — indicates TTS/LLM pipeline starvation."""
        if gap_ms < 25:
            return
        rounded = round(gap_ms, 1)
        self.playback_queue_gaps_ms.append(rounded)
        self.playback_queue_gap_total_ms += rounded

    def time_to_first_audio_ms(self) -> float | None:
        if self.first_packet_sent_ms is None:
            return None
        return self.first_packet_sent_ms

    def emit(self) -> None:
        gaps = self.playback_queue_gaps_ms
        logger.info(
            (
                "TURN_TIMELINE connection=%s "
                "user_end=0 "
                "stt_start=%s stt_done=%s stt_ms=%s "
                "openai_req=%s openai_first_token=%s llm_first_sentence=%s "
                "tts_first_req=%s tts_first_byte=%s "
                "first_packet=%s last_packet=%s "
                "sentences=%s queue_gaps=%s queue_gap_total_ms=%.0f "
                "time_to_first_audio_ms=%s"
            ),
            self.connection_id,
            self.stt_start_ms,
            self.stt_complete_ms,
            self.stt_ms,
            self.openai_request_ms,
            self.openai_first_token_ms,
            self.llm_first_sentence_ms,
            self.tts_first_request_ms,
            self.tts_first_byte_ms,
            self.first_packet_sent_ms,
            self.last_packet_sent_ms,
            self.sentences,
            gaps,
            self.playback_queue_gap_total_ms,
            self.time_to_first_audio_ms(),
        )
