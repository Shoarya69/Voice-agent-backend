"""
Detailed per-turn and per-call latency tracing for the streaming voice pipeline.

Logging-only — zero behavioral impact when ``ENABLE_LATENCY_TRACE`` is false.
"""

from __future__ import annotations

import contextvars
import logging
import statistics
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_current_trace: contextvars.ContextVar[TurnLatencyTrace | None] = contextvars.ContextVar(
    "current_turn_latency_trace", default=None
)


def active_trace() -> TurnLatencyTrace | None:
    return _current_trace.get()


def set_active_trace(trace: TurnLatencyTrace | None):
    return _current_trace.set(trace)


@dataclass
class TurnLatencyTrace:
    """Fine-grained metrics for one assistant turn."""

    connection_id: str
    turn_index: int
    anchor: float = field(default_factory=time.perf_counter)

    # STT (ms since anchor)
    stt_committed_ms: float | None = None
    stt_partial_count: int = 0
    user_text: str = ""

    # LLM
    llm_first_token_ms: float | None = None
    llm_token_count: int = 0
    llm_token_intervals_ms: list[float] = field(default_factory=list)
    llm_clause_times_ms: list[float] = field(default_factory=list)
    llm_clause_texts: list[str] = field(default_factory=list)

    # TTS WebSocket
    tts_ws_send_latencies_ms: list[float] = field(default_factory=list)
    tts_ws_send_labels: list[str] = field(default_factory=list)
    tts_first_byte_ms: float | None = None
    tts_chunk_intervals_ms: list[float] = field(default_factory=list)
    tts_chunk_bytes: list[int] = field(default_factory=list)
    tts_is_final_premature_count: int = 0
    tts_flush_ms: float | None = None

    # PCM playback
    pcm_residual_bytes_before_send: list[int] = field(default_factory=list)
    pcm_frames_sent: int = 0
    pcm_underflow_events: int = 0
    pcm_underflow_total_ms: float = 0.0
    pcm_underflow_gaps_ms: list[float] = field(default_factory=list)
    pcm_silence_pad_frames: int = 0
    pcm_queue_high_water_frames: int = 0

    # Exotel outbound
    exotel_pace_delay_ms: list[float] = field(default_factory=list)
    exotel_send_latency_ms: list[float] = field(default_factory=list)
    exotel_first_packet_ms: float | None = None
    exotel_last_packet_ms: float | None = None

    _last_llm_token_at: float | None = field(default=None, repr=False)
    _last_tts_chunk_at: float | None = field(default=None, repr=False)

    def _since_anchor_ms(self, at: float | None = None) -> float:
        return round(((at or time.perf_counter()) - self.anchor) * 1000, 1)

    def mark_stt_committed(self, text: str) -> None:
        self.user_text = text
        if self.stt_committed_ms is None:
            self.stt_committed_ms = self._since_anchor_ms()

    def record_stt_partial(self) -> None:
        self.stt_partial_count += 1

    def record_llm_token(self) -> None:
        now = time.perf_counter()
        if self.llm_first_token_ms is None:
            self.llm_first_token_ms = self._since_anchor_ms(now)
        if self._last_llm_token_at is not None:
            interval = (now - self._last_llm_token_at) * 1000
            if interval >= 0:
                self.llm_token_intervals_ms.append(round(interval, 1))
        self._last_llm_token_at = now
        self.llm_token_count += 1

    def record_llm_clause(self, clause: str) -> None:
        self.llm_clause_times_ms.append(self._since_anchor_ms())
        self.llm_clause_texts.append(clause[:60])

    def record_tts_ws_send(self, label: str, latency_ms: float) -> None:
        self.tts_ws_send_labels.append(label)
        self.tts_ws_send_latencies_ms.append(round(latency_ms, 1))

    def record_tts_chunk(self, nbytes: int) -> None:
        now = time.perf_counter()
        if self.tts_first_byte_ms is None:
            self.tts_first_byte_ms = self._since_anchor_ms(now)
        if self._last_tts_chunk_at is not None:
            interval = (now - self._last_tts_chunk_at) * 1000
            if interval >= 0:
                self.tts_chunk_intervals_ms.append(round(interval, 1))
        self._last_tts_chunk_at = now
        self.tts_chunk_bytes.append(nbytes)

    def record_tts_premature_is_final(self) -> None:
        self.tts_is_final_premature_count += 1

    def mark_tts_flush(self) -> None:
        self.tts_flush_ms = self._since_anchor_ms()

    def record_pcm_underflow(self, wait_ms: float) -> None:
        if wait_ms < 10:
            return
        self.pcm_underflow_events += 1
        self.pcm_underflow_total_ms += wait_ms
        self.pcm_underflow_gaps_ms.append(round(wait_ms, 1))

    def record_pcm_residual(self, nbytes: int) -> None:
        self.pcm_residual_bytes_before_send.append(nbytes)
        frames = nbytes // 320 if nbytes else 0
        if frames > self.pcm_queue_high_water_frames:
            self.pcm_queue_high_water_frames = frames

    def record_pcm_silence_pad(self) -> None:
        self.pcm_silence_pad_frames += 1

    def record_exotel_send(self, *, pace_delay_ms: float, send_latency_ms: float) -> None:
        if pace_delay_ms > 0.1:
            self.exotel_pace_delay_ms.append(round(pace_delay_ms, 1))
        self.exotel_send_latency_ms.append(round(send_latency_ms, 1))
        now_ms = self._since_anchor_ms()
        if self.exotel_first_packet_ms is None:
            self.exotel_first_packet_ms = now_ms
        self.exotel_last_packet_ms = now_ms
        self.pcm_frames_sent += 1

    def time_to_first_audio_ms(self) -> float | None:
        return self.exotel_first_packet_ms

    def emit_turn_summary(self) -> None:
        llm_ttft = self.llm_first_token_ms
        tts_ttfb = self.tts_first_byte_ms
        ttfa = self.time_to_first_audio_ms()
        llm_interval_p50 = _percentile(self.llm_token_intervals_ms, 50)
        llm_interval_p95 = _percentile(self.llm_token_intervals_ms, 95)
        tts_interval_p50 = _percentile(self.tts_chunk_intervals_ms, 50)
        tts_interval_p95 = _percentile(self.tts_chunk_intervals_ms, 95)
        send_p50 = _percentile(self.exotel_send_latency_ms, 50)
        send_p95 = _percentile(self.exotel_send_latency_ms, 95)
        underflow_p95 = _percentile(self.pcm_underflow_gaps_ms, 95)

        logger.info(
            (
                "LATENCY_TURN connection=%s turn=%s "
                "e2e_stt_ms=%s e2e_llm_ttft_ms=%s e2e_tts_ttfb_ms=%s e2e_ttfa_ms=%s "
                "stt_committed_ms=%s llm_ttft_ms=%s llm_tokens=%s "
                "llm_token_interval_p50=%s llm_token_interval_p95=%s "
                "clauses=%s clause_times_ms=%s "
                "tts_ws_send_ms=%s tts_ttfb_ms=%s tts_chunk_interval_p50=%s "
                "tts_chunk_interval_p95=%s tts_premature_is_final=%s tts_flush_ms=%s "
                "pcm_frames=%s pcm_underflows=%s pcm_underflow_total_ms=%.0f "
                "pcm_underflow_p95=%s pcm_silence_pads=%s pcm_queue_high_water_frames=%s "
                "exotel_send_p50=%s exotel_send_p95=%s playback_ms=%s"
            ),
            self.connection_id,
            self.turn_index,
            self.stt_committed_ms,
            llm_ttft,
            tts_ttfb,
            ttfa,
            self.stt_committed_ms,
            llm_ttft,
            self.llm_token_count,
            llm_interval_p50,
            llm_interval_p95,
            len(self.llm_clause_times_ms),
            self.llm_clause_times_ms,
            self.tts_ws_send_latencies_ms,
            tts_ttfb,
            tts_interval_p50,
            tts_interval_p95,
            self.tts_is_final_premature_count,
            self.tts_flush_ms,
            self.pcm_frames_sent,
            self.pcm_underflow_events,
            self.pcm_underflow_total_ms,
            underflow_p95,
            self.pcm_silence_pad_frames,
            self.pcm_queue_high_water_frames,
            send_p50,
            send_p95,
            (
                round(self.exotel_last_packet_ms - self.exotel_first_packet_ms, 1)
                if self.exotel_first_packet_ms is not None
                and self.exotel_last_packet_ms is not None
                else None
            ),
        )


@dataclass
class CallLatencySummary:
    """Aggregated latency across all turns in one WSS call."""

    connection_id: str
    call_started_at: float = field(default_factory=time.perf_counter)
    turns: list[TurnLatencyTrace] = field(default_factory=list)

    def add_turn(self, trace: TurnLatencyTrace) -> None:
        self.turns.append(trace)

    def emit_call_summary(self) -> None:
        if not self.turns:
            logger.info(
                "LATENCY_CALL connection=%s turns=0 duration_ms=%.0f",
                self.connection_id,
                (time.perf_counter() - self.call_started_at) * 1000,
            )
            return

        ttfa_values = [t.time_to_first_audio_ms() for t in self.turns]
        ttfa_values = [v for v in ttfa_values if v is not None]
        stt_values = [t.stt_committed_ms for t in self.turns if t.stt_committed_ms is not None]
        llm_ttft = [t.llm_first_token_ms for t in self.turns if t.llm_first_token_ms is not None]
        tts_ttfb = [t.tts_first_byte_ms for t in self.turns if t.tts_first_byte_ms is not None]
        underflows = sum(t.pcm_underflow_events for t in self.turns)
        underflow_ms = sum(t.pcm_underflow_total_ms for t in self.turns)
        premature_is_final = sum(t.tts_is_final_premature_count for t in self.turns)
        silence_pads = sum(t.pcm_silence_pad_frames for t in self.turns)

        bottleneck = _identify_bottleneck(self.turns)

        logger.info(
            (
                "LATENCY_CALL connection=%s turns=%s duration_ms=%.0f "
                "ttfa_p50=%s ttfa_p95=%s stt_commit_p50=%s llm_ttft_p50=%s "
                "tts_ttfb_p50=%s pcm_underflow_events=%s pcm_underflow_total_ms=%.0f "
                "tts_premature_is_final_total=%s silence_pad_frames=%s "
                "bottleneck=%s"
            ),
            self.connection_id,
            len(self.turns),
            (time.perf_counter() - self.call_started_at) * 1000,
            _percentile(ttfa_values, 50),
            _percentile(ttfa_values, 95),
            _percentile(stt_values, 50),
            _percentile(llm_ttft, 50),
            _percentile(tts_ttfb, 50),
            underflows,
            underflow_ms,
            premature_is_final,
            silence_pads,
            bottleneck,
        )


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 1)
    try:
        return round(statistics.quantiles(values, n=100)[int(pct) - 1], 1)
    except Exception:
        sorted_vals = sorted(values)
        idx = min(len(sorted_vals) - 1, int(len(sorted_vals) * pct / 100))
        return round(sorted_vals[idx], 1)


def _identify_bottleneck(turns: list[TurnLatencyTrace]) -> str:
    """Heuristic bottleneck label from aggregated turn traces."""
    if not turns:
        return "none"

    avg_underflow = sum(t.pcm_underflow_total_ms for t in turns) / len(turns)
    avg_ttft = _avg([t.llm_first_token_ms for t in turns])
    avg_tts_ttfb = _avg([t.tts_first_byte_ms for t in turns])
    avg_stt = _avg([t.stt_committed_ms for t in turns])
    premature = sum(t.tts_is_final_premature_count for t in turns)

    if avg_underflow >= 200:
        return "tts_pcm_starvation"
    if premature > 0:
        return "tts_is_final_between_clauses"
    if avg_ttft is not None and avg_ttft >= 1200:
        return "llm_ttft"
    if avg_tts_ttfb is not None and avg_tts_ttfb >= 400:
        return "tts_first_byte"
    if avg_stt is not None and avg_stt >= 600:
        return "stt_commit"
    return "balanced"


def _avg(values: list[float | None]) -> float | None:
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)
