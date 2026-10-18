"""
Human-like VAD, Barge-in (>300ms), and Back-channel (<800ms) controller for Gemini Live.

Rules:
1. While the AI is speaking, caller audio is analyzed with WebRTC VAD + RMS energy.
2. If the AI asked a question, caller speech >= 320 ms immediately interrupts playback
   and flushes buffered speech to Gemini Live.
3. If the AI is mid-explanation (did not ask a question), short caller utterances
   under 800 ms ("haan", "ok", "hmm", "accha") are treated as back-channels and do NOT
   interrupt the AI. Sustained speech > 800 ms interrupts immediately.
4. Response/turn IDs prevent late model audio chunks from playing after an interruption.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from app import config
from app.audio_utils import pcm_rms

try:
    import webrtcvad
except ImportError:
    webrtcvad = None  # type: ignore[assignment]

_BACKCHANNEL_WORDS = frozenset(
    {
        "haan",
        "han",
        "ha",
        "hmm",
        "hm",
        "hmmm",
        "ok",
        "okay",
        "k",
        "accha",
        "achha",
        "acha",
        "ji",
        "haanji",
        "hanji",
        "yes",
        "yeah",
        "yep",
        "right",
        "uhhuh",
        "uh-huh",
        "sahi",
        "theek",
        "thik",
        "हां",
        "हाँ",
        "हम्म",
        "जी",
        "अच्छा",
        "ठीक",
    }
)

_QUESTION_HINT_RE = re.compile(
    r"(\?|\b(kya|kab|kaise|kahan|kaun|konsa|kitna|kitne|batao|bataiye|"
    r"would you|can you|do you|may i|shall we|right\?)\b)",
    re.IGNORECASE,
)


def is_backchannel_text(text: str) -> bool:
    """Return True if a short transcript is purely a back-channel acknowledgement."""
    cleaned = re.sub(r"[^\w\u0900-\u097F\s-]", "", (text or "").strip().lower())
    words = [w for w in cleaned.split() if w]
    if not words or len(words) > 2:
        return False
    return all(w in _BACKCHANNEL_WORDS for w in words)


def assistant_text_is_question(text: str) -> bool:
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    if cleaned.endswith("?"):
        return True
    return bool(_QUESTION_HINT_RE.search(cleaned))


@dataclass
class GeminiBargeInController:
    """Per-call VAD + back-channel filter while the AI is speaking."""

    vad_mode: int = 2
    speech_ms: float = 0.0
    silence_ms: float = 0.0
    barge_in_triggered: bool = False
    ai_asked_question: bool = False
    last_assistant_text: str = ""
    response_generation_id: int = 0
    interrupted_generation_id: int = -1
    playback_started_at: float = 0.0
    buffered_pcm16k: list[bytes] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._vad = None
        if webrtcvad is not None:
            try:
                self._vad = webrtcvad.Vad(self.vad_mode)
            except Exception:
                self._vad = None

    def on_assistant_speaking_start(self) -> int:
        if self.playback_started_at == 0.0:
            self.playback_started_at = time.monotonic()
            self.barge_in_triggered = False
            self.speech_ms = 0.0
            self.silence_ms = 0.0
            self.buffered_pcm16k.clear()
        return self.response_generation_id

    def on_assistant_transcript(self, text: str) -> None:
        if not text:
            return
        self.last_assistant_text = (self.last_assistant_text + " " + text).strip()[-240:]
        self.ai_asked_question = assistant_text_is_question(self.last_assistant_text)

    def on_assistant_turn_complete(self) -> None:
        self.playback_started_at = 0.0
        self.speech_ms = 0.0
        self.silence_ms = 0.0
        self.barge_in_triggered = False
        self.buffered_pcm16k.clear()
        self.response_generation_id += 1

    def mark_interrupted(self) -> int:
        self.interrupted_generation_id = self.response_generation_id
        self.response_generation_id += 1
        self.playback_started_at = 0.0
        self.speech_ms = 0.0
        self.silence_ms = 0.0
        self.barge_in_triggered = True
        return self.response_generation_id

    def is_frame_speech(self, pcm_8k: bytes) -> bool:
        if not pcm_8k or len(pcm_8k) < 160:
            return False
        rms = pcm_rms(pcm_8k)
        min_rms = getattr(config, "GEMINI_BARGE_IN_RMS_THRESHOLD", 550.0)
        if rms < min_rms:
            return False
        if self._vad is not None and len(pcm_8k) in (160, 320, 480):
            try:
                return bool(self._vad.is_speech(pcm_8k, 8000))
            except Exception:
                pass
        return rms >= min_rms * 1.15

    def process_caller_frame_during_playback(
        self,
        pcm_8k: bytes,
        pcm_16k: bytes,
    ) -> tuple[bool, list[bytes]]:
        """
        Inspect one 20ms caller frame while the AI is speaking.

        Returns ``(should_interrupt_now, pcm16k_frames_to_send)``.
        """
        if not getattr(config, "GEMINI_BARGE_IN_ENABLED", True):
            return False, []

        if self.barge_in_triggered:
            return False, [pcm_16k]

        frame_ms = (len(pcm_8k) / 2.0) / 8.0  # 20ms for 320 bytes @ 8kHz
        is_speech = self.is_frame_speech(pcm_8k)

        if is_speech:
            self.speech_ms += frame_ms
            self.silence_ms = 0.0
            self.buffered_pcm16k.append(pcm_16k)
            if len(self.buffered_pcm16k) > 60:
                self.buffered_pcm16k.pop(0)

            min_barge_ms = float(getattr(config, "GEMINI_BARGE_IN_MIN_SPEECH_MS", 320))
            backchannel_max_ms = float(getattr(config, "GEMINI_BACKCHANNEL_MAX_MS", 800))

            # If AI asked a question, >320ms of speech barges in immediately.
            # If AI is mid-explanation, wait until speech exceeds backchannel_max_ms (800ms)
            # so a short "haan" / "ok" / "hmm" (<800ms) does not cut off the explanation.
            threshold_ms = min_barge_ms if self.ai_asked_question else backchannel_max_ms
            if self.speech_ms >= threshold_ms:
                self.barge_in_triggered = True
                flushed = list(self.buffered_pcm16k)
                self.buffered_pcm16k.clear()
                return True, flushed
            return False, []

        # Silence frame while accumulating a short utterance during AI playback
        if self.speech_ms > 0:
            self.silence_ms += frame_ms
            if self.silence_ms >= 220.0:
                # Caller finished a short utterance (<800ms) during AI explanation ->
                # treat as back-channel ("haan"/"ok"/"hmm") and do NOT interrupt!
                self.speech_ms = 0.0
                self.silence_ms = 0.0
                self.buffered_pcm16k.clear()

        return False, []
