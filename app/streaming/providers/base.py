"""
Provider protocols for the low-latency streaming voice pipeline.

Business logic (CRM, agent lookup, call logs) stays outside these interfaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Protocol


class STTEventKind(str, Enum):
    PARTIAL = "partial"
    COMMITTED = "committed"
    ERROR = "error"


@dataclass
class STTEvent:
    kind: STTEventKind
    text: str = ""
    language_code: str = ""
    language_probability: float | None = None
    error: str = ""


@dataclass
class LLMChunk:
    delta: str
    is_final: bool = False


@dataclass
class TTSStats:
    first_byte_ms: float | None = None
    total_ms: float | None = None
    clauses: int = 0
    reconnects: int = 0


class StreamingSTTProvider(ABC):
    """Realtime speech-to-text over a persistent WebSocket."""

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def send_audio(self, pcm: bytes, *, commit: bool = False) -> None: ...

    @abstractmethod
    async def events(self) -> AsyncIterator[STTEvent]: ...

    @abstractmethod
    async def close(self) -> None: ...


class StreamingLLMProvider(ABC):
    """Token-streaming language model."""

    @abstractmethod
    async def stream_reply(self, history: list[dict]) -> AsyncIterator[str]: ...


class StreamingTTSProvider(ABC):
    """Streaming text-to-speech with persistent connection when supported."""

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def begin_utterance(self) -> None:
        """Start a new assistant reply on the persistent connection."""

    @abstractmethod
    async def send_text(self, text: str) -> None: ...

    @abstractmethod
    async def flush_utterance(self) -> None:
        """Signal end of the current assistant reply and drain remaining audio."""

    @abstractmethod
    async def iter_utterance_audio(self) -> AsyncIterator[bytes]:
        """Yield PCM chunks for the current utterance until generation completes."""

    @abstractmethod
    async def audio_chunks(self) -> AsyncIterator[bytes]:
        """Yield PCM chunks for the lifetime of the provider (legacy)."""

    @abstractmethod
    async def close(self) -> None: ...


class PCMFrameSink(Protocol):
    async def send_pcm_frame(self, pcm_frame: bytes) -> bool: ...
