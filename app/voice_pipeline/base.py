"""
Shared call-session contract for all voice pipeline providers.

Business logic in ai_server.py (agent lookup, CRM call logs, channel release)
depends only on this interface — not on a specific STT/LLM/TTS stack.
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import ClassVar

import websockets.exceptions

from app import config
from app.audio_utils import encode_payload, scale_pcm_volume

logger = logging.getLogger(__name__)


class VoicePipelineSession(ABC):
    """
    Per-call session shared by every voice pipeline implementation.

    Subclasses implement Exotel media handling (STT/LLM/TTS path or Realtime path).
    CRM / agent / channel fields live here so ai_server stays provider-agnostic.
    """

    pipeline_name: ClassVar[str] = "base"

    def __init__(self, connection_id: str, websocket) -> None:
        self.connection_id = connection_id
        self.websocket = websocket

        # Exotel stream identifiers
        self.stream_sid: str | None = None
        self.call_sid: str | None = None

        # WSS routing (token or Exophone number)
        self.wss_route_kind: str | None = None
        self.wss_route_value: str | None = None
        self.lovable_token: str | None = None

        # MoontechPro agent bundle (set by agent_setup)
        self.agent_id: str | None = None
        self.agent_config = None
        self.greeting_audio_pcm: bytes | None = None

        # Call metadata (CRM / call-log)
        self.call_started_at = None
        self.caller_from: str | None = None
        self.caller_to: str | None = None
        self.history: list[dict] = []
        self.recording_url: str | None = None

        # Agent prefetch (shared across pipelines)
        self._agent_prefetch_task = None
        self._prefetch_started_at: float | None = None

        # Channel reservation + call-log idempotency flags
        self._call_log_posted = False
        self._channel_reserved = False
        self._channel_reserved_logged = False
        self._channel_released = False
        self._call_started_logged = False
        self._call_finalized = False

        self._closed = False
        self._should_end_call = False

    # ------------------------------------------------------------------
    # Shared CRM / conversation history helpers
    # ------------------------------------------------------------------
    def _trim_history(self) -> None:
        max_messages = config.MAX_CONVERSATION_TURNS * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    def _strip_call_end_marker(self, text: str) -> str:
        marker = config.CALL_END_MARKER
        if marker in text:
            self._should_end_call = True
            return text.replace(marker, "").strip()
        return text

    def _append_history_message(self, role: str, content: str) -> None:
        content = (content or "").strip()
        if not content:
            return
        self.history.append({"role": role, "content": content})
        self._trim_history()

    async def _close_after_final_reply(self) -> None:
        """Close the Exotel media stream after the assistant finishes speaking."""
        self._closed = True
        try:
            await self.websocket.close(code=1000, reason="assistant ended call")
        except Exception as exc:
            logger.warning(
                "Error closing websocket for %s: %s",
                self.connection_id,
                exc,
            )

    def _schedule_close_after_playback_if_needed(self) -> None:
        if self._should_end_call and not self._closed:
            asyncio.create_task(self._close_after_final_reply())

    # ------------------------------------------------------------------
    # Pipeline-specific Exotel media handling
    # ------------------------------------------------------------------
    @abstractmethod
    async def add_audio_chunk(self, media_data: dict) -> None:
        """Handle inbound Exotel ``media`` event payload."""

    @abstractmethod
    async def speak_welcome(self) -> None:
        """Play the per-agent or fallback greeting after ``start``."""

    @abstractmethod
    async def handle_clear(self) -> None:
        """Handle inbound Exotel ``clear`` (barge-in / buffer flush)."""

    @abstractmethod
    async def close(self) -> None:
        """Release pipeline resources for this call."""

    async def on_agent_ready(self) -> None:
        """Hook after MoontechPro agent config is resolved (``start`` event)."""

    # ------------------------------------------------------------------
    # Shared Exotel outbound audio (reused by all providers)
    # ------------------------------------------------------------------
    def _websocket_is_open(self) -> bool:
        if self._closed:
            return False
        state = getattr(self.websocket, "state", None)
        if state is None:
            return True
        try:
            from websockets.protocol import State

            return state is State.OPEN
        except ImportError:
            return True

    async def send_pcm_frame(self, pcm_frame: bytes) -> bool:
        """Send one 20ms PCM frame to Exotel as a ``media`` event."""
        if not self._websocket_is_open():
            return False
        pcm_frame = scale_pcm_volume(pcm_frame, config.TTS_VOLUME)
        media_response = {
            "event": "media",
            "stream_sid": self.stream_sid,
            "media": {"payload": encode_payload(pcm_frame)},
        }
        try:
            await self.websocket.send(json.dumps(media_response))
            return True
        except websockets.exceptions.ConnectionClosed:
            self._closed = True
            return False
        except Exception as exc:
            logger.warning(
                "Failed to send PCM frame for %s: %s",
                self.connection_id,
                exc,
            )
            self._closed = True
            return False

    async def send_exotel_clear(self) -> None:
        try:
            await self.websocket.send(
                json.dumps({"event": "clear", "stream_sid": self.stream_sid})
            )
        except Exception:
            pass
