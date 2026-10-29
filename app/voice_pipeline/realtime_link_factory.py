"""
Create the Gemini realtime transport for one call.

``VOICE_REALTIME_ENGINE=gemini``  — direct Google Live WebSocket (default)
``VOICE_REALTIME_ENGINE=livekit`` — LiveKit Agents Google plugin
"""

from __future__ import annotations

import logging
from typing import Awaitable, Callable

from app import config

logger = logging.getLogger(__name__)

AudioHandler = Callable[[bytes], Awaitable[None]]
TextHandler = Callable[[str, str], Awaitable[None]]
InterruptHandler = Callable[[], Awaitable[None]]
EndCallHandler = Callable[[], Awaitable[None]]
TurnCompleteHandler = Callable[[], Awaitable[None]]


def get_realtime_engine_name() -> str:
    name = (config.VOICE_REALTIME_ENGINE or "gemini").strip().lower()
    if name in ("livekit", "livekit_gemini", "lk"):
        return "livekit"
    return "gemini"


def create_realtime_link(
    connection_id: str,
    *,
    on_audio: AudioHandler,
    on_transcript: TextHandler | None = None,
    on_interrupt: InterruptHandler | None = None,
    on_turn_complete: TurnCompleteHandler | None = None,
    on_end_call: EndCallHandler | None = None,
):
    engine = get_realtime_engine_name()
    if engine == "livekit":
        from app.voice_pipeline.livekit_gemini.link import LiveKitGeminiLink

        logger.debug("Creating LiveKitGeminiLink conn=%s", connection_id)
        return LiveKitGeminiLink(
            connection_id,
            on_audio=on_audio,
            on_transcript=on_transcript,
            on_interrupt=on_interrupt,
            on_turn_complete=on_turn_complete,
            on_end_call=on_end_call,
        )

    from app.voice_pipeline.gemini_live.link import GeminiLiveLink

    return GeminiLiveLink(
        connection_id,
        on_audio=on_audio,
        on_transcript=on_transcript,
        on_interrupt=on_interrupt,
        on_turn_complete=on_turn_complete,
        on_end_call=on_end_call,
    )
