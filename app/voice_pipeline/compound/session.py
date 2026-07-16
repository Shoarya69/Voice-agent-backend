"""
Compound pipeline — production default.

Exotel → ElevenLabs STT → OpenAI (chat) → ElevenLabs TTS → Exotel

Implementation lives in ``app.voice_session.VoiceSession``; this module is the
registered provider alias so the factory can select it without coupling callers
to the legacy module path.
"""

from __future__ import annotations

from typing import ClassVar

from app.voice_session import VoiceSession


class CompoundPipelineSession(VoiceSession):
    """ElevenLabs STT + OpenAI + ElevenLabs TTS (existing production pipeline)."""

    pipeline_name: ClassVar[str] = "compound"
