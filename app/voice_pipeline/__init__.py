"""
Provider-agnostic voice pipeline layer.

Switch implementations via ``VOICE_PIPELINE`` in ``.env`` (no application code changes).

  compound         — ElevenLabs STT → OpenAI → ElevenLabs TTS (production default)
  openai_realtime  — OpenAI Realtime API (speech-to-speech)
"""

from app.voice_pipeline.base import VoicePipelineSession
from app.voice_pipeline.factory import create_voice_pipeline_session, get_pipeline_name

__all__ = [
    "VoicePipelineSession",
    "create_voice_pipeline_session",
    "get_pipeline_name",
]
