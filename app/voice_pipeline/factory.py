"""
Voice pipeline factory — selects provider from ``VOICE_PIPELINE`` env var.
"""

from __future__ import annotations

import logging

from app import config
from app.voice_pipeline.base import VoicePipelineSession

logger = logging.getLogger(__name__)

_REGISTERED_PIPELINES: dict[str, str] = {
    "compound": "app.voice_pipeline.compound.session.CompoundPipelineSession",
    "openai_realtime": "app.voice_pipeline.openai_realtime.session.OpenAIRealtimePipelineSession",
    "gemini_live": "app.voice_pipeline.gemini_live.session.GeminiLivePipelineSession",
}


def get_pipeline_name() -> str:
    """Active pipeline. Portal engine is Gemini Live for every call."""
    name = (config.VOICE_PIPELINE or "gemini_live").strip().lower()
    if name != "gemini_live":
        logger.info(
            "VOICE_PIPELINE=%s ignored — calls use gemini_live (portal engine)",
            name,
        )
    return "gemini_live"


def create_voice_pipeline_session(
    connection_id: str,
    websocket,
) -> VoicePipelineSession:
    """
    Instantiate the configured voice pipeline for a new Exotel WSS connection.

    Default: ``compound`` (ElevenLabs Realtime STT → OpenAI stream → ElevenLabs WS TTS).
    """
    pipeline = get_pipeline_name()
    if pipeline not in _REGISTERED_PIPELINES:
        pipeline = "gemini_live"

    module_path, class_name = _REGISTERED_PIPELINES[pipeline].rsplit(".", 1)
    import importlib

    module = importlib.import_module(module_path)
    session_cls = getattr(module, class_name)
    logger.debug(
        "Creating %s for %s (VOICE_PIPELINE=%s)",
        class_name,
        connection_id,
        pipeline,
    )
    return session_cls(connection_id, websocket)
