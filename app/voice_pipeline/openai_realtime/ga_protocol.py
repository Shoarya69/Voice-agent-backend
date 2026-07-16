"""
OpenAI Realtime API — GA protocol helpers (not Beta).

Reference: https://developers.openai.com/api/docs/guides/realtime
"""

from __future__ import annotations

from app import config

# GA server event names (replacements for Beta names).
EVENT_OUTPUT_AUDIO_DELTA = "response.output_audio.delta"
EVENT_OUTPUT_AUDIO_TRANSCRIPT_DELTA = "response.output_audio_transcript.delta"
EVENT_OUTPUT_AUDIO_TRANSCRIPT_DONE = "response.output_audio_transcript.done"
EVENT_RESPONSE_DONE = "response.done"
EVENT_INPUT_TRANSCRIPTION_COMPLETED = (
    "conversation.item.input_audio_transcription.completed"
)

# PCM at 24 kHz is required by GA Realtime audio/pcm format.
GA_PCM_SAMPLE_RATE = 24_000


def build_ga_session_update(*, instructions: str) -> dict:
    """Build a GA ``session.update`` payload for speech-to-speech voice agents."""
    session: dict = {
        "type": "realtime",
        "instructions": instructions,
        "output_modalities": ["audio"],
        "audio": {
            "input": {
                "format": {
                    "type": "audio/pcm",
                    "rate": GA_PCM_SAMPLE_RATE,
                },
                "transcription": {
                    "model": config.OPENAI_REALTIME_TRANSCRIPTION_MODEL,
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": config.OPENAI_REALTIME_VAD_THRESHOLD,
                    "prefix_padding_ms": config.OPENAI_REALTIME_PREFIX_PADDING_MS,
                    "silence_duration_ms": config.OPENAI_REALTIME_SILENCE_DURATION_MS,
                    "create_response": True,
                },
            },
            "output": {
                "format": {
                    "type": "audio/pcm",
                },
                "voice": config.OPENAI_REALTIME_VOICE,
            },
        },
        "temperature": config.OPENAI_TEMPERATURE,
        "max_output_tokens": config.OPENAI_MAX_TOKENS,
    }

    tools = build_ga_tools()
    if tools:
        session["tools"] = tools
        session["tool_choice"] = "auto"

    return {"type": "session.update", "session": session}


def build_ga_response_create(*, instructions: str) -> dict:
    """Build a GA ``response.create`` payload."""
    payload: dict = {
        "type": "response.create",
        "response": {
            "output_modalities": ["audio"],
            "instructions": instructions,
        },
    }
    tools = build_ga_tools()
    if tools:
        payload["response"]["tools"] = tools
    return payload


def build_ga_assistant_message_item(text: str) -> dict:
    """Inject assistant context (e.g. pre-recorded PCM greeting transcript)."""
    return {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": text.strip(),
                }
            ],
        },
    }


def build_ga_tools() -> list:
    """
    Provider-agnostic tool definitions in GA Realtime format.

    Extend via app.voice_pipeline.tools when function calling is added;
    both Compound and Realtime providers should register handlers there.
    """
    try:
        from app.voice_pipeline import tools as pipeline_tools

        return pipeline_tools.build_realtime_tools()
    except ImportError:
        return []
