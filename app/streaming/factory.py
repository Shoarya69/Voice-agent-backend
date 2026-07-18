"""Build configured streaming STT / LLM / TTS providers."""

from __future__ import annotations

from dataclasses import dataclass

from app import config
from app.streaming.providers.base import (
    StreamingLLMProvider,
    StreamingSTTProvider,
    StreamingTTSProvider,
)
from app.streaming.providers.deepgram_stt import DeepgramStreamingSTTProvider
from app.streaming.providers.elevenlabs_stt import ElevenLabsStreamingSTTProvider
from app.streaming.providers.elevenlabs_tts_http import ElevenLabsHttpTTSProvider
from app.streaming.providers.elevenlabs_tts_ws import ElevenLabsWebSocketTTSProvider
from app.streaming.providers.openai_llm import OpenAIStreamingLLMProvider


@dataclass
class StreamingProviders:
    stt: StreamingSTTProvider
    llm: StreamingLLMProvider
    tts: StreamingTTSProvider


def build_streaming_providers(connection_id: str) -> StreamingProviders:
    stt_name = (config.STT_PROVIDER or "elevenlabs").strip().lower()
    if stt_name == "deepgram":
        stt: StreamingSTTProvider = DeepgramStreamingSTTProvider(connection_id)
    else:
        stt = ElevenLabsStreamingSTTProvider(connection_id)

    llm: StreamingLLMProvider = OpenAIStreamingLLMProvider()

    tts_name = (config.TTS_PROVIDER or "elevenlabs_ws").strip().lower()
    if tts_name in ("http", "elevenlabs_http"):
        tts: StreamingTTSProvider = ElevenLabsHttpTTSProvider(connection_id)
    else:
        tts = ElevenLabsWebSocketTTSProvider(connection_id)

    return StreamingProviders(stt=stt, llm=llm, tts=tts)
