"""
ElevenLabs integration: Speech-to-Text (STT) for transcribing caller audio,
and Text-to-Speech (TTS) for turning the assistant's reply into audio that
matches Exotel's expected format (raw PCM, 8kHz, 16-bit, mono).
"""

import asyncio
import logging
import math
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx

from app import config
from app.http_limits import default_limits, default_timeout
from app.audio_utils import pcm_to_wav_bytes

logger = logging.getLogger(__name__)

_TTS_OUTPUT_FORMAT_BY_RATE = {
    8000: "pcm_8000",
    16000: "pcm_16000",
    22050: "pcm_22050",
    24000: "pcm_24000",
    44100: "pcm_44100",
}


class ElevenLabsError(Exception):
    """Raised when an ElevenLabs API call fails."""


@dataclass
class STTResult:
    text: str
    language_code: str = ""
    language_probability: float | None = None
    audio_duration_secs: float | None = None
    words: list[dict[str, Any]] = field(default_factory=list)
    avg_word_confidence: float | None = None


class ElevenLabsService:
    def __init__(self):
        self.api_key = config.ELEVENLABS_API_KEY
        self.base_url = config.ELEVENLABS_BASE_URL.rstrip("/")
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            async with self._client_lock:
                if self._client is None or self._client.is_closed:
                    self._client = httpx.AsyncClient(
                        limits=default_limits(),
                        timeout=default_timeout(30.0),
                    )
        return self._client

    async def close(self):
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    def _headers(self) -> dict:
        return {"xi-api-key": self.api_key}

    async def speech_to_text(self, pcm_bytes: bytes) -> STTResult:
        """Transcribe raw PCM audio (as sent by Exotel) into text."""
        if not self.api_key:
            raise ElevenLabsError("ELEVENLABS_API_KEY is not configured")
        if not pcm_bytes:
            return STTResult(text="")

        wav_bytes = pcm_to_wav_bytes(pcm_bytes)
        client = await self._get_client()

        data = {"model_id": config.ELEVENLABS_STT_MODEL}
        if config.ELEVENLABS_STT_LANGUAGE:
            data["language_code"] = config.ELEVENLABS_STT_LANGUAGE

        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}

        try:
            response = await client.post(
                f"{self.base_url}/v1/speech-to-text",
                headers=self._headers(),
                data=data,
                files=files,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error("ElevenLabs STT failed: %s - %s", exc.response.status_code, exc.response.text)
            raise ElevenLabsError(f"STT request failed: {exc.response.status_code}") from exc
        except httpx.HTTPError as exc:
            logger.error("ElevenLabs STT connection error: %s", exc)
            raise ElevenLabsError("STT connection error") from exc

        payload = response.json()
        words = payload.get("words") or []
        word_confidences = [
            math.exp(word["logprob"])
            for word in words
            if word.get("type") == "word" and isinstance(word.get("logprob"), (int, float))
        ]
        avg_word_confidence = (
            sum(word_confidences) / len(word_confidences) if word_confidences else None
        )

        return STTResult(
            text=(payload.get("text") or "").strip(),
            language_code=payload.get("language_code") or "",
            language_probability=payload.get("language_probability"),
            audio_duration_secs=payload.get("audio_duration_secs"),
            words=words,
            avg_word_confidence=avg_word_confidence,
        )

    async def text_to_speech(self, text: str) -> bytes:
        """Convert text into raw PCM audio at the configured sample rate."""
        chunks = []
        async for chunk in self.stream_text_to_speech(text):
            chunks.append(chunk)
        return b"".join(chunks)

    async def stream_text_to_speech(self, text: str) -> AsyncIterator[bytes]:
        """Stream text-to-speech bytes as ElevenLabs generates them."""
        if not self.api_key:
            raise ElevenLabsError("ELEVENLABS_API_KEY is not configured")
        if not text.strip():
            return

        output_format = _TTS_OUTPUT_FORMAT_BY_RATE.get(config.SAMPLE_RATE, "pcm_8000")
        client = await self._get_client()

        try:
            async with client.stream(
                "POST",
                f"{self.base_url}/v1/text-to-speech/{config.ELEVENLABS_VOICE_ID}/stream",
                headers={**self._headers(), "Content-Type": "application/json"},
                params={"output_format": output_format, "optimize_streaming_latency": 3},
                json={
                    "text": text,
                    "model_id": config.ELEVENLABS_TTS_MODEL,
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75,
                        "style": 0.0,
                        "use_speaker_boost": True,
                    },
                },
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    if chunk:
                        yield chunk
        except httpx.HTTPStatusError as exc:
            logger.error("ElevenLabs TTS failed: %s", exc.response.status_code)
            raise ElevenLabsError(f"TTS request failed: {exc.response.status_code}") from exc
        except httpx.HTTPError as exc:
            logger.error("ElevenLabs TTS connection error: %s", exc)
            raise ElevenLabsError("TTS connection error") from exc


elevenlabs_service = ElevenLabsService()
