"""
Gemini Native Multimodal Live API (speech-to-speech).

One WebSocket. Caller audio in, model audio out. No ElevenLabs, no ChatGPT.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Awaitable, Callable

import websockets
import websockets.exceptions

from app import config

logger = logging.getLogger(__name__)

AudioHandler = Callable[[bytes], Awaitable[None]]
TextHandler = Callable[[str, str], Awaitable[None]]
InterruptHandler = Callable[[], Awaitable[None]]


class GeminiLiveLink:
    """Bidirectional Gemini Live session for one phone call."""

    def __init__(
        self,
        connection_id: str,
        *,
        on_audio: AudioHandler,
        on_transcript: TextHandler | None = None,
        on_interrupt: InterruptHandler | None = None,
        on_turn_complete: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._connection_id = connection_id
        self._on_audio = on_audio
        self._on_transcript = on_transcript
        self._on_interrupt = on_interrupt
        self._on_turn_complete = on_turn_complete
        self._ws = None
        self._reader: asyncio.Task | None = None
        self._setup_done = asyncio.Event()
        self._setup_ok = False
        self._turn_done = asyncio.Event()
        self._turn_done.set()
        self._closed = False

    @property
    def ready(self) -> bool:
        return self._setup_done.is_set() and self._ws is not None

    async def connect(self, instructions: str) -> None:
        if self.ready or self._closed:
            return
        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not configured")

        url = (
            f"{config.GEMINI_LIVE_WS_URL}"
            f"?key={config.GEMINI_API_KEY}"
        )
        self._ws = await websockets.connect(
            url,
            ping_interval=20,
            ping_timeout=10,
            max_size=16 * 1024 * 1024,
        )
        self._reader = asyncio.create_task(self._read_loop())
        await self._send(
            {
                "setup": {
                    "model": f"models/{config.GEMINI_LIVE_MODEL}",
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {
                                "voiceName": config.GEMINI_LIVE_VOICE,
                            }
                        }
                    },
                    "systemInstruction": {
                        "parts": [{"text": instructions}],
                    },
                    "inputAudioTranscription": {},
                    "outputAudioTranscription": {},
                    "realtimeInputConfig": {
                        "automaticActivityDetection": {
                            "silenceDurationMs": config.GEMINI_LIVE_SILENCE_MS,
                        }
                    },
                }
            }
        )
        await asyncio.wait_for(self._setup_done.wait(), timeout=8.0)
        if not self._setup_ok or self._ws is None:
            raise RuntimeError(
                "Gemini Live setup rejected "
                f"(model={config.GEMINI_LIVE_MODEL}). Check GEMINI_API_KEY and GEMINI_LIVE_MODEL."
            )
        logger.info(
            "Gemini Live ready conn=%s model=%s voice=%s",
            self._connection_id,
            config.GEMINI_LIVE_MODEL,
            config.GEMINI_LIVE_VOICE,
        )

    async def send_pcm16k(self, pcm: bytes) -> None:
        if not pcm or self._closed or not self.ready:
            return
        await self._send(
            {
                "realtimeInput": {
                    "audio": {
                        "data": base64.b64encode(pcm).decode("ascii"),
                        "mimeType": "audio/pcm;rate=16000",
                    }
                }
            }
        )

    async def say(self, instruction: str) -> None:
        """Ask the model to speak. Used for the greeting."""
        if self._closed or not self.ready:
            return
        self._turn_done.clear()
        await self._send(
            {
                "clientContent": {
                    "turns": [
                        {
                            "role": "user",
                            "parts": [{"text": instruction}],
                        }
                    ],
                    "turnComplete": True,
                }
            }
        )

    async def wait_turn(self, timeout: float) -> None:
        try:
            await asyncio.wait_for(self._turn_done.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "Gemini Live turn wait timed out conn=%s",
                self._connection_id,
            )

    async def close(self) -> None:
        self._closed = True
        self._setup_done.set()
        self._turn_done.set()
        if self._reader is not None:
            self._reader.cancel()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def _send(self, payload: dict) -> None:
        if self._ws is None:
            return
        await self._ws.send(json.dumps(payload))

    async def _read_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                if self._closed:
                    break
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                await self._handle(message)
        except asyncio.CancelledError:
            raise
        except websockets.exceptions.ConnectionClosed as exc:
            logger.error(
                "Gemini Live socket closed conn=%s code=%s reason=%s",
                self._connection_id,
                getattr(exc, "code", None),
                getattr(exc, "reason", None),
            )
        except Exception as exc:
            logger.error("Gemini Live reader failed conn=%s: %s", self._connection_id, exc)
        finally:
            self._setup_done.set()
            self._turn_done.set()

    async def _handle(self, message: dict) -> None:
        if "setupComplete" in message:
            self._setup_ok = True
            self._setup_done.set()
            return
        if "error" in message:
            logger.error(
                "Gemini Live error conn=%s payload=%s",
                self._connection_id,
                message,
            )
            return
        if "goAway" in message:
            logger.warning("Gemini Live goAway conn=%s", self._connection_id)
            return

        content = message.get("serverContent") or {}
        if content.get("interrupted"):
            self._turn_done.set()
            if self._on_interrupt is not None:
                await self._on_interrupt()
            return

        transcript_in = _transcript_text(content.get("inputTranscription"))
        if transcript_in and self._on_transcript is not None:
            await self._on_transcript("user", transcript_in)

        transcript_out = _transcript_text(content.get("outputTranscription"))
        if transcript_out and self._on_transcript is not None:
            await self._on_transcript("assistant", transcript_out)

        model_turn = content.get("modelTurn") or {}
        for part in model_turn.get("parts") or []:
            inline = part.get("inlineData") or part.get("inline_data") or {}
            data = inline.get("data")
            if not data:
                continue
            mime = (inline.get("mimeType") or inline.get("mime_type") or "").lower()
            if mime and "audio" not in mime:
                continue
            await self._on_audio(base64.b64decode(data))

        if content.get("turnComplete") or content.get("generationComplete"):
            self._turn_done.set()
            if self._on_turn_complete is not None and content.get("turnComplete"):
                await self._on_turn_complete()


def _transcript_text(node) -> str:
    if not node:
        return ""
    if isinstance(node, str):
        return node.strip()
    if isinstance(node, dict):
        return (node.get("text") or "").strip()
    return ""
