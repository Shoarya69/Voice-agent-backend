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
from app.voice_pipeline.tools import END_CALL_TOOL_NAME, build_gemini_live_tools

logger = logging.getLogger(__name__)

AudioHandler = Callable[[bytes], Awaitable[None]]
TextHandler = Callable[[str, str], Awaitable[None]]
InterruptHandler = Callable[[], Awaitable[None]]
EndCallHandler = Callable[[], Awaitable[None]]


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
        on_end_call: EndCallHandler | None = None,
    ) -> None:
        self._connection_id = connection_id
        self._on_audio = on_audio
        self._on_transcript = on_transcript
        self._on_interrupt = on_interrupt
        self._on_turn_complete = on_turn_complete
        self._on_end_call = on_end_call
        self._ws = None
        self._reader: asyncio.Task | None = None
        self._setup_done = asyncio.Event()
        self._setup_ok = False
        self._turn_done = asyncio.Event()
        self._turn_done.set()
        self._connect_lock = asyncio.Lock()
        self._closed = False

    @property
    def ready(self) -> bool:
        return self._setup_done.is_set() and self._ws is not None

    async def open_socket(self) -> None:
        """Open TLS/WS to Gemini immediately (in parallel with Supabase lookup)."""
        if self._ws is not None or self._closed:
            return
        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not configured")
        from app.streaming import warm_pool

        borrowed = await warm_pool.borrow_gemini_socket()
        if borrowed is not None:
            self._ws = borrowed
            return
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

    async def _send_Setup(self, instructions: str) -> None:
        await self._send(
            {
                "setup": {
                    "model": f"models/{config.GEMINI_LIVE_MODEL}",
                    "generationConfig": {
                        "responseModalities": ["AUDIO"],
                        "speechConfig": {
                            "voiceConfig": {
                                "prebuiltVoiceConfig": {
                                    "voiceName": config.GEMINI_LIVE_VOICE,
                                }
                            }
                        },
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
                    "tools": build_gemini_live_tools(),
                }
            }
        )

    async def connect(self, instructions: str) -> None:
        if self.ready or self._closed:
            return
        async with self._connect_lock:
            if self.ready or self._closed:
                return
            await self.open_socket()
            if self._reader is None or self._reader.done():
                self._reader = asyncio.create_task(self._read_loop())
            await self._send_Setup(instructions)
            try:
                await asyncio.wait_for(self._setup_done.wait(), timeout=4.0)
            except asyncio.TimeoutError:
                pass
            if not self._setup_ok or self._ws is None:
                # Pooled socket may have gone stale — open one fresh socket and retry once.
                if self._reader is not None:
                    self._reader.cancel()
                try:
                    if self._ws is not None:
                        await self._ws.close()
                except Exception:
                    pass
                self._ws = None
                self._setup_done.clear()
                self._setup_ok = False
                url = f"{config.GEMINI_LIVE_WS_URL}?key={config.GEMINI_API_KEY}"
                self._ws = await websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=10,
                    max_size=16 * 1024 * 1024,
                )
                self._reader = asyncio.create_task(self._read_loop())
                await self._send_Setup(instructions)
                await asyncio.wait_for(self._setup_done.wait(), timeout=6.0)
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

        tool_call = message.get("toolCall")
        if tool_call:
            await self._handle_tool_call(tool_call)
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
            fn = part.get("functionCall") or part.get("function_call")
            if fn:
                await self._handle_function_call(fn)
                continue
            text_part = (part.get("text") or "").strip()
            if text_part and self._on_transcript is not None:
                await self._on_transcript("assistant", text_part)
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
            if self._on_turn_complete is not None:
                await self._on_turn_complete()

    async def _handle_tool_call(self, tool_call: dict) -> None:
        calls = tool_call.get("functionCalls") or tool_call.get("function_calls") or []
        if not calls:
            return
        responses: list[dict] = []
        for fn in calls:
            name = fn.get("name") or ""
            call_id = fn.get("id") or ""
            logger.info(
                "Gemini Live toolCall conn=%s name=%s id=%s",
                self._connection_id,
                name,
                call_id,
            )
            if name == END_CALL_TOOL_NAME and self._on_end_call is not None:
                await self._on_end_call()
            responses.append(
                {
                    "id": call_id,
                    "name": name,
                    "response": {"result": "ok"},
                }
            )
        if responses:
            await self._send({"toolResponse": {"functionResponses": responses}})
        self._turn_done.set()
        if self._on_turn_complete is not None:
            await self._on_turn_complete()

    async def _handle_function_call(self, fn: dict) -> None:
        name = fn.get("name") or ""
        if name != END_CALL_TOOL_NAME or self._on_end_call is None:
            return
        await self._on_end_call()
        call_id = fn.get("id") or ""
        if call_id:
            await self._send(
                {
                    "toolResponse": {
                        "functionResponses": [
                            {
                                "id": call_id,
                                "name": name,
                                "response": {"result": "ok"},
                            }
                        ]
                    }
                }
            )


def _transcript_text(node) -> str:
    if not node:
        return ""
    if isinstance(node, str):
        return node.strip()
    if isinstance(node, dict):
        return (node.get("text") or "").strip()
    return ""
