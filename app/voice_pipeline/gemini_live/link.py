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
from app.engine_config import (
    GeminiAuthError,
    GeminiVoiceError,
    gemini_voice_or_default,
    get_engine_config,
    is_gemini_auth_failure,
    is_gemini_voice_rejection,
    mask_secret,
)
from app.voice_pipeline.tools import END_CALL_TOOL_NAME, build_gemini_live_tools

logger = logging.getLogger(__name__)


def gemini_speak_message(instruction: str) -> dict:
    """Ask a native-audio Live model to speak.

    ``clientContent`` is stored as conversation context on
    gemini-2.5-flash-native-audio and the turn completes with no audio.
    ``realtimeInput.text`` is a complete user turn and is what produces speech.
    """
    return {"realtimeInput": {"text": instruction}}


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
        self._api_key = ""
        self._model = ""
        self._voice = ""
        self._auth_retried = False
        self._auth_failed = False
        self._voice_retried = False
        self._voice_rejected = False
        self._last_close_code: int | None = None
        self._last_close_reason = ""
        self._expect_speech = False
        self._audio_chunks_this_turn = 0

    @property
    def ready(self) -> bool:
        return self._setup_done.is_set() and self._ws is not None

    def rebind_handlers(
        self,
        *,
        on_audio: AudioHandler,
        on_transcript: TextHandler | None = None,
        on_interrupt: InterruptHandler | None = None,
        on_turn_complete: Callable[[], Awaitable[None]] | None = None,
        on_end_call: EndCallHandler | None = None,
    ) -> None:
        """Attach an already-warmed Gemini Live link to the live call session."""
        self._on_audio = on_audio
        self._on_transcript = on_transcript
        self._on_interrupt = on_interrupt
        self._on_turn_complete = on_turn_complete
        self._on_end_call = on_end_call

    async def open_socket(self) -> None:
        """Open TLS/WS to Gemini immediately (in parallel with Supabase lookup)."""
        if self._ws is not None or self._closed:
            return
        engine = await get_engine_config()
        if not self._api_key:
            self._api_key = engine.gemini_api_key
        if not self._model:
            self._model = engine.gemini_live_model
        if not self._voice:
            self._voice = engine.gemini_default_voice
        if not self._api_key:
            raise GeminiAuthError("Gemini API key is not configured")
        from app.streaming import warm_pool

        borrowed = await warm_pool.borrow_gemini_socket(expected_key=self._api_key)
        if borrowed is not None:
            self._ws = borrowed
            return
        self._ws = await self._connect_ws(self._api_key)

    async def _connect_ws(self, api_key: str):
        url = f"{config.GEMINI_LIVE_WS_URL}?key={api_key}"
        headers = {"x-goog-api-key": api_key}
        try:
            return await websockets.connect(
                url,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=10,
                max_size=16 * 1024 * 1024,
            )
        except Exception as exc:
            if is_gemini_auth_failure(exc):
                raise GeminiAuthError("Gemini WebSocket rejected the API key") from exc
            raise

    async def _send_Setup(self, instructions: str, *, disable_thinking: bool = True) -> None:
        model = self._model
        if model.startswith("models/"):
            model = model[len("models/") :]
        gen_cfg: dict = {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {
                        "voiceName": self._voice or "Kore",
                    }
                }
            },
        }
        # Only set thinkingConfig for models that support thinking (e.g. Gemini 3.x / thinking models)
        if disable_thinking and ("3" in model or "thinking" in model.lower()):
            gen_cfg["thinkingConfig"] = {"thinkingBudget": 0}

        setup_msg: dict = {
            "model": f"models/{model}",
            "generationConfig": gen_cfg,
            "systemInstruction": {
                "parts": [{"text": instructions}],
            },
            "realtimeInputConfig": {
                "automaticActivityDetection": {
                    "silenceDurationMs": config.GEMINI_LIVE_SILENCE_MS,
                    "prefixPaddingMs": config.GEMINI_LIVE_PREFIX_PADDING_MS,
                },
                "activityHandling": "START_OF_ACTIVITY_INTERRUPTS",
            },
            "tools": build_gemini_live_tools(),
        }

        await self._send({"setup": setup_msg})

    async def connect(
        self,
        instructions: str,
        *,
        api_key: str = "",
        model: str = "",
        voice: str = "",
    ) -> None:
        if api_key:
            self._api_key = api_key
        if model:
            self._model = model
        if voice:
            self._voice = voice
        default_voice = ""
        try:
            engine = await get_engine_config()
            default_voice = engine.gemini_default_voice
            if not self._api_key:
                self._api_key = engine.gemini_api_key
            if not self._model:
                self._model = engine.gemini_live_model
        except Exception:
            default_voice = default_voice or "Aoede"
        chosen = gemini_voice_or_default(self._voice, default_voice or "Aoede")
        if chosen != self._voice:
            logger.warning(
                "Gemini voice fallback conn=%s requested=%s using=%s",
                self._connection_id,
                (self._voice or "")[:8],
                chosen,
            )
        self._voice = chosen
        try:
            await self._connect_once(instructions)
        except GeminiVoiceError:
            if self._voice_retried or self._closed:
                raise
            self._voice_retried = True
            fallback = gemini_voice_or_default("", default_voice or "Aoede")
            logger.warning(
                "Gemini rejected voice conn=%s model=%s; retrying with default %s",
                self._connection_id,
                self._model,
                fallback,
            )
            self._voice = fallback
            await self._reset_socket()
            await self._connect_once(instructions)
        except GeminiAuthError:
            if self._auth_retried or self._closed:
                raise
            self._auth_retried = True
            logger.warning(
                "Gemini auth failed conn=%s model=%s — refreshing portal bootstrap once",
                self._connection_id,
                self._model,
            )
            fresh = await get_engine_config(force=True)
            self._api_key = fresh.gemini_api_key
            self._model = fresh.gemini_live_model
            await self._reset_socket()
            await self._connect_once(instructions)

    async def _reset_socket(self) -> None:
        if self._reader is not None:
            self._reader.cancel()
            self._reader = None
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        self._setup_done.clear()
        self._setup_ok = False
        self._auth_failed = False
        self._voice_rejected = False
        self._last_close_code = None
        self._last_close_reason = ""

    async def _connect_once(self, instructions: str) -> None:
        if self.ready or self._closed:
            return
        async with self._connect_lock:
            if self.ready or self._closed:
                return
            try:
                await self.open_socket()
                if self._reader is None or self._reader.done():
                    self._reader = asyncio.create_task(self._read_loop())
                await self._send_Setup(instructions)
                try:
                    await asyncio.wait_for(self._setup_done.wait(), timeout=4.0)
                except asyncio.TimeoutError:
                    pass
                if self._voice_rejected:
                    raise GeminiVoiceError(
                        self._last_close_reason or "Gemini Live rejected the voice"
                    )
                if self._auth_failed:
                    raise GeminiAuthError(
                        self._last_close_reason or "Gemini Live rejected the API key"
                    )
                if not self._setup_ok or self._ws is None:
                    await self._reset_socket()
                    self._ws = await self._connect_ws(self._api_key)
                    self._reader = asyncio.create_task(self._read_loop())
                    await self._send_Setup(instructions, disable_thinking=False)
                    await asyncio.wait_for(self._setup_done.wait(), timeout=6.0)
            except GeminiVoiceError:
                raise
            except GeminiAuthError:
                raise
            except Exception as exc:
                if is_gemini_auth_failure(exc):
                    raise GeminiAuthError("Gemini Live setup rejected the API key") from exc
                raise
            if self._voice_rejected:
                raise GeminiVoiceError(
                    self._last_close_reason or "Gemini Live rejected the voice"
                )
            if self._auth_failed:
                raise GeminiAuthError(
                    self._last_close_reason or "Gemini Live rejected the API key"
                )
            if not self._setup_ok or self._ws is None:
                raise RuntimeError(
                    "Gemini Live setup rejected "
                    f"(model={self._model})."
                )
            logger.info(
                "Gemini Live ready conn=%s model=%s voice=%s key=%s",
                self._connection_id,
                self._model,
                self._voice,
                mask_secret(self._api_key),
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
        """Ask the model to speak. Used for the greeting and live replies."""
        if self._closed or not self.ready:
            return
        text = (instruction or "").strip()
        if not text:
            return
        self._turn_done.clear()
        self._expect_speech = True
        self._audio_chunks_this_turn = 0
        await self._send(gemini_speak_message(text))

    async def inject_assistant_greeting_context(self, greeting_text: str) -> None:
        """Record an already-played cached PCM greeting into Gemini Live context without speaking."""
        if self._closed or not self.ready or not (greeting_text or "").strip():
            return
        await self._send(
            {
                "clientContent": {
                    "turns": [
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "text": (
                                        "[System Note: The customer just answered the phone and "
                                        f"heard your greeting: \"{greeting_text.strip()}\". "
                                        "Do NOT repeat the greeting. Respond naturally to whatever they say next.]"
                                    )
                                }
                            ],
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
            self._last_close_code = getattr(exc, "code", None)
            try:
                self._last_close_code = int(self._last_close_code) if self._last_close_code is not None else None
            except (TypeError, ValueError):
                self._last_close_code = None
            self._last_close_reason = str(getattr(exc, "reason", "") or "")
            if is_gemini_voice_rejection(self._last_close_reason, self._last_close_code):
                self._voice_rejected = True
            elif is_gemini_auth_failure(exc):
                self._auth_failed = True
            logger.error(
                "Gemini Live socket closed conn=%s code=%s reason=%s",
                self._connection_id,
                getattr(exc, "code", None),
                self._last_close_reason[:180],
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
            if part.get("thought"):
                continue
            text_part = (part.get("text") or "").strip()
            if text_part:
                if (
                    text_part.startswith("**")
                    or text_part.startswith("Crafting")
                    or "registered the user" in text_part
                    or "formulate" in text_part
                ):
                    continue
                if self._on_transcript is not None:
                    await self._on_transcript("assistant", text_part)
            inline = part.get("inlineData") or part.get("inline_data") or {}
            data = inline.get("data")
            if not data:
                continue
            mime = (inline.get("mimeType") or inline.get("mime_type") or "").lower()
            if mime and "audio" not in mime:
                continue
            try:
                pcm = base64.b64decode(data)
            except Exception:
                logger.warning(
                    "Gemini Live audio decode failed conn=%s",
                    self._connection_id,
                )
                continue
            if not pcm:
                continue
            self._audio_chunks_this_turn += 1
            self._expect_speech = False
            await self._on_audio(pcm)

        if content.get("turnComplete") or content.get("generationComplete"):
            if self._expect_speech and self._audio_chunks_this_turn == 0:
                logger.warning(
                    "Gemini Live turn had no audio conn=%s content_keys=%s",
                    self._connection_id,
                    ",".join(sorted(str(k) for k in content.keys())),
                )
                self._expect_speech = False
            self._audio_chunks_this_turn = 0
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



async def connect_session(link: GeminiLiveLink, session) -> None:
    """Open Gemini for one call using that call's engine snapshot and agent voice."""
    engine = getattr(session, "_engine", None)
    if engine is None:
        engine = await get_engine_config()
        session._engine = engine
    agent = getattr(session, "agent_config", None)
    if agent is not None and (getattr(agent, "system_prompt", "") or "").strip():
        from app.voice_pipeline.gemini_live.prewarm import build_gemini_instructions_for_agent

        instructions = build_gemini_instructions_for_agent(agent)
        voice = agent.gemini_voice_name(engine.gemini_default_voice)
    else:
        from app.openai_service import openai_brain
        from app.voice_pipeline.tools import gemini_live_system_instructions

        base = openai_brain._messages([])[0]["content"]
        instructions = gemini_live_system_instructions(base)
        voice = engine.gemini_default_voice
    try:
        await link.connect(
            instructions,
            api_key=engine.gemini_api_key,
            model=engine.gemini_live_model,
            voice=voice,
        )
    except GeminiAuthError:
        session._call_error_stage = "gemini_auth"
        logger.error(
            "Gemini auth failed conn=%s model=%s key=%s",
            getattr(session, "connection_id", ""),
            engine.gemini_live_model,
            engine.masked_key(),
        )
        raise


def _transcript_text(node) -> str:
    if not node:
        return ""
    if isinstance(node, str):
        return node.strip()
    if isinstance(node, dict):
        return (node.get("text") or "").strip()
    return ""
