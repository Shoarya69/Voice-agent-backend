"""
Gemini Live through LiveKit Agents ``livekit.plugins.google.realtime``.

Same phone contract as ``GeminiLiveLink``: 16 kHz PCM in, 24 kHz PCM out callbacks.
Use when ``VOICE_REALTIME_ENGINE=livekit`` in ``.env``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

from app import config
from app.engine_config import (
    GeminiAuthError,
    GeminiVoiceError,
    gemini_voice_or_default,
    get_engine_config,
    mask_secret,
)

logger = logging.getLogger(__name__)

AudioHandler = Callable[[bytes], Awaitable[None]]
TextHandler = Callable[[str, str], Awaitable[None]]
InterruptHandler = Callable[[], Awaitable[None]]
EndCallHandler = Callable[[], Awaitable[None]]


def _livekit_imports():
    from livekit import rtc
    from livekit.agents import llm
    from livekit.plugins import google
    from google.genai import types

    return rtc, llm, google, types


class LiveKitGeminiLink:
    """Bidirectional Gemini session using LiveKit's Google Realtime plugin."""

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
        self._model = None
        self._session = None
        self._setup_done = asyncio.Event()
        self._setup_ok = False
        self._turn_done = asyncio.Event()
        self._turn_done.set()
        self._connect_lock = asyncio.Lock()
        self._closed = False
        self._api_key = ""
        self._model_name = ""
        self._voice = ""
        self._consume_tasks: set[asyncio.Task] = set()

    @property
    def ready(self) -> bool:
        return self._setup_ok and self._session is not None

    def rebind_handlers(
        self,
        *,
        on_audio: AudioHandler | None = None,
        on_transcript: TextHandler | None = None,
        on_interrupt: InterruptHandler | None = None,
        on_turn_complete: Callable[[], Awaitable[None]] | None = None,
        on_end_call: EndCallHandler | None = None,
    ) -> None:
        if on_audio is not None:
            self._on_audio = on_audio
        if on_transcript is not None:
            self._on_transcript = on_transcript
        if on_interrupt is not None:
            self._on_interrupt = on_interrupt
        if on_turn_complete is not None:
            self._on_turn_complete = on_turn_complete
        if on_end_call is not None:
            self._on_end_call = on_end_call

    async def open_socket(self) -> None:
        """No-op — LiveKit session opens on ``connect``."""
        return

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
            self._model_name = model
        if voice:
            self._voice = voice
        default_voice = "Aoede"
        try:
            engine = await get_engine_config()
            default_voice = engine.gemini_default_voice
            if not self._api_key:
                self._api_key = engine.gemini_api_key
            if not self._model_name:
                self._model_name = engine.gemini_live_model
        except Exception:
            pass
        if not self._api_key:
            self._api_key = config.GEMINI_API_KEY
        if not self._model_name:
            self._model_name = config.LIVEKIT_GEMINI_MODEL or config.GEMINI_LIVE_MODEL

        chosen = gemini_voice_or_default(self._voice, default_voice or "Aoede")
        self._voice = chosen

        async with self._connect_lock:
            if self.ready or self._closed:
                return
            await self._start_session(instructions)

    async def _start_session(self, instructions: str) -> None:
        if not self._api_key:
            raise GeminiAuthError("No Google/Gemini API key for LiveKit engine")

        try:
            rtc, llm, google, types = _livekit_imports()
        except ImportError as exc:
            raise RuntimeError(
                "LiveKit engine requires: pip install 'livekit-agents[google]~=1.8'"
            ) from exc

        silence_ms = config.GEMINI_LIVE_SILENCE_MS
        prefix_ms = config.GEMINI_LIVE_PREFIX_PADDING_MS
        realtime_input_config = types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                silence_duration_ms=silence_ms,
                prefix_padding_ms=prefix_ms,
            ),
        )
        model_name = (config.LIVEKIT_GEMINI_MODEL or self._model_name).strip()
        if model_name.startswith("models/"):
            model_name = model_name[len("models/") :]

        try:
            self._model = google.realtime.RealtimeModel(
                instructions=instructions,
                api_key=self._api_key,
                model=model_name,
                voice=self._voice,
                realtime_input_config=realtime_input_config,
            )
            self._session = self._model.session()
        except Exception as exc:
            msg = str(exc).lower()
            if "api key" in msg or "401" in msg or "403" in msg:
                raise GeminiAuthError("LiveKit Google Realtime rejected the API key") from exc
            if "voice" in msg:
                raise GeminiVoiceError(str(exc)) from exc
            raise

        self._session.on("generation_created", self._on_generation_created)
        self._session.on(
            "input_audio_transcription_completed",
            self._on_input_transcription,
        )
        self._session.on("input_speech_started", self._on_input_speech_started)
        self._session.on("error", self._on_session_error)

        self._setup_ok = True
        self._setup_done.set()
        logger.info(
            "LiveKit Gemini ready conn=%s model=%s voice=%s key=%s",
            self._connection_id,
            model_name,
            self._voice,
            mask_secret(self._api_key),
        )

    def _on_input_speech_started(self, _ev) -> None:
        if self._on_interrupt is not None:
            asyncio.create_task(self._on_interrupt())

    def _on_session_error(self, err) -> None:
        logger.error(
            "LiveKit Gemini error conn=%s recoverable=%s err=%s",
            self._connection_id,
            getattr(err, "recoverable", False),
            getattr(err, "error", err),
        )

    def _on_input_transcription(self, ev) -> None:
        if not ev.is_final or not self._on_transcript:
            return
        text = (ev.transcript or "").strip()
        if text:
            asyncio.create_task(self._on_transcript("user", text))

    def _on_generation_created(self, ev) -> None:
        task = asyncio.create_task(self._consume_generation(ev))
        self._consume_tasks.add(task)
        task.add_done_callback(self._consume_tasks.discard)

    async def _consume_generation(self, ev) -> None:
        try:
            async for msg in ev.message_stream:
                await asyncio.gather(
                    self._consume_text(msg),
                    self._consume_audio(msg),
                )
        finally:
            self._turn_done.set()
            if self._on_turn_complete is not None:
                await self._on_turn_complete()

    async def _consume_text(self, msg) -> None:
        if not self._on_transcript:
            return
        try:
            async for text in msg.text_stream:
                cleaned = (text or "").strip()
                if not cleaned:
                    continue
                if cleaned.startswith("**") or cleaned.startswith("Crafting"):
                    continue
                await self._on_transcript("assistant", cleaned)
        except Exception as exc:
            logger.debug("LiveKit text stream ended conn=%s: %s", self._connection_id, exc)

    async def _consume_audio(self, msg) -> None:
        try:
            async for frame in msg.audio_stream:
                pcm = bytes(frame.data)
                if pcm:
                    await self._on_audio(pcm)
        except Exception as exc:
            logger.debug("LiveKit audio stream ended conn=%s: %s", self._connection_id, exc)

    async def send_pcm16k(self, pcm: bytes) -> None:
        if not pcm or self._closed or not self.ready:
            return
        rtc, _, _, _ = _livekit_imports()
        samples = len(pcm) // 2
        if samples <= 0:
            return
        frame = rtc.AudioFrame(pcm, 16000, 1, samples)
        self._session.push_audio(frame)

    async def say(self, instruction: str) -> None:
        if self._closed or not self.ready:
            return
        text = (instruction or "").strip()
        if not text:
            return
        self._turn_done.clear()
        self._session.generate_reply(instructions=text)

    async def inject_assistant_greeting_context(self, greeting_text: str) -> None:
        if self._closed or not self.ready or not (greeting_text or "").strip():
            return
        note = (
            "[System Note: The customer just answered and heard your greeting: "
            f"\"{greeting_text.strip()}\". Do NOT repeat it. Respond to what they say next.]"
        )
        self._session.generate_reply(instructions=note)

    async def wait_turn(self, timeout: float) -> None:
        try:
            await asyncio.wait_for(self._turn_done.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "LiveKit Gemini turn wait timed out conn=%s",
                self._connection_id,
            )

    async def close(self) -> None:
        self._closed = True
        self._setup_done.set()
        self._turn_done.set()
        for task in list(self._consume_tasks):
            task.cancel()
        if self._session is not None:
            try:
                await self._session.aclose()
            except Exception:
                pass
            self._session = None
        self._model = None
        self._setup_ok = False
