"""Vobiz WebSocket outbound commands (playAudio, checkpoint, clearAudio)."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Any, Protocol

from app import config

logger = logging.getLogger(__name__)

ULAW_FRAME_20MS = 160  # 20 ms μ-law @ 8 kHz


class TextSocket(Protocol):
    async def send_text(self, data: str) -> None: ...


class VobizStream:
    """Serialises writes to the Vobiz media WebSocket."""

    def __init__(self, ws: TextSocket) -> None:
        self.ws = ws
        self.stream_id: str | None = None
        self._lock = asyncio.Lock()
        self.closed = False
        self._play_bytes_sent = 0
        self._send_fail_logged = False
        self._first_play_logged = False

    async def _send_json(self, payload: str) -> None:
        """``websockets`` uses ``send``; Starlette/FastAPI uses ``send_text``."""
        ws = self.ws
        if hasattr(ws, "send_text"):
            await ws.send_text(payload)
            return
        if hasattr(ws, "send"):
            await ws.send(payload)
            return
        raise TypeError(f"Unsupported Vobiz WebSocket type: {type(ws)!r}")

    async def _send(self, msg: dict[str, Any]) -> None:
        if self.closed:
            return
        async with self._lock:
            try:
                await self._send_json(json.dumps(msg))
            except Exception as exc:
                if not self._send_fail_logged:
                    logger.error(
                        "Vobiz WebSocket send failed (no playAudio to caller): %s",
                        exc,
                    )
                    self._send_fail_logged = True
                self.closed = True

    async def play_ulaw(self, audio: bytes) -> None:
        if not audio or self.closed:
            return
        if not self.stream_id:
            logger.warning("Vobiz play_ulaw skipped — streamId not set yet")
            return
        chunk = max(ULAW_FRAME_20MS, config.VOBIZ_ULAW_PLAY_CHUNK)
        for i in range(0, len(audio), chunk):
            block = audio[i : i + chunk]
            await self._send(
                {
                    "event": "playAudio",
                    "streamId": self.stream_id,
                    "media": {
                        "contentType": "audio/x-mulaw",
                        "sampleRate": 8000,
                        "payload": base64.b64encode(block).decode("ascii"),
                    },
                }
            )
            self._play_bytes_sent += len(block)
            if not self._first_play_logged:
                self._first_play_logged = True
                logger.info(
                    "Vobiz playAudio started streamId=%s first_chunk_ulaw_bytes=%s",
                    self.stream_id,
                    len(block),
                )

    async def checkpoint(self, name: str) -> None:
        await self._send(
            {"event": "checkpoint", "streamId": self.stream_id, "name": name}
        )

    async def clear(self) -> None:
        await self._send({"event": "clearAudio", "streamId": self.stream_id})
