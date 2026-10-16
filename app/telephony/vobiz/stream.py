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

    async def _send(self, msg: dict[str, Any]) -> None:
        if self.closed:
            return
        async with self._lock:
            try:
                await self.ws.send_text(json.dumps(msg))
            except Exception as exc:
                logger.debug("Vobiz send failed: %s", exc)
                self.closed = True

    async def play_ulaw(self, audio: bytes) -> None:
        chunk = max(ULAW_FRAME_20MS, config.VOBIZ_ULAW_PLAY_CHUNK)
        for i in range(0, len(audio), chunk):
            await self._send(
                {
                    "event": "playAudio",
                    "streamId": self.stream_id,
                    "media": {
                        "contentType": "audio/x-mulaw",
                        "sampleRate": 8000,
                        "payload": base64.b64encode(audio[i : i + chunk]).decode("ascii"),
                    },
                }
            )

    async def checkpoint(self, name: str) -> None:
        await self._send(
            {"event": "checkpoint", "streamId": self.stream_id, "name": name}
        )

    async def clear(self) -> None:
        await self._send({"event": "clearAudio", "streamId": self.stream_id})
