"""Vobiz media session: μ-law I/O with the compound VoiceSession pipeline."""

from __future__ import annotations

import asyncio
import logging
from typing import ClassVar

from app import config
from app.audio_utils import (
    decode_payload,
    encode_payload,
    pcm_bytes_to_ulaw,
    scale_pcm_volume,
    ulaw_bytes_to_pcm,
)
from app.telephony.vobiz.stream import VobizStream
from app.voice_pipeline.compound.session import CompoundPipelineSession

logger = logging.getLogger(__name__)


class VobizCompoundSession(CompoundPipelineSession):
    """ElevenLabs + OpenAI pipeline with Vobiz playAudio outbound and μ-law inbound."""

    pipeline_name: ClassVar[str] = "compound_vobiz"

    def __init__(self, connection_id: str, websocket, vobiz: VobizStream) -> None:
        super().__init__(connection_id, websocket)
        self.telephony_provider = "vobiz"
        self._vobiz = vobiz
        self._ulaw_out_buffer = bytearray()
        self._playback_turn = 0

    async def add_audio_chunk(self, media_data: dict) -> None:
        payload = media_data.get("payload", "")
        ulaw = decode_payload(payload)
        if not ulaw:
            return
        pcm = ulaw_bytes_to_pcm(ulaw)
        await super().add_audio_chunk({"payload": encode_payload(pcm)})

    async def send_pcm_frame(self, pcm_frame: bytes) -> bool:
        if self._closed or self._vobiz.closed:
            return False
        pcm_frame = scale_pcm_volume(pcm_frame, config.TTS_VOLUME)
        self._ulaw_out_buffer.extend(pcm_bytes_to_ulaw(pcm_frame))
        chunk = max(160, config.VOBIZ_ULAW_PLAY_CHUNK)
        while len(self._ulaw_out_buffer) >= chunk:
            block = bytes(self._ulaw_out_buffer[:chunk])
            del self._ulaw_out_buffer[:chunk]
            await self._vobiz.play_ulaw(block)
        return True

    async def _flush_ulaw_out_buffer(self) -> None:
        if self._ulaw_out_buffer:
            await self._vobiz.play_ulaw(bytes(self._ulaw_out_buffer))
            self._ulaw_out_buffer.clear()

    async def _send_pcm_frame(self, pcm_frame: bytes) -> bool:
        return await self.send_pcm_frame(pcm_frame)

    def _arm_echo_guard(self) -> None:
        super()._arm_echo_guard()
        asyncio.create_task(self._vobiz_finish_playback())

    async def _vobiz_finish_playback(self) -> None:
        if self._vobiz.closed:
            return
        try:
            await self._flush_ulaw_out_buffer()
            self._playback_turn += 1
            await self._vobiz.checkpoint(f"utt-{self._playback_turn}")
        except Exception as exc:
            logger.debug(
                "Vobiz playback finish skipped for %s: %s",
                self.connection_id,
                exc,
            )

    async def _send_clear(self) -> None:
        self._ulaw_out_buffer.clear()
        await self._vobiz.clear()

    async def handle_clear(self) -> None:
        await super().handle_clear()

    async def close(self) -> None:
        self._ulaw_out_buffer.clear()
        self._vobiz.closed = True
        await super().close()
