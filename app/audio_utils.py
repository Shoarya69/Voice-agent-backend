"""
Small audio helper utilities shared by the STT/TTS services and the voice
session. Exotel voice streaming sends/expects raw PCM (signed 16-bit
little-endian, mono, 8kHz) audio encoded as base64 - no WAV/container header.
"""

import base64
import io
import math
import struct

from app import config


def decode_payload(payload_b64: str) -> bytes:
    """Decode a base64 Exotel media payload into raw PCM bytes."""
    if not payload_b64:
        return b""
    try:
        return base64.b64decode(payload_b64)
    except Exception:
        return b""


def encode_payload(pcm_bytes: bytes) -> str:
    """Encode raw PCM bytes into a base64 string suitable for an Exotel media event."""
    return base64.b64encode(pcm_bytes).decode("ascii")


def pcm_to_wav_bytes(
    pcm_bytes: bytes,
    sample_rate: int = config.SAMPLE_RATE,
    sample_width: int = config.SAMPLE_WIDTH,
    channels: int = config.CHANNELS,
) -> bytes:
    """Wrap raw headerless PCM data in a minimal WAV container.

    Speech-to-text APIs (including ElevenLabs) expect a proper audio file
    with header, not headerless raw PCM, so we build one in-memory.
    """
    byte_rate = sample_rate * channels * sample_width
    block_align = channels * sample_width
    data_size = len(pcm_bytes)

    buffer = io.BytesIO()
    buffer.write(b"RIFF")
    buffer.write(struct.pack("<I", 36 + data_size))
    buffer.write(b"WAVE")
    buffer.write(b"fmt ")
    buffer.write(struct.pack("<I", 16))  # PCM fmt chunk size
    buffer.write(struct.pack("<H", 1))  # PCM format
    buffer.write(struct.pack("<H", channels))
    buffer.write(struct.pack("<I", sample_rate))
    buffer.write(struct.pack("<I", byte_rate))
    buffer.write(struct.pack("<H", block_align))
    buffer.write(struct.pack("<H", sample_width * 8))
    buffer.write(b"data")
    buffer.write(struct.pack("<I", data_size))
    buffer.write(pcm_bytes)
    return buffer.getvalue()


def chunk_pcm(pcm_bytes: bytes, chunk_size: int = config.BYTES_PER_CHUNK):
    """Yield successive fixed-size slices of PCM audio (pads the final chunk with silence)."""
    if not pcm_bytes:
        return
    total = len(pcm_bytes)
    for start in range(0, total, chunk_size):
        piece = pcm_bytes[start:start + chunk_size]
        if len(piece) < chunk_size:
            piece = piece + b"\x00" * (chunk_size - len(piece))
        yield piece


def iter_pcm_frames(pcm_bytes: bytes, frame_size: int = config.BYTES_PER_CHUNK):
    """Yield complete PCM frames suitable for WebRTC VAD without padding."""
    if not pcm_bytes:
        return
    usable = len(pcm_bytes) - (len(pcm_bytes) % frame_size)
    for start in range(0, usable, frame_size):
        yield pcm_bytes[start:start + frame_size]


def pcm_duration_ms(pcm_bytes: bytes) -> float:
    """Return the duration in milliseconds of a raw PCM buffer."""
    bytes_per_ms = config.SAMPLE_RATE * config.SAMPLE_WIDTH * config.CHANNELS / 1000
    if bytes_per_ms == 0:
        return 0.0
    return len(pcm_bytes) / bytes_per_ms


def pcm_rms(pcm_bytes: bytes) -> float:
    """Return RMS amplitude for signed 16-bit little-endian PCM audio."""
    if not pcm_bytes:
        return 0.0

    # Exotel raw/slin is 16-bit PCM, so trim any incomplete final sample.
    sample_bytes = pcm_bytes[: len(pcm_bytes) - (len(pcm_bytes) % 2)]
    if not sample_bytes:
        return 0.0

    sample_count = len(sample_bytes) // 2
    total = 0
    for (sample,) in struct.iter_unpack("<h", sample_bytes):
        total += sample * sample
    return math.sqrt(total / sample_count)


def scale_pcm_volume(pcm_bytes: bytes, volume: float) -> bytes:
    """Scale signed 16-bit PCM volume and clamp samples to avoid clipping noise."""
    if not pcm_bytes or volume == 1.0:
        return pcm_bytes

    sample_bytes = pcm_bytes[: len(pcm_bytes) - (len(pcm_bytes) % 2)]
    remainder = pcm_bytes[len(sample_bytes):]
    output = bytearray()

    for (sample,) in struct.iter_unpack("<h", sample_bytes):
        scaled = int(sample * volume)
        scaled = max(-32768, min(32767, scaled))
        output.extend(struct.pack("<h", scaled))

    output.extend(remainder)
    return bytes(output)
