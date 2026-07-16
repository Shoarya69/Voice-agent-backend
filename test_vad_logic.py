#!/usr/bin/env python3
"""Small smoke tests for the VAD/noise-floor endpointing helpers."""

import math
import struct

from app import config
from app.voice_session import VoiceSession


class DummyWebSocket:
    async def send(self, _message):
        return None


def make_sine_frame(amplitude=4000, frequency=440):
    samples = []
    for index in range(int(config.SAMPLE_RATE * config.CHUNK_DURATION_MS / 1000)):
        sample = int(amplitude * math.sin(2 * math.pi * frequency * index / config.SAMPLE_RATE))
        samples.append(struct.pack("<h", sample))
    return b"".join(samples)


def main():
    session = VoiceSession("test_vad", DummyWebSocket())

    silence = b"\x00\x00" * int(config.SAMPLE_RATE * config.CHUNK_DURATION_MS / 1000)
    silence_decision = session._classify_frame(silence)
    assert not silence_decision.accepted_as_speech, silence_decision

    # A pure tone is not a perfect speech sample, but it verifies that the
    # classifier executes on a valid 8kHz/20ms frame without frame-size errors.
    tone = make_sine_frame()
    tone_decision = session._classify_frame(tone)
    assert tone_decision.rms > silence_decision.rms

    print("VAD logic smoke test passed")


if __name__ == "__main__":
    main()
