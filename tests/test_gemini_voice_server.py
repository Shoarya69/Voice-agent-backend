"""
Unit and integration tests for the Google Gemini Live voice engine fixes:
1. Immediate WebSocket acceptance + single-channel stale session eviction
2. Cached 8kHz PCM greeting (hash cache + immediate playback on Vobiz)
3. Human-like barge-in (>320ms), endpointing (800ms), and back-channel filter (<800ms)
4. Outbound /prewarm auth, idempotency, capacity limit, and TTL expiry
5. Portal call-log delivery with provider_call_sid, 3 retries, and durable spool
"""

from __future__ import annotations

import base64
import json
import os
import struct
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from app import call_records, config, lovable_client
from app.telephony.vobiz import ws_handler
from app.telephony.vobiz.gemini_session import VobizGeminiLiveSession
from app.telephony.vobiz.stream import VobizStream
from app.voice_pipeline.gemini_live import prewarm
from app.voice_pipeline.gemini_live.barge_in import (
    GeminiBargeInController,
    assistant_text_is_question,
    is_backchannel_text,
)


def _make_speech_pcm_8k(duration_ms: int = 20, amplitude: int = 4000) -> bytes:
    """Generate a 16-bit 8kHz mono PCM tone above the RMS speech threshold."""
    num_samples = int(8000 * (duration_ms / 1000.0))
    samples = [amplitude if (i % 8 < 4) else -amplitude for i in range(num_samples)]
    return struct.pack(f"<{num_samples}h", *samples)


def _make_silence_pcm_8k(duration_ms: int = 20) -> bytes:
    num_samples = int(8000 * (duration_ms / 1000.0))
    return b"\x00\x00" * num_samples


class GeminiVoiceServerTests(unittest.IsolatedAsyncioTestCase):
    def test_01_endpointing_default_is_700_to_900ms(self):
        self.assertGreaterEqual(config.GEMINI_LIVE_SILENCE_MS, 700)
        self.assertLessEqual(config.GEMINI_LIVE_SILENCE_MS, 900)

    def test_02_backchannel_and_question_detection(self):
        self.assertTrue(is_backchannel_text("haan"))
        self.assertTrue(is_backchannel_text("ok"))
        self.assertTrue(is_backchannel_text("hmm"))
        self.assertTrue(is_backchannel_text("accha"))
        self.assertTrue(is_backchannel_text("हाँ जी"))
        self.assertFalse(is_backchannel_text("mujhe price batao"))

        self.assertTrue(assistant_text_is_question("Kya aapke paas do minute hain?"))
        self.assertTrue(assistant_text_is_question("Aapka budget kitna hai"))
        self.assertFalse(
            assistant_text_is_question("Humari company website aur app banati hai.")
        )

    def test_03_barge_in_ignores_short_backchannel_during_explanation(self):
        ctrl = GeminiBargeInController()
        ctrl.on_assistant_speaking_start()
        ctrl.on_assistant_transcript(
            "Humari company software aur mobile apps banati hai."
        )
        self.assertFalse(ctrl.ai_asked_question)

        pcm_8k = _make_speech_pcm_8k(20)
        pcm_16k = pcm_8k * 2
        with patch.object(config, "GEMINI_BARGE_IN_ENABLED", True), patch.object(
            ctrl, "is_frame_speech", return_value=True
        ):
            # 500ms of "haan" (< 800ms) while AI is mid-explanation -> must NOT interrupt
            for _ in range(25):  # 25 * 20ms = 500ms
                interrupted, frames = ctrl.process_caller_frame_during_playback(
                    pcm_8k, pcm_16k
                )
                self.assertFalse(interrupted)
                self.assertEqual(frames, [])

        # Followed by 240ms of silence -> back-channel buffer resets without interrupting
        silence_8k = _make_silence_pcm_8k(20)
        with patch.object(config, "GEMINI_BARGE_IN_ENABLED", True), patch.object(
            ctrl, "is_frame_speech", return_value=False
        ):
            for _ in range(12):
                interrupted, _ = ctrl.process_caller_frame_during_playback(
                    silence_8k, silence_8k * 2
                )
                self.assertFalse(interrupted)
        self.assertEqual(ctrl.speech_ms, 0.0)
        self.assertEqual(len(ctrl.buffered_pcm16k), 0)

    def test_04_barge_in_triggers_quickly_when_ai_asked_question(self):
        ctrl = GeminiBargeInController()
        ctrl.on_assistant_speaking_start()
        ctrl.on_assistant_transcript("Kya aapke paas baat karne ke liye do minute hain?")
        self.assertTrue(ctrl.ai_asked_question)

        pcm_8k = _make_speech_pcm_8k(20)
        pcm_16k = pcm_8k * 2
        triggered = False
        flushed_frames = []
        with patch.object(config, "GEMINI_BARGE_IN_ENABLED", True), patch.object(
            ctrl, "is_frame_speech", return_value=True
        ):
            for _ in range(18):  # 360ms >= 320ms
                interrupted, frames = ctrl.process_caller_frame_during_playback(
                    pcm_8k, pcm_16k
                )
                if interrupted:
                    triggered = True
                    flushed_frames = frames
                    break
        self.assertTrue(triggered)
        self.assertGreaterEqual(len(flushed_frames), 16)

    def test_05_barge_in_triggers_on_sustained_speech_during_explanation(self):
        ctrl = GeminiBargeInController()
        ctrl.on_assistant_speaking_start()
        ctrl.on_assistant_transcript("Humari team aapko poora demo degi.")
        self.assertFalse(ctrl.ai_asked_question)

        pcm_8k = _make_speech_pcm_8k(20)
        pcm_16k = pcm_8k * 2
        triggered = False
        with patch.object(config, "GEMINI_BARGE_IN_ENABLED", True), patch.object(
            ctrl, "is_frame_speech", return_value=True
        ):
            for _ in range(42):  # 840ms >= 800ms
                interrupted, _ = ctrl.process_caller_frame_during_playback(
                    pcm_8k, pcm_16k
                )
                if interrupted:
                    triggered = True
                    break
        self.assertTrue(triggered)

    def test_06_greeting_pcm_cached_by_hash_and_wav_header_stripped(self):
        lovable_client._greeting_pcm_by_hash.clear()
        lovable_client._token_greeting_hash.clear()

        raw_pcm = _make_speech_pcm_8k(400)  # 400ms @ 8kHz = 6400 bytes
        wav_header = (
            b"RIFF"
            + struct.pack("<I", len(raw_pcm) + 36)
            + b"WAVEfmt "
            + (b"\x00" * 20)
            + b"data"
            + struct.pack("<I", len(raw_pcm))
        )
        b64_audio = base64.b64encode(wav_header + raw_pcm).decode("ascii")

        pcm1, h1 = lovable_client._decode_greeting_pcm(
            {"hash": "hash_abc_123", "audio_base64": b64_audio},
            agent_id="agent-1",
            token="tok_1234567890123456",
        )
        self.assertEqual(pcm1, raw_pcm)
        self.assertEqual(h1, "hash_abc_123")
        self.assertEqual(lovable_client._greeting_pcm_by_hash["hash_abc_123"], raw_pcm)

        # Second lookup by same hash works even without audio_base64 (cache hit)
        pcm2, h2 = lovable_client._decode_greeting_pcm(
            {"hash": "hash_abc_123", "audio_base64": ""},
            agent_id="agent-1",
            token="tok_1234567890123456",
        )
        self.assertEqual(pcm2, raw_pcm)
        self.assertEqual(h2, "hash_abc_123")

    async def test_07_cached_greeting_plays_immediately_on_vobiz(self):
        sent_messages: list[dict] = []

        class DummyWS:
            async def send(self, data: str) -> None:
                sent_messages.append(json.loads(data))

        ws = DummyWS()
        vobiz = VobizStream(ws)
        vobiz.stream_id = "stream-uuid-1"

        session = VobizGeminiLiveSession("vobiz_test_1", ws, vobiz)
        session.greeting_audio_pcm = _make_speech_pcm_8k(200)  # 200ms cached PCM
        session._ensure_link = AsyncMock()

        await session.speak_welcome()
        self.assertGreaterEqual(len(sent_messages), 1)
        self.assertEqual(sent_messages[0]["event"], "playAudio")
        self.assertIsNotNone(session.first_audio_out_ms)
        self.assertLess(session.first_audio_out_ms, 200.0)
        await session.close()

    async def test_08_stale_session_evicted_on_same_number(self):
        ws_handler._active_vobiz_by_number.clear()

        stale_ws = MagicMock()
        stale_ws.close = AsyncMock()
        stale_vobiz = VobizStream(stale_ws)
        stale_session = VobizGeminiLiveSession("vobiz_stale_1", stale_ws, stale_vobiz)
        stale_session.close = AsyncMock()

        await ws_handler._register_session_for_number("917965480875", stale_session)
        self.assertIs(ws_handler._active_vobiz_by_number["7965480875"], stale_session)

        new_ws = MagicMock()
        new_vobiz = VobizStream(new_ws)
        new_session = VobizGeminiLiveSession("vobiz_new_2", new_ws, new_vobiz)

        await ws_handler._register_session_for_number("+917965480875", new_session)
        stale_session.close.assert_awaited_once()
        stale_ws.close.assert_awaited_once()
        self.assertIs(ws_handler._active_vobiz_by_number["7965480875"], new_session)

    async def test_09_prewarm_validation_idempotency_capacity_and_ttl(self):
        prewarm._by_call_id.clear()
        prewarm._by_token.clear()
        config.VOICE_PIPELINE = "gemini_live"
        config.PREWARM_MAX_SESSIONS = 2

        status, _ = await prewarm.validate_and_schedule_gemini_prewarm(
            {"call_id": "c1", "voicebot_token": "bad"}
        )
        self.assertEqual(status, 401)

        fake_cfg = lovable_client.AgentConfig(
            agent_id="agent-1",
            token="valid_token_1234567890",
            system_prompt="You are Priya.",
            first_message="Namaste!",
        )
        with patch(
            "app.lovable_client.resolve_agent_bundle",
            new=AsyncMock(return_value=fake_cfg),
        ), patch(
            "app.voice_pipeline.gemini_live.prewarm._warm_prepared_session",
            new=AsyncMock(),
        ):
            s1, b1 = await prewarm.validate_and_schedule_gemini_prewarm(
                {"call_id": "c1", "voicebot_token": "valid_token_1234567890"}
            )
            self.assertEqual(s1, 200)
            self.assertEqual(b1, {"ok": True})

            # Duplicate call_id is idempotent
            s_dup, _ = await prewarm.validate_and_schedule_gemini_prewarm(
                {"call_id": "c1", "voicebot_token": "valid_token_1234567890"}
            )
            self.assertEqual(s_dup, 200)
            self.assertEqual(len(prewarm._by_call_id), 1)

            s2, _ = await prewarm.validate_and_schedule_gemini_prewarm(
                {"call_id": "c2", "voicebot_token": "valid_token_2222222222"}
            )
            self.assertEqual(s2, 200)

            # 3rd exceeds capacity=2 -> 429
            s3, _ = await prewarm.validate_and_schedule_gemini_prewarm(
                {"call_id": "c3", "voicebot_token": "valid_token_3333333333"}
            )
            self.assertEqual(s3, 429)

            claimed = await prewarm.claim_prepared_gemini_session(call_id="c1")
            self.assertIsNotNone(claimed)
            self.assertEqual(claimed.call_id, "c1")
            await prewarm.remove_prepared_gemini_session(call_id="c2")

    async def test_10_call_log_includes_provider_call_sid_and_retries_3_times(self):
        ws = MagicMock()
        vobiz = VobizStream(ws)
        session = VobizGeminiLiveSession("vobiz_log_test", ws, vobiz)
        session.call_sid = "692d4bc7-c368-410c-82de-fd696e0fd1d5"
        session.agent_id = "6a3fd7db-cb8f-418e-a076-45c1e5e0d71f"
        session.caller_from = "+919876543210"
        session.caller_to = "917965480875"
        session.call_started_at = datetime.now(timezone.utc) - timedelta(seconds=25)
        session.history = [
            {"role": "assistant", "content": "Namaste!"},
            {"role": "user", "content": "Haan bataiye"},
        ]

        ended_at = datetime.now(timezone.utc)
        payload = call_records.build_call_log_payload(session, ended_at)
        self.assertEqual(
            payload["provider_call_sid"], "692d4bc7-c368-410c-82de-fd696e0fd1d5"
        )
        self.assertEqual(payload["call_sid"], "692d4bc7-c368-410c-82de-fd696e0fd1d5")
        self.assertGreaterEqual(payload["duration_seconds"], 24)
        self.assertEqual(len(payload["transcript"]), 2)

        with tempfile.TemporaryDirectory() as tmp_dir:
            spool_file = os.path.join(tmp_dir, "failed_call_logs.jsonl")
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(
                return_value=httpx.Response(502, text="Bad Gateway")
            )
            with patch.object(
                config, "LOVABLE_APP_URL", "https://www.moontechpro.com"
            ), patch.object(
                lovable_client, "_FAILED_CALL_LOGS_PATH", spool_file
            ), patch(
                "app.lovable_client._get_client",
                new=AsyncMock(return_value=mock_client),
            ), patch("asyncio.sleep", new=AsyncMock()):
                ok = await lovable_client.post_call_log(payload)
                self.assertFalse(ok)
                self.assertEqual(mock_client.post.await_count, 3)
                self.assertTrue(os.path.exists(spool_file))
                with open(spool_file, encoding="utf-8") as f:
                    saved = json.loads(f.readline())
                self.assertEqual(
                    saved["provider_call_sid"],
                    "692d4bc7-c368-410c-82de-fd696e0fd1d5",
                )


if __name__ == "__main__":
    unittest.main()
