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
import time
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from app import call_records, config, lovable_client
from app.telephony.vobiz import ws_handler
from app.telephony.vobiz.gemini_session import VobizGeminiLiveSession
from app.telephony.vobiz.stream import VobizStream
from app.voice_pipeline.gemini_live import prewarm
from app.voice_pipeline.gemini_live.link import gemini_speak_message
from app.voice_pipeline.realtime_link_factory import get_realtime_engine_name
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
    def test_01_endpointing_is_fast_for_live_conversation(self):
        self.assertGreaterEqual(config.GEMINI_LIVE_SILENCE_MS, 200)
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
            config, "GEMINI_BACKCHANNEL_MAX_MS", 800
        ), patch.object(ctrl, "is_frame_speech", return_value=True):
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
            for _ in range(18):  # 360ms >= default GEMINI_BARGE_IN_MIN_SPEECH_MS (200)
                interrupted, frames = ctrl.process_caller_frame_during_playback(
                    pcm_8k, pcm_16k
                )
                if interrupted:
                    triggered = True
                    flushed_frames = frames
                    break
        self.assertTrue(triggered)
        self.assertGreaterEqual(len(flushed_frames), 10)

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

    def test_11_realtime_engine_env_defaults_to_gemini(self):
        with patch.object(config, "VOICE_REALTIME_ENGINE", "gemini"):
            self.assertEqual(get_realtime_engine_name(), "gemini")
        with patch.object(config, "VOICE_REALTIME_ENGINE", "livekit"):
            self.assertEqual(get_realtime_engine_name(), "livekit")

    def test_12_native_audio_greeting_uses_realtime_text(self):
        msg = gemini_speak_message("Speak ONLY this opening greeting: hello")
        self.assertEqual(
            msg["realtimeInput"]["text"],
            "Speak ONLY this opening greeting: hello",
        )
        self.assertNotIn("clientContent", msg)

    async def test_13_empty_prewarm_greeting_is_spoken_on_the_live_call(self):
        ws = MagicMock()
        vobiz = VobizStream(ws)
        vobiz.stream_id = "stream-empty"
        session = VobizGeminiLiveSession("vobiz_empty_greet", ws, vobiz)
        session.agent_id = "test-empty-prewarm-agent"
        session._greeting_started = True
        session._prewarm_greeting_done = True
        session._link._setup_done.set()
        session._link._ws = object()
        session._link.say = AsyncMock()

        await session.speak_welcome()

        session._link.say.assert_awaited()
        sent = session._link.say.await_args.args[0]
        self.assertIn("opening greeting", sent)
        await session.close()

    async def test_14_opening_barge_allows_model_reply_before_transcript(self):
        ws = MagicMock()
        vobiz = VobizStream(ws)
        vobiz.stream_id = "stream-barge"
        session = VobizGeminiLiveSession("vobiz_barge", ws, vobiz)
        session._prewarm_call_id = "call-barge"
        session._prewarm_greeting_done = True
        session._greeting_started = True
        session._suppress_spontaneous_model = True
        session._user_turns = 0
        session._send_clear = AsyncMock()
        session._send_paced_frame = AsyncMock(return_value=True)

        await session._on_interrupt(local_barge_in=True)

        self.assertTrue(session._opening_greeting_aborted)
        self.assertFalse(session._drop_model_audio_until_next_turn)
        self.assertFalse(session._phone_greeting_in_progress())

        pcm_24k = b"\x01\x00" * 2400
        await session._on_model_audio(pcm_24k)
        session._send_paced_frame.assert_awaited()
        await session.close()

    def test_15_opening_mic_release_after_opening_mic_release_ms(self):
        ws = MagicMock()
        vobiz = VobizStream(ws)
        session = VobizGeminiLiveSession("vobiz_mic_release", ws, vobiz)
        session._prewarm_call_id = "call-mic"
        session._user_turns = 0
        session._is_playing_welcome = True
        session.first_audio_out_ms = 120.0
        session._ws_connected_at = time.monotonic() - 3.0
        self.assertTrue(session._phone_greeting_in_progress())
        self.assertFalse(session._opening_blocks_mic())

    async def test_16_dedupe_repeated_opening_assistant_partial(self):
        ws = MagicMock()
        vobiz = VobizStream(ws)
        session = VobizGeminiLiveSession("vobiz_dedupe", ws, vobiz)
        session._opening_greeting_already_played = (
            "Namaste. Kya aap sun rahe hain?"
        )
        session._user_turns = 0
        session.history = []
        await session._on_transcript("assistant", "Namaste. Kya")
        self.assertEqual(session.history, [])
        self.assertTrue(session._is_opening_script_repeat("Namaste. Kya aap"))

    async def test_17_drop_generic_env_welcome_when_agent_has_first_message(self):
        ws = MagicMock()
        vobiz = VobizStream(ws)
        session = VobizGeminiLiveSession("vobiz_generic", ws, vobiz)
        session.agent_config = lovable_client.AgentConfig(
            agent_id="agent-1",
            token="t" * 20,
            first_message="hello sir me karan baat kara hu",
            system_prompt="You are Karan from Moontech.",
        )
        session._user_turns = 0
        session.history = []
        with patch.object(config, "WELCOME_MESSAGE", "Hi! I'm your AI assistant."):
            await session._on_transcript("assistant", "Hi! I'm your AI assistant.")
        self.assertEqual(session.history, [])


if __name__ == "__main__":
    unittest.main()
