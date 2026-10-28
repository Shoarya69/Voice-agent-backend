"""Portal Gemini bootstrap, per-agent voice, caller language, and auth failure."""

from __future__ import annotations

import logging
import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from app import call_records, config, engine_config, lovable_client
from app.engine_config import EngineConfig, GeminiAuthError, gemini_voice_or_default, mask_secret
from app.voice_pipeline.factory import get_pipeline_name
from app.voice_pipeline.gemini_live.link import GeminiLiveLink
from app.voice_pipeline.tools import gemini_live_system_instructions


def _engine(key: str, model: str = "gemini-3.8-live", voice: str = "Aoede") -> EngineConfig:
    return EngineConfig(
        engine="gemini",
        gemini_api_key=key,
        gemini_live_model=model,
        gemini_default_voice=voice,
        ttl_seconds=60,
        fetched_at=0.0,
    )


class PortalGeminiEngineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        engine_config.reset_engine_config_for_tests()

    def tearDown(self) -> None:
        engine_config.reset_engine_config_for_tests()

    def test_elevenlabs_voice_id_falls_back_to_portal_default(self):
        self.assertEqual(
            gemini_voice_or_default("90ipbRoKi4CpHXvKVtl0", "Aoede"),
            "Aoede",
        )
        self.assertEqual(gemini_voice_or_default("Puck", "Aoede"), "Puck")

    def test_mask_secret_hides_the_middle(self):
        self.assertEqual(mask_secret("AIzaSECRETVALUE99"), "AIza••••UE99")
        self.assertNotIn("SECRET", mask_secret("AIzaSECRETVALUE99"))

    def test_pipeline_is_gemini_even_if_env_says_compound(self):
        previous = config.VOICE_PIPELINE
        config.VOICE_PIPELINE = "compound"
        try:
            self.assertEqual(get_pipeline_name(), "gemini_live")
        finally:
            config.VOICE_PIPELINE = previous

    def test_language_rule_follows_the_caller(self):
        text = gemini_live_system_instructions(
            "You sell CRM software.",
            language="Hindi",
            closing_message="Shukriya.",
        )
        self.assertIn("Reply in the caller's language", text)
        self.assertIn("Start the call in Hindi", text)
        self.assertIn("Shukriya.", text)
        self.assertNotIn("Never switch to English", text)
        self.assertNotIn("Always reply in Hindi/Hinglish", text)

    def test_agent_payload_uses_engine_voice_and_language(self):
        agent = lovable_client._parse_agent_config_payload(
            {
                "agent_id": "agent-1",
                "system_prompt": "You are Priya from Moon Tech Pro.",
                "first_message": "Namaste",
                "closing_message": "Bye",
                "voice_id": "legacy-voice",
                "language": "en",
                "engine": {"voice": "Puck", "language": "Hindi", "speed": 1.05},
            }
        )
        self.assertEqual(agent.gemini_voice, "Puck")
        self.assertEqual(agent.gemini_voice_name("Aoede"), "Puck")
        self.assertEqual(agent.language, "Hindi")
        self.assertEqual(agent.closing_message, "Bye")
        self.assertEqual(agent.system_prompt, "You are Priya from Moon Tech Pro.")

    async def test_bootstrap_cache_and_failed_refresh_keeps_last_key(self):
        first = _engine("AIzaFIRSTKEY0001", model="gemini-3.8-live", voice="Kore")
        fetches = {"n": 0}

        async def _fetch():
            fetches["n"] += 1
            if fetches["n"] == 1:
                return first
            raise RuntimeError("portal down")

        with patch.object(engine_config, "_fetch_bootstrap", new=_fetch), patch.object(
            engine_config, "_schedule_pool_refill"
        ):
            loaded = await engine_config.get_engine_config()
            self.assertEqual(loaded.gemini_api_key, "AIzaFIRSTKEY0001")
            again = await engine_config.get_engine_config()
            self.assertIs(again, loaded)
            self.assertEqual(fetches["n"], 1)

            engine_config._expires_at = 0.0
            kept = await engine_config.get_engine_config()
            self.assertEqual(kept.gemini_api_key, "AIzaFIRSTKEY0001")
            self.assertEqual(fetches["n"], 2)

    async def test_new_call_snapshot_does_not_change_when_portal_key_rotates(self):
        first = _engine("AIzaOLDKEYVALUE1")
        second = _engine("AIzaNEWKEYVALUE2", model="gemini-3.8-live", voice="Puck")
        calls = {"n": 0}

        async def _fetch():
            calls["n"] += 1
            return first if calls["n"] == 1 else second

        with patch.object(engine_config, "_fetch_bootstrap", new=_fetch), patch.object(
            engine_config, "_schedule_pool_refill"
        ):
            snapshot = await engine_config.get_engine_config()
            rotated = await engine_config.get_engine_config(force=True)
        self.assertEqual(snapshot.gemini_api_key, "AIzaOLDKEYVALUE1")
        self.assertEqual(rotated.gemini_api_key, "AIzaNEWKEYVALUE2")
        self.assertNotEqual(snapshot.gemini_api_key, rotated.gemini_api_key)

    async def test_auth_failure_refreshes_once_then_call_log_marks_gemini_auth(self):
        link = GeminiLiveLink("conn-auth", on_audio=AsyncMock())
        attempts = {"n": 0}

        async def _once(_instructions: str) -> None:
            attempts["n"] += 1
            raise GeminiAuthError("401")

        link._connect_once = _once  # type: ignore[method-assign]
        link._reset_socket = AsyncMock()  # type: ignore[method-assign]
        fresh = _engine("AIzaREFRESHEDKEY1", model="gemini-3.8-live", voice="Kore")

        with patch(
            "app.voice_pipeline.gemini_live.link.get_engine_config",
            new=AsyncMock(return_value=fresh),
        ):
            with self.assertRaises(GeminiAuthError):
                await link.connect(
                    "hello",
                    api_key="AIzaBADKEYVALUE01",
                    model="gemini-3.8-live",
                    voice="Kore",
                )
        self.assertEqual(attempts["n"], 2)
        self.assertEqual(link._api_key, "AIzaREFRESHEDKEY1")

        session = MagicMock()
        session.connection_id = "conn-auth"
        session._engine = fresh
        session.agent_config = None
        session._call_error_stage = ""
        session.call_sid = "call-1"
        session.call_started_at = datetime.now(timezone.utc)
        session.history = []
        session.agent_id = "agent-1"
        session.caller_from = ""
        session.caller_to = ""
        session.telephony_provider = "vobiz"
        session.call_direction = "inbound"
        session.wss_route_kind = None
        session.recording_url = None
        session.websocket = MagicMock()
        session.websocket.close = AsyncMock()

        async def _boom(*_args, **_kwargs):
            session._call_error_stage = "gemini_auth"
            raise GeminiAuthError("401")

        link.connect = _boom  # type: ignore[method-assign]
        with self.assertRaises(GeminiAuthError):
            await session_ensure(link, session)
        payload = call_records.build_call_log_payload(session, datetime.now(timezone.utc))
        self.assertEqual(payload["error_stage"], "gemini_auth")
        self.assertEqual(payload["status"], "failed")
        self.assertNotIn("AIza", str(payload))


async def session_ensure(link, session) -> None:
    from app.voice_pipeline.gemini_live.link import connect_session

    try:
        await connect_session(link, session)
    except GeminiAuthError:
        session._call_error_stage = "gemini_auth"
        raise


class _LogKeyGuard(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


class MaskedLogTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        engine_config.reset_engine_config_for_tests()

    def tearDown(self) -> None:
        engine_config.reset_engine_config_for_tests()

    async def test_bootstrap_log_does_not_contain_the_raw_key(self):
        secret = "AIzaSUPERSECRETKEY99"
        guard = _LogKeyGuard()
        logger = logging.getLogger("app.engine_config")
        logger.addHandler(guard)
        logger.setLevel(logging.INFO)
        try:
            with patch.object(
                engine_config,
                "_fetch_bootstrap",
                new=AsyncMock(return_value=_engine(secret)),
            ), patch.object(engine_config, "_schedule_pool_refill"):
                await engine_config.get_engine_config()
        finally:
            logger.removeHandler(guard)
        joined = "\n".join(guard.lines)
        self.assertNotIn(secret, joined)
        self.assertIn("AIza••••EY99", joined)


if __name__ == "__main__":
    unittest.main()
