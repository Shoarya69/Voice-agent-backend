"""Vobiz bidirectional Stream WebSocket handler (/ws by default)."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from urllib.parse import urlparse

import websockets.exceptions

from app import call_records, config
from app.agent_setup import start_agent_prefetch
from app.runtime import (
    decrement_active_sessions,
    increment_active_sessions,
    release_session_slot,
    try_acquire_session,
)
from app.agent_setup import load_agent_bundle_into_session
from app.telephony.vobiz.metadata import extract_call_sid, extract_parties, extract_stream_id
from app.telephony.vobiz.session import VobizCompoundSession
from app.telephony.vobiz.stream import VobizStream

logger = logging.getLogger(__name__)


def is_vobiz_websocket_path(path: str | None) -> bool:
    if not config.VOBIZ_ENABLED:
        return False
    if not path:
        return False
    parsed = urlparse(path)
    route = parsed.path.rstrip("/") or "/"
    expected = config.VOBIZ_WS_PATH.rstrip("/") or "/ws"
    if route == expected:
        return True
    if route.startswith(f"{expected}/"):
        return True
    return False


async def handle_vobiz_websocket(websocket) -> None:
    """
    Vobiz media stream: μ-law inbound, playAudio outbound.
    Agent resolution matches Exotel (token or number in URL).
    """
    # Import here to avoid circular import with ai_server at module load time.
    import ai_server as voice_server

    connection_id = f"vobiz_{int(datetime.now(timezone.utc).timestamp() * 1000)}"
    end_reason = "websocket_close"
    session: VobizCompoundSession | None = None

    if not await try_acquire_session():
        try:
            await websocket.close(code=1013, reason="server at capacity")
        except Exception:
            pass
        return

    try:
        active = await increment_active_sessions()
        logger.info(
            "🔗 Vobiz WebSocket connected: %s (active=%s/%s)",
            connection_id,
            active,
            config.MAX_CONCURRENT_SESSIONS,
        )

        vobiz = VobizStream(websocket)
        session = VobizCompoundSession(connection_id, websocket, vobiz)
        session.telephony_provider = "vobiz"
        session.wss_route_kind, session.wss_route_value, request_path = (
            voice_server._extract_wss_route(websocket)
        )
        session.lovable_token = (
            session.wss_route_value if session.wss_route_kind == "token" else None
        )
        logger.info(
            "🔎 Vobiz WS path=%r route=%s",
            request_path,
            session.wss_route_kind or "none",
        )

        session.agent_id = None
        session.agent_config = None
        session.call_started_at = None
        session.caller_from = None
        session.caller_to = None
        session._call_log_posted = False
        session._agent_prefetch_task = None
        session._call_finalized = False

        voice_server._assume_channel_reserved(session)
        voice_server._log_channel_reserved(session, connection_id)
        start_agent_prefetch(session, connection_id)

        # Load agent before `start` so Vobiz hears playAudio within ~1–2s (not after 6s HTTP).
        agent_preloaded = False
        if session.wss_route_kind in ("token", "number") and session.wss_route_value:
            logger.info(
                "⏳ Vobiz early agent prefetch conn=%s route=%s",
                connection_id,
                session.wss_route_kind,
            )
            agent_preloaded = await voice_server._await_agent_prefetch(
                session, connection_id
            )
            if agent_preloaded:
                voice_server._commit_agent_config_to_session(session)
                await session.warm_playback_pipeline()
                if session._streaming.enabled:
                    asyncio.create_task(session._streaming.ensure_stt_if_needed())

        agent_started = False

        async for message in websocket:
            try:
                data = json.loads(message)
            except json.JSONDecodeError as exc:
                logger.error("❌ Invalid Vobiz JSON from %s: %s", connection_id, exc)
                continue

            event_type = data.get("event")

            if event_type == "start":
                start_data = data.get("start") or data
                if not isinstance(start_data, dict):
                    start_data = data
                vobiz.stream_id = extract_stream_id(data, start_data)
                session.stream_sid = vobiz.stream_id
                session.call_sid = extract_call_sid(data, start_data)
                session.call_started_at = datetime.now(timezone.utc)
                session.caller_from, session.caller_to = extract_parties(data, start_data)
                call_records.apply_start_metadata(session, start_data)
                logger.info(
                    "🚀 Vobiz START %s call_sid=%s stream=%s from=%s to=%s "
                    "route=%s/%s direction=%s start_keys=%s",
                    connection_id,
                    session.call_sid,
                    session.stream_sid,
                    session.caller_from,
                    session.caller_to,
                    session.wss_route_kind,
                    (session.wss_route_value or "")[:12],
                    session.call_direction,
                    sorted(start_data.keys()),
                )
                if not vobiz.stream_id:
                    logger.error(
                        "Vobiz start missing streamId conn=%s — playAudio will fail",
                        connection_id,
                    )
                voice_server._log_call_started(session, connection_id)

                if agent_preloaded and session.agent_config:
                    agent_resolved = True
                else:
                    agent_resolved = await voice_server._resolve_agent_on_start(
                        session, connection_id, start_data
                    )
                if not agent_resolved:
                    logger.error(
                        "Vobiz agent lookup failed (conn=%s path=%r to=%s)",
                        connection_id,
                        request_path,
                        session.caller_to,
                    )
                    await voice_server._speak_fallback_and_close(session, connection_id)
                    end_reason = "agent_lookup_failed"
                    break

                if not agent_preloaded:
                    voice_server._commit_agent_config_to_session(session)
                call_records.schedule_call_started(session)
                await session.on_agent_ready()
                await session.speak_welcome()
                agent_started = True

            elif event_type == "media":
                if not agent_started:
                    continue
                media = data.get("media") or {}
                await session.add_audio_chunk({"payload": media.get("payload", "")})

            elif event_type == "playedStream":
                logger.debug(
                    "Vobiz playedStream %s name=%s",
                    connection_id,
                    data.get("name"),
                )

            elif event_type == "clearedAudio":
                logger.info("🧹 Vobiz clearedAudio %s", connection_id)

            else:
                logger.debug(
                    "Vobiz event %s from %s keys=%s",
                    event_type,
                    connection_id,
                    list(data.keys()),
                )

    except websockets.exceptions.ConnectionClosed:
        end_reason = "connection_closed"
        logger.info("👋 Vobiz connection closed: %s", connection_id)
    except Exception as exc:
        end_reason = "exception"
        logger.exception("❌ Vobiz connection error for %s: %s", connection_id, exc)
    finally:
        if session is not None:
            await voice_server._finalize_call_session(session, connection_id, end_reason)
            await session.close()
        remaining = await decrement_active_sessions()
        release_session_slot()
        logger.info(
            "🔚 Vobiz connection ended: %s (active=%s/%s)",
            connection_id,
            remaining,
            config.MAX_CONCURRENT_SESSIONS,
        )
