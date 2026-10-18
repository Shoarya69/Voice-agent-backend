#!/usr/bin/env python3
"""
AI Voice Agent Server for Exotel
=================================

A WebSocket server that plugs into Exotel's bidirectional voice streaming
(Voicebot Applet) and turns it into a real AI phone agent:

    Caller audio -> ElevenLabs (Speech-to-Text)
                 -> OpenAI      (the "brain": decides what to say)
                 -> ElevenLabs (Text-to-Speech)
                 -> back to the caller as Exotel media events

Usage:
    python3 ai_server.py

Configuration is done entirely through environment variables - copy
`.env.example` to `.env` and fill in your API keys before running.
"""

import asyncio
import http
import json
import logging
import os
import time
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse

import websockets

from app import call_records, config, lovable_client, webhook_server
from app.agent_setup import load_agent_bundle_into_session, start_agent_prefetch
from app.elevenlabs_service import elevenlabs_service
from app.openai_service import openai_brain
from app.runtime import (
    active_sessions_count,
    decrement_active_sessions,
    increment_active_sessions,
    release_session_slot,
    shutdown_runtime,
    try_acquire_session,
)
from app.voice_pipeline import VoicePipelineSession, create_voice_pipeline_session, get_pipeline_name

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/ai_voice_agent.log"),
    ],
)
logger = logging.getLogger(__name__)

# This WSS port is exposed directly to the internet (for Exotel's media
# stream), which means it's constantly probed by port scanners, uptime
# monitors and vulnerability bots sending things that are not a WebSocket
# handshake at all - plain HTTP GETs, WebDAV PROPFIND requests, garbage
# bytes with no CRLF, etc. The `websockets` library logs every single one of
# these as an ERROR with a full traceback, which is just internet background
# noise on any publicly exposed port, not an application problem - it has
# nothing to do with Exotel or real calls. Anything that fails before a
# request line can even be parsed (bad methods, malformed/non-HTTP bytes)
# can't be intercepted in `_process_request` below since it happens earlier
# in the library, so the only way to keep logs clean is to raise this
# logger's level.
logging.getLogger("websockets.server").setLevel(logging.CRITICAL)

active_sessions: dict[str, VoicePipelineSession] = {}


async def _process_request(connection, request):
    """
    Answer plain (non-WebSocket) HTTP requests with a normal 200 OK instead
    of letting the handshake fail.

    Exotel always connects with a real WebSocket upgrade (Upgrade/Connection
    headers set). Anything hitting this port without those headers - a
    load-balancer/uptime-monitor health probe, someone opening the URL in a
    browser, a stray `curl` - is not trying to start a call, so there's no
    reason to reject it with a 426 and log a scary "opening handshake
    failed: missing Upgrade header" traceback. Returning a plain response
    here (via `connection.respond`, as recommended by the websockets docs)
    answers it cleanly and skips the WebSocket handshake/error path
    entirely.
    """
    if request.headers.get("Upgrade", "").lower() != "websocket":
        return connection.respond(
            http.HTTPStatus.OK,
            "AI Voice Agent WebSocket server is running.\n",
        )
    return None


def _extract_request_path(websocket) -> str | None:
    request = getattr(websocket, "request", None)
    if request is not None:
        path = getattr(request, "path", None)
        if path:
            return path
    return getattr(websocket, "path", None)


def _extract_request_headers(websocket) -> dict:
    request = getattr(websocket, "request", None)
    headers = getattr(request, "headers", None) if request is not None else None
    if headers is None:
        headers = getattr(websocket, "request_headers", None)
    if not headers:
        return {}
    try:
        # websockets' Headers object supports both dict-like access and
        # raw_items(); normalize to a plain lowercase-keyed dict.
        items = headers.raw_items() if hasattr(headers, "raw_items") else headers.items()
        return {str(k).lower(): str(v) for k, v in items}
    except Exception:
        return {}


_TOKEN_PATH_IGNORE_SEGMENTS = {"exotel", "voicebot", "ws", "stream", "media", "number"}


def _extract_wss_route(websocket) -> tuple[str | None, str | None, str | None]:
    """
    Detect how this WSS connection identifies the agent.

    Returns (route_kind, value, raw_path):
      - ('number', '<mobile>', path) for .../voicebot/number/<mobile>
      - ('token', '<token>', path) for ?token=... or .../voicebot/<token>
    """
    path = _extract_request_path(websocket)
    parsed = urlparse(path) if path else None
    query = parsed.query if parsed else ""

    if not query:
        headers = _extract_request_headers(websocket)
        for header_name in ("x-original-uri", "x-forwarded-uri", "x-original-url"):
            header_value = headers.get(header_name)
            if header_value and "?" in header_value:
                query = urlparse(header_value).query
                break

    if parsed is not None:
        segments = [seg for seg in parsed.path.split("/") if seg]
        for index, segment in enumerate(segments):
            if segment.lower() == "number" and index + 1 < len(segments):
                mobile = segments[index + 1]
                if mobile:
                    return "number", mobile, path

    values = parse_qs(query).get("token") if query else None
    if values and values[0]:
        return "token", values[0], path

    number_values = parse_qs(query).get("number") if query else None
    if number_values and number_values[0]:
        return "number", number_values[0], path

    if parsed is not None:
        segments = [seg for seg in parsed.path.split("/") if seg]
        if segments and segments[-1].lower() not in _TOKEN_PATH_IGNORE_SEGMENTS:
            candidate = segments[-1]
            if len(candidate) >= 8:
                return "token", candidate, path

    return None, None, path


def _extract_token(websocket) -> tuple[str | None, str | None]:
    """
    Pull the per-agent token from the incoming WSS request, trying (in order):

      1. ?token=... in the query string (preferred).
      2. An X-Original-URI/X-Forwarded-Uri header carrying the original
         query string (some reverse proxies forward it that way instead of
         preserving it on the request line).
      3. The last path segment, e.g. .../exotel/voicebot/<token> - path
         segments are essentially never stripped by proxies, so this is a
         reliable fallback for setups where the query string keeps getting
         dropped before it reaches this server.

    Returns (token, raw_path). raw_path is returned even on failure so the
    caller can log exactly what the server received, to tell apart "Exotel
    never sent a token" from "a reverse proxy stripped it".
    """
    path = _extract_request_path(websocket)
    parsed = urlparse(path) if path else None
    query = parsed.query if parsed else ""

    if not query:
        headers = _extract_request_headers(websocket)
        for header_name in ("x-original-uri", "x-forwarded-uri", "x-original-url"):
            header_value = headers.get(header_name)
            if header_value and "?" in header_value:
                query = urlparse(header_value).query
                break

    token = None
    values = parse_qs(query).get("token") if query else None
    if values:
        token = values[0]

    if not token and parsed is not None:
        segments = [seg for seg in parsed.path.split("/") if seg]
        if segments and segments[-1].lower() not in _TOKEN_PATH_IGNORE_SEGMENTS:
            candidate = segments[-1]
            if len(candidate) >= 8:
                token = candidate

    return token, path


def _extract_call_id(websocket) -> str | None:
    """Extract ?call_id=... (or callId / call_sid) from the WebSocket request URL."""
    path = _extract_request_path(websocket)
    parsed = urlparse(path) if path else None
    query = parsed.query if parsed else ""

    if not query:
        headers = _extract_request_headers(websocket)
        for header_name in ("x-original-uri", "x-forwarded-uri", "x-original-url"):
            header_value = headers.get(header_name)
            if header_value and "?" in header_value:
                query = urlparse(header_value).query
                break

    if not query:
        return None
    qs = parse_qs(query)
    for key in ("call_id", "callId", "call_sid", "callSid"):
        values = qs.get(key)
        if values and values[0]:
            return values[0].strip()
    return None


def _token_prefix(token: str | None) -> str:
    return (token or "")[:6]


def _commit_agent_config_to_session(session: VoicePipelineSession) -> None:
    """Apply a resolved session.agent_config as per-call config overrides."""
    agent_config = getattr(session, "agent_config", None)
    if agent_config is not None:
        config.set_agent_overrides(agent_config.as_overrides())


async def _await_agent_prefetch(session: VoicePipelineSession, connection_id: str) -> bool:
    """
    Wait for bundle prefetch started at WSS connect, then apply overrides.
    When prewarm populated the cache during ring, this returns immediately.
    """
    task = getattr(session, "_agent_prefetch_task", None)
    prefetch_started = getattr(session, "_prefetch_started_at", None)

    if task is None:
        kind = session.wss_route_kind
        value = session.wss_route_value
        if kind == "token" and value:
            ok = await load_agent_bundle_into_session(
                session, connection_id, token=value
            )
        elif kind == "number" and value:
            ok = await load_agent_bundle_into_session(
                session, connection_id, number=value
            )
        else:
            return False
        if ok:
            _commit_agent_config_to_session(session)
        return ok

    was_pending = not task.done()
    if not was_pending:
        elapsed_ms = (
            (time.perf_counter() - prefetch_started) * 1000
            if prefetch_started is not None
            else 0
        )
        logger.info(
            "✅ Agent bundle ready for %s (cache/prewarm hit, elapsed_ms=%.0f)",
            connection_id,
            elapsed_ms,
        )
    else:
        wait_start = time.perf_counter()
        logger.info("⏳ Waiting for agent bundle for %s", connection_id)

    try:
        resolved = await task
    except Exception as exc:  # noqa: BLE001 - prefetch must not crash the call handler
        logger.error("❌ Agent prefetch task failed for %s: %s", connection_id, exc)
        return False

    if was_pending:
        logger.info(
            "⏳ Agent bundle loaded for %s after wait_ms=%.0f",
            connection_id,
            (time.perf_counter() - wait_start) * 1000,
        )

    if resolved:
        _commit_agent_config_to_session(session)
    return resolved


async def _speak_fallback_and_close(session: VoicePipelineSession, connection_id: str):
    """Speak the .env WELCOME_MESSAGE (no agent overrides applied) then close."""
    try:
        await session.speak_welcome()
        response_task = getattr(session, "_response_task", None)
        if response_task is not None:
            await response_task
    except websockets.exceptions.ConnectionClosed:
        logger.info(
            "WebSocket closed while speaking fallback greeting for %s",
            connection_id,
        )
    except Exception as exc:
        logger.error("Error speaking fallback greeting for %s: %s", connection_id, exc)
    try:
        await session.websocket.close(code=1011, reason="agent lookup failed")
    except Exception:
        pass


def _post_call_log_once(session: VoicePipelineSession) -> None:
    """Post the call log exactly once (Supabase + Lovable)."""
    call_records.schedule_finalize_call(session)


async def _resolve_agent_on_start(
    session: VoicePipelineSession,
    connection_id: str,
    start_data: dict,
) -> bool:
    """
    Resolve agent from WSS token/number route, or inbound fallback to dialed ``to``.
    """
    if session.wss_route_kind == "token" and session.wss_route_value:
        ok = await _await_agent_prefetch(session, connection_id)
        if ok:
            return True
        return await load_agent_bundle_into_session(
            session,
            connection_id,
            token=session.wss_route_value,
        )

    if session.wss_route_kind == "number" and session.wss_route_value:
        ok = await _await_agent_prefetch(session, connection_id)
        if ok:
            return True
        return await load_agent_bundle_into_session(
            session,
            connection_id,
            number=session.wss_route_value,
        )

    fallback_number = (
        start_data.get("to")
        or start_data.get("To")
        or session.caller_to
    )
    if fallback_number:
        logger.info(
            "Inbound fallback agent lookup by to=...%s conn=%s",
            str(fallback_number)[-4:],
            connection_id,
        )
        session.wss_route_kind = "number"
        session.wss_route_value = str(fallback_number)
        return await load_agent_bundle_into_session(
            session,
            connection_id,
            number=str(fallback_number),
        )

    return False


def _session_release_token(session: VoicePipelineSession) -> str:
    """Best per-call token for channel reservation logging."""
    token = getattr(session, "lovable_token", None)
    if token:
        return token
    if session.wss_route_kind == "token" and session.wss_route_value:
        return session.wss_route_value
    agent_config = getattr(session, "agent_config", None)
    if agent_config and agent_config.token:
        return agent_config.token
    return ""


def _assume_channel_reserved(session: VoicePipelineSession) -> None:
    """
    Supabase reserves a channel when the call is accepted (try_reserve_voice_channel).
    Any WSS connection with a token or number route implies a reservation.
    """
    if getattr(session, "_channel_reserved", False):
        return
    if session.wss_route_kind in ("token", "number") and session.wss_route_value:
        session._channel_reserved = True


def _log_channel_reserved(session: VoicePipelineSession, connection_id: str) -> None:
    if getattr(session, "_channel_reserved_logged", False):
        return
    if not getattr(session, "_channel_reserved", False):
        return

    session._channel_reserved_logged = True
    agent_config = getattr(session, "agent_config", None)
    reservation_id = ""
    if agent_config is not None:
        reservation_id = getattr(agent_config, "channel_reservation_id", "") or ""

    logger.info(
        "VOICE_CHANNEL event=channel_reserved connection_id=%s call_sid=%s "
        "agent_id=%s token=%s... reservation_id=%s route=%s",
        connection_id,
        session.call_sid or "pending",
        getattr(session, "agent_id", None) or (agent_config.agent_id if agent_config else ""),
        _token_prefix(_session_release_token(session)),
        reservation_id or "unknown",
        session.wss_route_kind or "none",
    )


def _log_call_started(session: VoicePipelineSession, connection_id: str) -> None:
    if getattr(session, "_call_started_logged", False):
        return
    session._call_started_logged = True
    logger.info(
        "VOICE_CHANNEL event=call_started connection_id=%s call_sid=%s "
        "stream_sid=%s from=%s to=%s route=%s",
        connection_id,
        session.call_sid,
        session.stream_sid,
        session.caller_from,
        session.caller_to,
        session.wss_route_kind or "none",
    )


async def _finalize_call_session(
    session: VoicePipelineSession, connection_id: str, reason: str
) -> None:
    """
    Guaranteed end-of-call cleanup: log end and post call log.
    Idempotent — safe from stop handler, finally block, or error paths.
    """
    if getattr(session, "_call_finalized", False):
        return
    session._call_finalized = True

    logger.info(
        "VOICE_CHANNEL event=call_ended connection_id=%s call_sid=%s reason=%s",
        connection_id,
        session.call_sid or "unknown",
        reason,
    )

    try:
        _post_call_log_once(session)
    except Exception as exc:  # noqa: BLE001 - cleanup must continue
        logger.error(
            "VOICE_CHANNEL event=cleanup_failure call_sid=%s step=call_log "
            "reason=%s error=%s",
            session.call_sid or "unknown",
            reason,
            exc,
        )


async def handle_websocket(websocket):
    request_path = _extract_request_path(websocket)
    if request_path:
        from app.telephony.vobiz.ws_handler import (
            handle_vobiz_websocket,
            is_vobiz_websocket_path,
        )

        if is_vobiz_websocket_path(request_path):
            await handle_vobiz_websocket(websocket)
            return

    connection_id = f"conn_{int(datetime.now().timestamp() * 1000)}"

    if not await try_acquire_session():
        active = await active_sessions_count()
        logger.warning(
            "VOICE_CAPACITY rejected connection_id=%s active=%s max=%s",
            connection_id,
            active,
            config.MAX_CONCURRENT_SESSIONS,
        )
        try:
            await websocket.close(code=1013, reason="server at capacity")
        except Exception:
            pass
        return

    end_reason = "websocket_close"
    session: VoicePipelineSession | None = None
    try:
        active = await increment_active_sessions()
        logger.info(
            "🔗 New WebSocket connection established: %s (active=%s/%s)",
            connection_id,
            active,
            config.MAX_CONCURRENT_SESSIONS,
        )

        session = create_voice_pipeline_session(connection_id, websocket)
        session.telephony_provider = "exotel"
        logger.info(
            "🎛️ Voice pipeline for %s: %s",
            connection_id,
            getattr(session, "pipeline_name", get_pipeline_name()),
        )
        session.wss_route_kind, session.wss_route_value, request_path = _extract_wss_route(websocket)
        ws_call_id = _extract_call_id(websocket)
        session.lovable_token = session.wss_route_value if session.wss_route_kind == "token" else None
        logger.info(
            "🔎 WS request path for %s: %r (route=%s)",
            connection_id,
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
        session._channel_reserved = False
        session._channel_reserved_logged = False
        session._call_started_logged = False
        session._call_finalized = False
        active_sessions[connection_id] = session
        _assume_channel_reserved(session)
        _log_channel_reserved(session, connection_id)

        prewarm_attached = False
        if (
            getattr(session, "pipeline_name", "") == "gemini_live"
            and hasattr(session, "attach_prepared_session")
            and (ws_call_id or session.lovable_token)
        ):
            from app.voice_pipeline.gemini_live.prewarm import (
                claim_prepared_gemini_session,
            )

            prepared = await claim_prepared_gemini_session(
                call_id=ws_call_id,
                token=session.lovable_token,
            )
            if prepared is not None:
                prewarm_attached = await session.attach_prepared_session(prepared)
                if prewarm_attached:
                    _commit_agent_config_to_session(session)

        if not prewarm_attached:
            start_agent_prefetch(session, connection_id)

        async for message in websocket:
            try:
                data = json.loads(message)
            except json.JSONDecodeError as exc:
                logger.error("❌ Invalid JSON from %s: %s", connection_id, exc)
                continue

            event_type = data.get("event")

            if event_type == "connected":
                logger.info("🎉 CONNECTED - %s", connection_id)

            elif event_type == "start":
                start_data = data.get("start", data)
                session.stream_sid = start_data.get("stream_sid", data.get("stream_sid"))
                session.call_sid = start_data.get("call_sid")
                session.call_started_at = datetime.now(timezone.utc)
                session.caller_from = start_data.get("from")
                session.caller_to = start_data.get("to")
                call_records.apply_start_metadata(session, start_data)
                logger.info(
                    "🚀 START - %s | call_sid=%s stream_sid=%s from=%s to=%s provider=%s direction=%s",
                    connection_id,
                    session.call_sid,
                    session.stream_sid,
                    session.caller_from,
                    session.caller_to,
                    session.telephony_provider,
                    session.call_direction,
                )
                _log_call_started(session, connection_id)

                if (
                    not prewarm_attached
                    and getattr(session, "pipeline_name", "") == "gemini_live"
                    and hasattr(session, "attach_prepared_session")
                ):
                    from app.voice_pipeline.gemini_live.prewarm import (
                        claim_prepared_gemini_session,
                    )

                    prepared = await claim_prepared_gemini_session(
                        call_id=session.call_sid,
                        token=session.lovable_token,
                    )
                    if prepared is not None:
                        prewarm_attached = await session.attach_prepared_session(prepared)
                        if prewarm_attached:
                            _commit_agent_config_to_session(session)

                if prewarm_attached and session.agent_config:
                    agent_resolved = True
                else:
                    agent_resolved = await _resolve_agent_on_start(
                        session, connection_id, start_data
                    )
                if not agent_resolved and session.wss_route_kind not in ("number", "token"):
                    logger.error(
                        "missing voicebot token/number in wss url and no to= on start "
                        "(connection=%s, raw_path=%r)",
                        connection_id,
                        request_path,
                    )
                    await _speak_fallback_and_close(session, connection_id)
                    end_reason = "missing_route"
                    break

                if agent_resolved:
                    _commit_agent_config_to_session(session)
                    call_records.schedule_call_started(session)
                    await session.on_agent_ready()
                    await session.speak_welcome()
                else:
                    await _speak_fallback_and_close(session, connection_id)
                    end_reason = "agent_lookup_failed"
                    break

            elif event_type == "media":
                await session.add_audio_chunk(data.get("media", {}))

            elif event_type == "clear":
                logger.info("🧹 CLEAR - %s", connection_id)
                await session.handle_clear()

            elif event_type == "dtmf":
                digit = data.get("dtmf", {}).get("digit", "unknown")
                logger.info("🔢 DTMF - %s pressed %s", connection_id, digit)

            elif event_type == "mark":
                mark_name = data.get("mark", {}).get("name", "unknown")
                logger.info("📍 MARK - %s (%s)", connection_id, mark_name)

            elif event_type == "stop":
                logger.info("🛑 STOP - %s", connection_id)
                end_reason = "stop"
                await _finalize_call_session(session, connection_id, end_reason)
                await session.close()
                break

            else:
                logger.warning("❓ Unknown event '%s' from %s", event_type, connection_id)

    except websockets.exceptions.ConnectionClosed:
        end_reason = "connection_closed"
        logger.info("👋 Connection closed normally: %s", connection_id)
    except Exception as exc:
        end_reason = "exception"
        logger.error("❌ Connection error for %s: %s", connection_id, exc)
    finally:
        active_sessions.pop(connection_id, None)
        if session is not None:
            if getattr(session, "pipeline_name", "") == "gemini_live":
                from app.voice_pipeline.gemini_live.prewarm import (
                    remove_prepared_gemini_session,
                )

                await remove_prepared_gemini_session(
                    call_id=ws_call_id or session.call_sid,
                    token=session.lovable_token,
                )
            await _finalize_call_session(session, connection_id, end_reason)
            await session.close()
        remaining = await decrement_active_sessions()
        release_session_slot()
        logger.info(
            "🔚 Connection ended: %s (active=%s/%s)",
            connection_id,
            remaining,
            config.MAX_CONCURRENT_SESSIONS,
        )


from websockets.asyncio.server import ServerConnection


class _PrewarmAwareServerConnection(ServerConnection):
    """
    Allows ``POST /prewarm`` and ``POST /voicebot/prewarm`` on the main WS port
    (8002) in addition to ``WEBHOOK_PORT`` (8081), so nginx configs that proxy
    ``/`` to 8002 work without requiring an nginx reload.
    """

    _http_post_mode: bool = False
    _http_post_handled: bool = False
    _http_post_buf: bytearray | None = None

    def data_received(self, data: bytes) -> None:
        if not self._http_post_mode and self._http_post_buf is None:
            if data.startswith((b"POST /prewarm", b"POST /voicebot/prewarm")):
                self._http_post_mode = True
                self._http_post_buf = bytearray(data)
                self._try_dispatch_http_post()
                return
        if self._http_post_mode:
            if self._http_post_buf is not None:
                self._http_post_buf.extend(data)
            self._try_dispatch_http_post()
            return
        super().data_received(data)

    def _try_dispatch_http_post(self) -> None:
        if self._http_post_handled or self._http_post_buf is None:
            return
        sep = self._http_post_buf.find(b"\r\n\r\n")
        if sep == -1:
            return
        header_block = bytes(self._http_post_buf[:sep]).decode("latin-1", errors="replace")
        content_length = 0
        for line in header_block.split("\r\n")[1:]:
            if ":" in line:
                k, v = line.split(":", 1)
                if k.strip().lower() == "content-length":
                    try:
                        content_length = int(v.strip())
                    except ValueError:
                        content_length = 0
        body_start = sep + 4
        if len(self._http_post_buf) < body_start + content_length:
            return
        self._http_post_handled = True
        raw_body = bytes(self._http_post_buf[body_start : body_start + content_length])
        asyncio.create_task(self._respond_prewarm_post(raw_body))

    async def _respond_prewarm_post(self, raw_body: bytes) -> None:
        status = 400
        payload: dict = {"ok": False, "error": "invalid_json"}
        try:
            body = json.loads(raw_body.decode("utf-8"))
            if isinstance(body, dict):
                from app.voice_pipeline.gemini_live.prewarm import (
                    validate_and_schedule_gemini_prewarm,
                )

                status, payload = await validate_and_schedule_gemini_prewarm(body)
        except Exception:
            status = 400
            payload = {"ok": False, "error": "invalid_json"}

        reason = {200: "OK", 400: "Bad Request", 401: "Unauthorized", 429: "Too Many Requests"}.get(
            status, "OK"
        )
        resp_body = json.dumps(payload).encode("utf-8")
        raw_resp = (
            f"HTTP/1.1 {status} {reason}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(resp_body)}\r\n"
            "Connection: close\r\n\r\n"
        ).encode("ascii") + resp_body
        if self.transport is not None:
            self.transport.write(raw_resp)
            self.transport.close()


async def _probe_elevenlabs_api_key() -> None:
    if (
        not config.ELEVENLABS_API_KEY
        or config.VOICE_PIPELINE in ("openai_realtime", "gemini_live")
    ):
        return
    import httpx

    url = f"{config.ELEVENLABS_BASE_URL.rstrip('/')}/v1/user"
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(
                url, headers={"xi-api-key": config.ELEVENLABS_API_KEY}
            )
        if resp.status_code == 401:
            logger.error(
                "❌ ELEVENLABS_API_KEY rejected (401) — calls will not transcribe or speak. "
                "Fix the key in .env and ensure Speech-to-Text is enabled on your ElevenLabs plan."
            )
        elif resp.is_success:
            logger.info("✅ ElevenLabs API key accepted (user profile OK)")
        else:
            logger.warning(
                "ElevenLabs key probe returned HTTP %s — verify ELEVENLABS_API_KEY",
                resp.status_code,
            )
    except Exception as exc:
        logger.warning("ElevenLabs key probe skipped: %s", exc)


async def main():
    warnings = config.validate()
    for warning in warnings:
        logger.warning("⚠️  %s", warning)

    await _probe_elevenlabs_api_key()

    from app.streaming.warm_pool import warm_on_process_start

    await warm_on_process_start()

    logger.info("🚀 Starting AI Voice Agent Server...")
    logger.info("📡 Listening on %s:%s", config.HOST, config.PORT)
    if config.VOBIZ_ENABLED:
        logger.info(
            "📞 Vobiz media WebSocket path: %s (wss://%s%s)",
            config.VOBIZ_WS_PATH,
            config.PUBLIC_HOST or "<PUBLIC_HOST>",
            config.VOBIZ_WS_PATH,
        )
    pipeline = get_pipeline_name()
    logger.info("🎛️ Voice pipeline: %s (VOICE_PIPELINE=%s)", pipeline, config.VOICE_PIPELINE or "compound")

    if pipeline == "compound":
        logger.info("🧠 Brain: OpenAI model '%s'", config.OPENAI_MODEL)
        stt_label = (
            config.STREAMING_STT_MODEL
            if config.STREAMING_PIPELINE and config.STT_PROVIDER == "elevenlabs"
            else config.ELEVENLABS_STT_MODEL
        )
        logger.info(
            "🗣️  Voice: ElevenLabs voice '%s' (TTS model '%s', STT model '%s')",
            config.ELEVENLABS_VOICE_ID,
            config.ELEVENLABS_TTS_MODEL,
            stt_label,
        )
        logger.info(
            "⚙️  Capacity: max_sessions=%s stt=%s tts=%s vad_threads=%s",
            config.MAX_CONCURRENT_SESSIONS,
            config.MAX_CONCURRENT_STT,
            config.MAX_CONCURRENT_TTS,
            config.VAD_WORKER_THREADS,
        )
    elif pipeline == "gemini_live":
        logger.info(
            "🧠 Gemini Live: model '%s' voice '%s' (no ElevenLabs, no ChatGPT)",
            config.GEMINI_LIVE_MODEL,
            config.GEMINI_LIVE_VOICE,
        )
    elif pipeline == "openai_realtime":
        logger.info(
            "🧠 Realtime: model '%s' voice '%s'",
            config.OPENAI_REALTIME_MODEL,
            config.OPENAI_REALTIME_VOICE,
        )
        logger.info(
            "⚙️  Capacity: max_sessions=%s",
            config.MAX_CONCURRENT_SESSIONS,
        )

    server = await websockets.serve(
        handle_websocket,
        config.HOST,
        config.PORT,
        ping_interval=30,
        ping_timeout=10,
        process_request=_process_request,
        create_connection=_PrewarmAwareServerConnection,
    )

    webhook_runner = await webhook_server.start()

    logger.info("✅ AI Voice Agent Server running at ws://%s:%s", config.HOST, config.PORT)

    try:
        await server.wait_closed()
    finally:
        await webhook_server.stop(webhook_runner)
        if get_pipeline_name() == "compound":
            await elevenlabs_service.close()
        await lovable_client.close()
        await openai_brain.close()
        shutdown_runtime()


if __name__ == "__main__":
    try:
        try:
            import uvloop

            uvloop.install()
        except ImportError:
            pass
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 AI Voice Agent Server shutting down...")
