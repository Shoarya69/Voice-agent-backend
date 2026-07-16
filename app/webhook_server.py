"""
Plain-HTTP webhook relay for Exotel's telephony callbacks.

This is intentionally separate from the WebSocket Voicebot Applet server in
ai_server.py: Exotel calls these two routes directly (configured as the
Status Callback / conversation applet URL for a phone number in the Exotel
Flow), and each one is relayed as a form-encoded POST to the matching
Lovable hook, identified only by the caller's phone number (never an
agent_id - Lovable resolves the agent itself).

Also exposes POST /voicebot/prewarm so MoontechPro (or Exotel status
callbacks) can warm agent caches during the ringing phase, before the WSS
media stream connects.

Runs on its own port (config.WEBHOOK_PORT) so it can be exposed/proxied
independently from the WSS media-streaming port.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging

from aiohttp import web

from app import config, lovable_client

logger = logging.getLogger(__name__)

# Exotel statuses where the callee is still ringing / not yet on media WSS.
_PREWARM_EXOTEL_STATUSES = frozenset({"ringing", "in-progress", "queued"})


def _caller_number(form: dict) -> str:
    # "number" is documented as the caller's phone (any format - Lovable
    # matches on the last 10 digits), i.e. Exotel's "From" field.
    return form.get("From") or form.get("from") or ""


def _exophone_number(form: dict) -> str:
    """Dialed Exotel number — matches the WSS /number/{exophone} route."""
    return form.get("To") or form.get("to") or ""


def _authorize_prewarm(request: web.Request) -> bool:
    secret = config.LOVABLE_API_SECRET
    if not secret:
        return False
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return False
    token = auth[7:].strip()
    return hmac.compare_digest(token, secret)


def _schedule_prewarm(
    *,
    token: str | None = None,
    number: str | None = None,
    source: str,
) -> None:
    asyncio.create_task(
        lovable_client.prewarm_agent_bundle(
            token=token,
            number=number,
            source=source,
        )
    )


async def _maybe_prewarm_from_exotel_status(form: dict) -> None:
    if not config.PREWARM_ON_EXOTEL_STATUS:
        return
    status = (form.get("CallStatus") or form.get("Status") or "").strip().lower()
    if status not in _PREWARM_EXOTEL_STATUSES:
        return
    exophone = _exophone_number(form)
    if not exophone:
        return
    logger.info(
        "🔥 Scheduling prewarm from Exotel status=%s exophone=...%s call_sid=%s",
        status,
        exophone[-4:] if exophone else "",
        form.get("CallSid"),
    )
    _schedule_prewarm(number=exophone, source=f"exotel_status:{status}")


async def _handle_prewarm(request: web.Request) -> web.Response:
    if not _authorize_prewarm(request):
        return web.Response(status=401, text="Unauthorized")
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return web.Response(status=400, text="Invalid JSON")

    token = (body.get("token") or "").strip()
    number = (body.get("number") or "").strip()
    if bool(token) == bool(number):
        return web.Response(status=400, text="Provide exactly one of token or number")

    logger.info(
        "🔥 Prewarm request accepted source=http key=%s",
        token[:6] + "..." if token else f"...{number[-4:]}",
    )
    _schedule_prewarm(
        token=token or None,
        number=number or None,
        source="http_prewarm",
    )
    return web.json_response({"status": "accepted"})


async def _handle_status(request: web.Request) -> web.Response:
    form = dict(await request.post())
    number = _caller_number(form)
    logger.info(
        "📞 Exotel status callback: call_sid=%s status=%s from=%s to=%s",
        form.get("CallSid"),
        form.get("CallStatus"),
        form.get("From"),
        form.get("To"),
    )
    await _maybe_prewarm_from_exotel_status(form)
    await lovable_client.post_exotel_status(number, form)
    return web.Response(text="OK")


async def _handle_conversation(request: web.Request) -> web.Response:
    form = dict(await request.post())
    number = _caller_number(form)
    logger.info(
        "💬 Exotel conversation callback: call_sid=%s from=%s to=%s",
        form.get("CallSid"),
        form.get("From"),
        form.get("To"),
    )
    await lovable_client.post_exotel_conversation(number, form)
    return web.Response(text="OK")


async def _handle_health(_request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


def _build_app() -> web.Application:
    app = web.Application()
    app.router.add_post("/exotel/status", _handle_status)
    app.router.add_post("/exotel/conversation", _handle_conversation)
    app.router.add_post("/voicebot/prewarm", _handle_prewarm)
    app.router.add_get("/health", _handle_health)
    return app


async def start() -> web.AppRunner:
    """Start the webhook relay HTTP server. Returns the runner so callers can stop() it later."""
    app = _build_app()
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, config.HOST, config.WEBHOOK_PORT)
    await site.start()
    logger.info(
        "🌐 Exotel webhook relay listening on http://%s:%s (/exotel/status, /exotel/conversation, /voicebot/prewarm)",
        config.HOST,
        config.WEBHOOK_PORT,
    )
    return runner


async def stop(runner: "web.AppRunner | None") -> None:
    if runner is not None:
        await runner.cleanup()
