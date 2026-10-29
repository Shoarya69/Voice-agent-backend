"""Vobiz HTTP webhooks (answer XML, hangup relay) — optional if Moontech URLs point here."""

from __future__ import annotations

import asyncio
import logging
from urllib.parse import quote
from xml.sax.saxutils import escape

from aiohttp import web

from app import call_records, config, lovable_client
from app.voice_pipeline.factory import get_pipeline_name

logger = logging.getLogger(__name__)


def stream_xml(wss_url: str, *, status_callback_url: str | None = None) -> str:
    status_attr = ""
    if status_callback_url:
        status_attr = (
            f' statusCallbackUrl="{escape(status_callback_url)}"'
            ' statusCallbackMethod="POST"'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n<Response>\n'
        f'    <Stream bidirectional="true" keepCallAlive="true" '
        f'contentType="audio/x-mulaw;rate=8000"{status_attr}>'
        f"{escape(wss_url)}</Stream>\n</Response>"
    )


def _public_wss_base() -> str:
    host = config.PUBLIC_HOST.strip()
    if not host:
        raise ValueError("PUBLIC_HOST is not configured")
    return f"wss://{host}"


def _wss_url_for_request(
    *,
    token: str | None,
    number: str | None,
    call_id: str | None = None,
) -> str:
    base = _public_wss_base()
    path = config.VOBIZ_WS_PATH.rstrip("/") or "/ws"
    if token:
        url = f"{base}{path}?token={quote(token, safe='')}"
        if call_id:
            url += f"&call_id={quote(call_id, safe='')}"
        return url
    if number:
        digits = "".join(ch for ch in number if ch.isdigit())
        return f"{base}{path}/number/{digits}"
    return base + path


async def _schedule_gemini_prewarm_from_answer(body: dict) -> None:
    try:
        from app.voice_pipeline.gemini_live.prewarm import (
            validate_and_schedule_gemini_prewarm,
        )

        status, payload = await validate_and_schedule_gemini_prewarm(body)
        logger.info(
            "PREWARM event=answer_scheduled call_id=%s http_status=%s ok=%s",
            body.get("call_id"),
            status,
            payload.get("ok"),
        )
    except Exception as exc:
        logger.warning(
            "PREWARM event=answer_schedule_failed call_id=%s error=%s",
            body.get("call_id"),
            type(exc).__name__,
        )


async def handle_answer(request: web.Request) -> web.Response:
    """Return Vobiz Stream XML pointing at this server's /ws endpoint."""
    data = dict(request.query)
    if request.method == "POST":
        try:
            form = await request.post()
            data.update({k: str(v) for k, v in form.items()})
        except Exception:
            pass

    token = (data.get("token") or "").strip()
    call_id = (
        data.get("call_id")
        or data.get("callId")
        or data.get("CallUUID")
        or data.get("callUUID")
        or ""
    ).strip()
    to_number = (data.get("To") or data.get("to") or data.get("number") or "").strip()
    logger.info(
        "Vobiz answer webhook CallUUID=%s To=%s token=%s",
        data.get("CallUUID") or data.get("callUUID"),
        to_number[-4:] if to_number else "",
        (token[:6] + "...") if token else "",
    )

    if config.VOBIZ_PREWARM_ON_ANSWER:
        if token:
            asyncio.create_task(
                lovable_client.prewarm_agent_bundle(
                    token=token,
                    source="vobiz_answer",
                )
            )
        elif to_number:
            asyncio.create_task(
                lovable_client.prewarm_agent_bundle(
                    number=to_number,
                    source="vobiz_answer",
                )
            )
        if (
            get_pipeline_name() == "gemini_live"
            and token
            and call_id
        ):
            prewarm_body = {
                "voicebot_token": token,
                "token": token,
                "call_id": call_id,
                "direction": "outbound",
            }
            asyncio.create_task(_schedule_gemini_prewarm_from_answer(prewarm_body))

    try:
        if token:
            wss = _wss_url_for_request(token=token, number=None, call_id=call_id or None)
        else:
            wss = _wss_url_for_request(token=None, number=to_number)
        status_url = None
        if config.PUBLIC_HOST:
            status_url = f"https://{config.PUBLIC_HOST}/vobiz/stream-status"
        body = stream_xml(wss, status_callback_url=status_url)
    except ValueError as exc:
        logger.error("Vobiz answer misconfigured: %s", exc)
        return web.Response(status=500, text="PUBLIC_HOST not configured")

    logger.info(
        "Vobiz answer XML ready CallUUID=%s wss_target=%s to=...%s",
        data.get("CallUUID") or data.get("callUUID"),
        wss,
        to_number[-4:] if to_number else "",
    )
    return web.Response(text=body, content_type="application/xml")


async def handle_hangup(request: web.Request) -> web.Response:
    form = dict(request.query)
    if request.method == "POST":
        try:
            post = await request.post()
            form.update({k: str(v) for k, v in post.items()})
        except Exception:
            pass

    number = form.get("From") or form.get("from") or form.get("To") or form.get("to") or ""
    logger.info(
        "Vobiz hangup CallUUID=%s status=%s",
        form.get("CallUUID") or form.get("callUUID"),
        form.get("CallStatus") or form.get("Status"),
    )
    relay_fields = dict(form)
    relay_fields.setdefault("provider", "vobiz")
    await call_records.merge_provider_hangup("vobiz", relay_fields)
    await lovable_client.post_exotel_status(number, relay_fields)
    return web.Response(text="OK")


async def handle_stream_status(request: web.Request) -> web.Response:
    form = dict(request.query)
    if request.method == "POST":
        try:
            post = await request.post()
            form.update({k: str(v) for k, v in post.items()})
        except Exception:
            pass
    logger.info(
        "Vobiz stream-status Event=%s StreamID=%s Error=%s",
        form.get("Event"),
        form.get("StreamID"),
        form.get("Error"),
    )
    return web.Response(text="OK")


def register_routes(app: web.Application) -> None:
    app.router.add_route("*", "/vobiz/answer", handle_answer)
    app.router.add_route("*", "/vobiz/hangup", handle_hangup)
    app.router.add_route("*", "/vobiz/stream-status", handle_stream_status)
