"""
Call CRM persistence: Supabase ``ai_calls`` + Lovable call-log API.

Inbound Vobiz/Exotel calls that hit this voice server directly get:
  - ``in_progress`` row when the agent is resolved on ``start``
  - final transcript + duration on WebSocket end
  - recording/outcome merged from provider hangup webhooks (may arrive before/after WSS close)
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any
from weakref import WeakValueDictionary

from app import config, lovable_client
from app import supabase_agent_provider
from app.voice_pipeline.base import VoicePipelineSession

logger = logging.getLogger(__name__)

# call_sid -> live session (weak — does not keep calls alive)
_sessions_by_call_sid: WeakValueDictionary[str, VoicePipelineSession] = (
    WeakValueDictionary()
)
# call_sid -> hangup fields received before/after media ends
_hangup_enrichment: dict[str, dict[str, Any]] = {}
_enrichment_lock = asyncio.Lock()

AI_CALLS_TABLE = "ai_calls"
_supabase_call_writes_blocked = False


def _log_phone(value: str | None) -> str:
    """Lovable call-log API requires strings, not JSON null."""
    return str(value).strip() if value else ""


def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def transcript_role(role: str) -> str:
    if role == "assistant":
        return "agent"
    return role if role in ("user", "agent") else "user"


def infer_direction(
    session: VoicePipelineSession,
    start_data: dict[str, Any] | None = None,
) -> str:
    if start_data:
        raw = (start_data.get("Direction") or start_data.get("direction") or "").strip().lower()
        if raw in ("inbound", "outbound"):
            return raw
    if getattr(session, "call_direction", None) in ("inbound", "outbound"):
        return session.call_direction  # type: ignore[return-value]
    if session.wss_route_kind == "number":
        return "inbound"
    if session.wss_route_kind == "token":
        return "outbound"
    return "inbound"


def apply_start_metadata(
    session: VoicePipelineSession,
    start_data: dict[str, Any] | None = None,
) -> None:
    """Set provider/direction defaults and register for hangup correlation."""
    if not getattr(session, "telephony_provider", None):
        session.telephony_provider = "exotel"
    session.call_direction = infer_direction(session, start_data)
    if session.call_sid:
        _sessions_by_call_sid[str(session.call_sid)] = session


def unregister_session(session: VoicePipelineSession) -> None:
    call_sid = session.call_sid
    if call_sid and str(call_sid) in _sessions_by_call_sid:
        try:
            del _sessions_by_call_sid[str(call_sid)]
        except KeyError:
            pass


def build_call_log_payload(
    session: VoicePipelineSession,
    ended_at: datetime,
) -> dict[str, Any]:
    started_at = getattr(session, "call_started_at", None) or ended_at
    duration_seconds = max(0, int((ended_at - started_at).total_seconds()))

    history = getattr(session, "history", []) or []
    total = len(history)
    transcript: list[dict[str, Any]] = []
    for index, entry in enumerate(history):
        fraction = index / (total - 1) if total > 1 else 0.0
        entry_ts = started_at + (ended_at - started_at) * fraction
        transcript.append(
            {
                "role": transcript_role(entry.get("role", "assistant")),
                "text": entry.get("content", ""),
                "ts": int(entry_ts.timestamp()),
            }
        )

    agent_config = getattr(session, "agent_config", None)
    call_sid = session.call_sid or ""
    enrichment = _hangup_enrichment.get(str(call_sid), {})

    recording_url = getattr(session, "recording_url", None) or enrichment.get(
        "recording_url"
    )
    outcome = enrichment.get("outcome")
    if enrichment.get("duration_seconds") is not None:
        duration_seconds = int(enrichment["duration_seconds"])

    payload: dict[str, Any] = {
        "call_sid": call_sid,
        "provider_call_sid": call_sid,
        "user_id": agent_config.user_id if agent_config else "",
        "agent_id": getattr(session, "agent_id", None)
        or (agent_config.agent_id if agent_config else ""),
        "phone_number_id": getattr(agent_config, "phone_number_id", "") if agent_config else "",
        "provider": getattr(session, "telephony_provider", None) or "exotel",
        "from_number": _log_phone(getattr(session, "caller_from", None)),
        "to_number": _log_phone(
            getattr(session, "caller_to", None)
            or (
                session.wss_route_value
                if getattr(session, "wss_route_kind", None) == "number"
                else None
            )
        ),
        "direction": getattr(session, "call_direction", None) or "inbound",
        "duration_seconds": duration_seconds,
        "started_at": iso_z(started_at),
        "ended_at": iso_z(ended_at),
        "transcript": transcript,
        "status": "failed" if getattr(session, "_call_error_stage", "") else "completed",
    }
    error_stage = getattr(session, "_call_error_stage", "") or ""
    if error_stage:
        payload["error_stage"] = error_stage
    if recording_url:
        payload["recording_url"] = recording_url
    if outcome:
        payload["outcome"] = outcome
    reservation = ""
    if agent_config:
        reservation = getattr(agent_config, "channel_reservation_id", "") or ""
    if reservation:
        payload["channel_reservation_id"] = reservation
    return payload


def _uuid_or_none(value: str | None) -> str | None:
    if not value:
        return None
    value = str(value).strip()
    if len(value) == 36 and value.count("-") == 4:
        return value
    return None


def _row_from_session_started(session: VoicePipelineSession) -> dict[str, Any]:
    agent_config = getattr(session, "agent_config", None)
    started = getattr(session, "call_started_at", None) or datetime.now(timezone.utc)
    row: dict[str, Any] = {
        "call_sid": session.call_sid,
        "provider": getattr(session, "telephony_provider", None) or "exotel",
        "direction": getattr(session, "call_direction", None) or "inbound",
        "from_number": _log_phone(getattr(session, "caller_from", None)),
        "to_number": _log_phone(getattr(session, "caller_to", None)),
        "status": "in_progress",
        "started_at": iso_z(started),
    }
    uid = _uuid_or_none(agent_config.user_id if agent_config else "")
    aid = _uuid_or_none(
        getattr(session, "agent_id", None) or (agent_config.agent_id if agent_config else "")
    )
    pnid = _uuid_or_none(getattr(agent_config, "phone_number_id", "") if agent_config else "")
    if uid:
        row["user_id"] = uid
    if aid:
        row["agent_id"] = aid
    if pnid:
        row["phone_number_id"] = pnid
    reservation = _uuid_or_none(
        getattr(agent_config, "channel_reservation_id", "") if agent_config else ""
    )
    if reservation:
        row["channel_reservation_id"] = reservation
    return row


def _row_from_completed_payload(payload: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "call_sid": payload["call_sid"],
        "provider": payload.get("provider") or "exotel",
        "direction": payload.get("direction") or "inbound",
        "from_number": payload.get("from_number"),
        "to_number": payload.get("to_number"),
        "status": payload.get("status") or "completed",
        "started_at": payload.get("started_at"),
        "ended_at": payload.get("ended_at"),
        "duration_seconds": payload.get("duration_seconds"),
        "transcript": payload.get("transcript") or [],
    }
    for key in ("user_id", "agent_id", "phone_number_id", "channel_reservation_id"):
        uuid_val = _uuid_or_none(payload.get(key))
        if uuid_val:
            row[key] = uuid_val
    if payload.get("recording_url"):
        row["recording_url"] = payload["recording_url"]
    if payload.get("outcome"):
        row["outcome"] = payload["outcome"]
    return row


def _is_upsert_conflict_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "42p10" in msg or "on conflict" in msg or "on_conflict" in msg


async def _supabase_upsert(row: dict[str, Any]) -> None:
    if not config.ENABLE_SUPABASE_CALL_RECORDS:
        return
    if not supabase_agent_provider.is_configured():
        return
    call_sid = row.get("call_sid")
    if not call_sid:
        return

    client = await supabase_agent_provider._get_supabase()  # noqa: SLF001
    run_sync = supabase_agent_provider._run_sync  # noqa: SLF001

    async def _try_upsert() -> None:
        await run_sync(
            lambda: client.table(AI_CALLS_TABLE)
            .upsert(row, on_conflict="call_sid")
            .execute()
        )

    async def _insert_or_update() -> None:
        """Fallback when DB has no UNIQUE(call_sid) — common on older Moontech schemas."""

        def _select():
            return (
                client.table(AI_CALLS_TABLE)
                .select("call_sid")
                .eq("call_sid", call_sid)
                .limit(1)
                .execute()
            )

        existing = await run_sync(_select)
        data = getattr(existing, "data", None) or []
        if data:
            patch = {k: v for k, v in row.items() if k != "call_sid" and v is not None}

            def _update():
                return (
                    client.table(AI_CALLS_TABLE)
                    .update(patch)
                    .eq("call_sid", call_sid)
                    .execute()
                )

            await run_sync(_update)
        else:

            def _insert():
                return client.table(AI_CALLS_TABLE).insert(row).execute()

            await run_sync(_insert)

    global _supabase_call_writes_blocked
    if _supabase_call_writes_blocked:
        return

    try:
        try:
            await _try_upsert()
        except Exception as exc:
            if _is_upsert_conflict_error(exc):
                logger.warning(
                    "ai_calls upsert on_conflict=call_sid unsupported — using select/update fallback"
                )
                await _insert_or_update()
            else:
                raise
        logger.info(
            "Supabase ai_calls upsert ok call_sid=%s provider=%s status=%s",
            call_sid,
            row.get("provider"),
            row.get("status"),
        )
    except supabase_agent_provider.SupabaseAgentError as exc:
        detail = str(exc).lower()
        if exc.status_code in (401, 403) or "permission denied" in detail:
            if not _supabase_call_writes_blocked:
                _supabase_call_writes_blocked = True
                logger.warning(
                    "Supabase ai_calls writes disabled for this process (anon key / RLS). "
                    "Set SUPABASE_SERVICE_ROLE_KEY on the voice server or "
                    "ENABLE_SUPABASE_CALL_RECORDS=false. First error: %s",
                    exc,
                )
            return
        logger.error(
            "Supabase ai_calls upsert failed call_sid=%s error=%s",
            call_sid,
            exc,
        )
    except Exception as exc:
        logger.error(
            "Supabase ai_calls upsert failed call_sid=%s error=%s",
            call_sid,
            exc,
        )


async def persist_call_started(session: VoicePipelineSession) -> None:
    if not session.call_sid:
        return
    if not (getattr(session, "agent_config", None) or getattr(session, "agent_id", None)):
        return
    row = _row_from_session_started(session)
    await _supabase_upsert(row)


async def persist_call_completed(payload: dict[str, Any]) -> None:
    row = _row_from_completed_payload(payload)
    await _supabase_upsert(row)


def _normalize_outcome(status: str | None) -> str | None:
    if not status:
        return None
    s = status.strip().lower()
    mapping = {
        "completed": "completed",
        "in-progress": "in_progress",
        "busy": "busy",
        "no-answer": "no_answer",
        "failed": "failed",
        "canceled": "canceled",
        "cancelled": "canceled",
    }
    return mapping.get(s, s)


async def merge_provider_hangup(
    provider: str,
    form: dict[str, Any],
) -> None:
    """
    Apply Vobiz/Exotel hangup/status fields to the live session (if any) and DB.
    """
    call_sid = (
        form.get("CallUUID")
        or form.get("CallSid")
        or form.get("callUUID")
        or form.get("call_sid")
        or ""
    )
    if not call_sid:
        return

    recording_url = (
        form.get("RecordingUrl")
        or form.get("recording_url")
        or form.get("RecordingURL")
    )
    outcome = _normalize_outcome(
        form.get("CallStatus") or form.get("Status") or form.get("status")
    )
    duration_raw = (
        form.get("CallDuration")
        or form.get("Duration")
        or form.get("duration")
    )
    duration_seconds: int | None = None
    if duration_raw is not None and str(duration_raw).strip().isdigit():
        duration_seconds = int(str(duration_raw).strip())

    patch: dict[str, Any] = {"provider": provider}
    if recording_url:
        patch["recording_url"] = str(recording_url)
    if outcome:
        patch["outcome"] = outcome
        patch["status"] = "completed" if outcome == "completed" else outcome
    if duration_seconds is not None:
        patch["duration_seconds"] = duration_seconds

    async with _enrichment_lock:
        existing = _hangup_enrichment.get(str(call_sid), {})
        existing.update({k: v for k, v in patch.items() if v is not None})
        _hangup_enrichment[str(call_sid)] = existing

    session = _sessions_by_call_sid.get(str(call_sid))
    if session is not None:
        if recording_url:
            session.recording_url = str(recording_url)

    row = {"call_sid": str(call_sid), **patch}
    await _supabase_upsert(row)


async def finalize_call(session: VoicePipelineSession) -> None:
    """Dual-write: Supabase ai_calls + Lovable call-log (never raises)."""
    if getattr(session, "_call_log_posted", False):
        return
    has_agent = bool(
        getattr(session, "agent_config", None) or getattr(session, "agent_id", None)
    )
    if not has_agent and not getattr(session, "call_sid", None):
        return

    session._call_log_posted = True
    ended_at = datetime.now(timezone.utc)
    payload = build_call_log_payload(session, ended_at)

    call_sid = str(session.call_sid or "")
    async with _enrichment_lock:
        _hangup_enrichment.pop(call_sid, None)

    if has_agent:
        await persist_call_completed(payload)
    await lovable_client.post_call_log(payload)
    unregister_session(session)


def schedule_call_started(session: VoicePipelineSession) -> None:
    asyncio.create_task(persist_call_started(session))


def schedule_finalize_call(session: VoicePipelineSession) -> None:
    asyncio.create_task(finalize_call(session))
