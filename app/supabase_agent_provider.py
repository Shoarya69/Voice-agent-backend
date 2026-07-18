"""
Direct Supabase reads for agent config (replaces Moontech HTTP GET endpoints).

Inbound (by number): single RPC ``get_inbound_agent_bundle``.
Outbound (by token): Supabase tables + Moontech greeting cold-cache when needed.
Greeting enrichment: Moontech HTTP only when PCM not cached (write-side synth).

Write paths (call-log, webhooks) stay on Moontech via lovable_client.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

import httpx

from app import config

logger = logging.getLogger(__name__)

AGENT_COLS = (
    "id, user_id, name, business_name, default_language, voice, system_prompt, "
    "greeting_message, closing_message, script, status, speed, tone, voicebot_token, "
    "greeting_audio_url, greeting_audio_base64, greeting_audio_mime, greeting_audio_bytes, "
    "greeting_audio_hash, greeting_audio_generated_at"
)

_INBOUND_AGENT_RPC = "get_inbound_agent_bundle"

_DEFAULT_VOICE_ID = "cgSgspJ2msm6clMCkdW9"
_DEFAULT_SYSTEM_PROMPT = "You are a helpful voice assistant."

_supabase_client = None
_supabase_key_mode: str | None = None
_supabase_lock = asyncio.Lock()


class SupabaseAgentError(Exception):
    """Raised when a Supabase agent lookup fails."""

    def __init__(self, message: str, *, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code


def _resolve_api_key() -> tuple[str, str]:
    """
    Return (api_key, mode). Prefer service_role; fall back to anon/publishable.
    """
    if config.SUPABASE_SERVICE_ROLE_KEY:
        return config.SUPABASE_SERVICE_ROLE_KEY, "service_role"
    if config.SUPABASE_ANON_KEY:
        return config.SUPABASE_ANON_KEY, "anon"
    return "", ""


def is_configured() -> bool:
    key, _ = _resolve_api_key()
    return bool(config.SUPABASE_URL and key)


def using_anon_key() -> bool:
    _, mode = _resolve_api_key()
    return mode == "anon"


async def _get_supabase():
    global _supabase_client, _supabase_key_mode
    api_key, mode = _resolve_api_key()
    if not api_key:
        raise SupabaseAgentError("Supabase API key not configured")

    if _supabase_client is not None and _supabase_key_mode == mode:
        return _supabase_client

    async with _supabase_lock:
        if _supabase_client is not None and _supabase_key_mode == mode:
            return _supabase_client
        from supabase import create_client

        _supabase_client = create_client(config.SUPABASE_URL, api_key)
        _supabase_key_mode = mode
        logger.info(
            "Supabase client ready mode=%s url=%s",
            mode,
            config.SUPABASE_URL.rsplit("/", 1)[-1][:20],
        )
    return _supabase_client


def normalize_phone(raw: str) -> str:
    cleaned = re.sub(r"[^\d+]", "", str(raw or ""))
    if cleaned.startswith("+"):
        return cleaned
    digits = re.sub(r"\D", "", cleaned)
    if len(digits) == 10:
        return f"+91{digits}"
    if len(digits) == 11 and digits.startswith("0"):
        return f"+91{digits[1:]}"
    if len(digits) == 12 and digits.startswith("91"):
        return f"+{digits}"
    return f"+{digits}" if digits else ""


def normalize_language(value: str | None) -> str:
    s = str(value or "").strip().lower()
    if s in ("hin", "hi", "hi-in", "hindi", "bilingual"):
        return "hin"
    if s in ("english", "en", "en-us", "en-in"):
        return "eng"
    return s or "hin"


def _base_first_message(agent: dict[str, Any]) -> str:
    if agent.get("greeting_message"):
        return str(agent["greeting_message"])
    name = agent.get("name") or "AI Voice Assistant"
    business = agent.get("business_name") or "our team"
    return f"Hello! This is {name} from {business}. How can I help you today?"


def build_campaign_prompt_suffix(
    script: dict[str, Any] | None, customer_name: str | None
) -> str:
    if not script:
        return ""
    parts: list[str] = []
    if customer_name:
        parts.append(f"The person you are calling is named {customer_name}.")
    for key, heading in (
        ("opening", "## Opening / Greeting"),
        ("pitch", "## Main pitch / value proposition"),
        ("objection", "## Objection handling"),
        ("closing", "## Closing / call to action"),
    ):
        value = (script.get(key) or "").strip()
        if value:
            parts.append(f"{heading}\n{value}")
    if not parts:
        return ""
    return (
        "\n\n---\n## CAMPAIGN SCRIPT (follow this for the current call)\n"
        + "\n\n".join(parts)
    )


def build_campaign_first_message(
    script: dict[str, Any] | None,
    fallback: str,
    customer_name: str | None,
) -> str:
    opening = (script or {}).get("opening", "") or ""
    opening = str(opening).strip()
    if not opening:
        return fallback
    if customer_name:
        opening = re.sub(r"\[lead name\]", customer_name, opening, flags=re.IGNORECASE)
        opening = re.sub(r"\{\{\s*name\s*\}\}", customer_name, opening, flags=re.IGNORECASE)
        opening = re.sub(r"\{name\}", customer_name, opening, flags=re.IGNORECASE)
    return opening


def _agent_bundle_fields(agent: dict[str, Any]) -> dict[str, Any]:
    return {
        "system_prompt": agent.get("system_prompt") or agent.get("script") or _DEFAULT_SYSTEM_PROMPT,
        "voice_id": agent.get("voice") or _DEFAULT_VOICE_ID,
        "first_message": _base_first_message(agent),
        "greeting_audio_url": agent.get("greeting_audio_url"),
        "closing_message": agent.get("closing_message"),
        "language": normalize_language(agent.get("default_language")),
        "temperature": 0.7,
        "max_tokens": 512,
        "speed": agent.get("speed") if isinstance(agent.get("speed"), (int, float)) else 1.0,
        "tone": agent.get("tone") or "professional",
    }


def _greeting_object(agent: dict[str, Any], first_message: str) -> dict[str, Any]:
    audio_b64 = agent.get("greeting_audio_base64")
    mime = agent.get("greeting_audio_mime") or (
        "audio/L16;rate=8000" if audio_b64 else None
    )
    return {
        "ready": bool(audio_b64),
        "text": first_message,
        "audio_base64": audio_b64,
        "mime": mime,
        "bytes": agent.get("greeting_audio_bytes"),
        "url": agent.get("greeting_audio_url"),
        "hash": agent.get("greeting_audio_hash"),
        "generated_at": agent.get("greeting_audio_generated_at"),
    }


async def _run_sync(fn, *args, **kwargs):
    try:
        return await asyncio.to_thread(fn, *args, **kwargs)
    except Exception as exc:
        detail = str(exc).lower()
        if any(
            token in detail
            for token in (
                "permission denied",
                "row-level security",
                "rls",
                "42501",
                "401",
                "403",
                "jwt",
                "not authorized",
            )
        ):
            raise SupabaseAgentError(
                f"Supabase permission denied (anon key + RLS?): {exc}",
                status_code=403,
            ) from exc
        raise SupabaseAgentError(f"Supabase query failed: {exc}") from exc


async def _fetch_greeting_from_moontech(agent_id: str) -> dict[str, Any] | None:
    """Cold-cache fallback — Moontech synthesises and persists greeting audio."""
    base_url = (config.MOONTECH_BASE_URL or config.LOVABLE_APP_URL or "").rstrip("/")
    if not base_url:
        logger.warning("Moontech greeting fallback skipped: MOONTECH_BASE_URL not set")
        return None

    url = f"{base_url}/api/public/voicebot/agent-with-greeting/{quote(agent_id, safe='')}"
    headers: dict[str, str] = {}
    if config.LOVABLE_API_SECRET:
        headers["Authorization"] = f"Bearer {config.LOVABLE_API_SECRET}"

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(6.0, connect=4.0)) as client:
            response = await client.get(url, headers=headers)
            if response.status_code >= 400:
                logger.warning(
                    "Moontech greeting fallback failed agent_id=%s status=%s",
                    agent_id[:8],
                    response.status_code,
                )
                return None
            payload = response.json()
            return (payload or {}).get("greeting")
    except Exception as exc:
        logger.warning(
            "Moontech greeting fallback error agent_id=%s: %s",
            agent_id[:8],
            exc,
        )
        return None


def _merge_greeting(agent: dict[str, Any], greeting: dict[str, Any]) -> dict[str, Any]:
    merged = dict(agent)
    merged.update(
        {
            "greeting_audio_base64": greeting.get("audio_base64"),
            "greeting_audio_mime": greeting.get("mime"),
            "greeting_audio_bytes": greeting.get("bytes"),
            "greeting_audio_url": greeting.get("url"),
            "greeting_audio_hash": greeting.get("hash"),
            "greeting_audio_generated_at": greeting.get("generated_at"),
        }
    )
    return merged


def _payload_from_inbound_rpc_row(row: dict[str, Any]) -> dict[str, Any]:
    """Map ``get_inbound_agent_bundle`` RPC row to lovable_client payload shape."""
    first_message = (row.get("first_message") or "").strip()
    if not first_message:
        name = row.get("assistant_name") or "AI Voice Assistant"
        business = row.get("business_name") or "our team"
        first_message = (
            f"Hello! This is {name} from {business}. How can I help you today?"
        )

    greeting_source = {
        "greeting_audio_base64": row.get("greeting_audio_base64"),
        "greeting_audio_mime": row.get("greeting_audio_mime"),
        "greeting_audio_bytes": row.get("greeting_audio_bytes"),
        "greeting_audio_url": row.get("greeting_audio_url"),
        "greeting_audio_hash": row.get("greeting_audio_hash"),
        "greeting_audio_generated_at": row.get("greeting_audio_generated_at"),
    }

    return {
        "phone_number_id": row.get("phone_number_id"),
        "number": row.get("number"),
        "voicebot_token": row.get("voicebot_token"),
        "token": row.get("voicebot_token") or "",
        "agent_id": row["agent_id"],
        "user_id": row.get("user_id"),
        "system_prompt": row.get("system_prompt") or _DEFAULT_SYSTEM_PROMPT,
        "voice_id": row.get("voice_id") or _DEFAULT_VOICE_ID,
        "first_message": first_message,
        "closing_message": row.get("closing_message"),
        "language": normalize_language(row.get("language")),
        "temperature": 0.7,
        "max_tokens": 512,
        "speed": row.get("speed") if isinstance(row.get("speed"), (int, float)) else 1.0,
        "tone": row.get("tone") or "professional",
        "greeting": _greeting_object(greeting_source, first_message),
    }


async def fetch_agent_payload_by_number(number: str) -> dict[str, Any]:
    """
    Resolve inbound agent bundle via Supabase RPC (single source of truth).

    Replaces legacy REST lookups to phone_numbers / ai_agents and Moontech HTTP
    ``/agent-by-number``.
    """
    raw = (number or "").strip()
    if not raw:
        raise SupabaseAgentError("invalid number", status_code=400)

    client = await _get_supabase()

    def _rpc_call():
        return client.rpc(
            _INBOUND_AGENT_RPC,
            {"phone_number": raw},
        ).execute()

    try:
        result = await _run_sync(_rpc_call)
    except SupabaseAgentError:
        raise
    except Exception as exc:
        logger.exception(
            "get_inbound_agent_bundle RPC failed number=...%s",
            raw[-4:] if len(raw) >= 4 else raw,
        )
        raise SupabaseAgentError(f"inbound agent lookup failed: {exc}") from exc

    data = result.data
    row: dict[str, Any] | None = None
    if isinstance(data, list):
        row = data[0] if data else None
    elif isinstance(data, dict):
        row = data

    if not row:
        logger.info(
            "number_not_found number=...%s raw=%r",
            raw[-4:] if len(raw) >= 4 else raw,
            raw,
        )
        raise SupabaseAgentError("number_not_found", status_code=404)

    logger.info(
        "get_inbound_agent_bundle ok number=...%s agent_id=%s",
        raw[-4:] if len(raw) >= 4 else raw,
        row.get("agent_id"),
    )
    return _payload_from_inbound_rpc_row(row)


async def fetch_agent_payload_by_token(token: str) -> dict[str, Any]:
    if not token or len(token) < 16:
        raise SupabaseAgentError("missing or invalid token", status_code=400)

    client = await _get_supabase()
    override_script = None
    override_customer_name = None
    agent_id_from_call_token = None

    def _fetch_call_token():
        return (
            client.table("voicebot_call_tokens")
            .select("agent_id, overrides, expires_at")
            .eq("token", token)
            .maybe_single()
            .execute()
        )

    call_tok_result = await _run_sync(_fetch_call_token)
    call_tok = _extract_supabase_data(call_tok_result)
    if call_tok:
        expires_at = call_tok.get("expires_at")
        if expires_at:
            try:
                exp = datetime.fromisoformat(str(expires_at).replace("Z", "+00:00"))
                if exp.tzinfo is None:
                    exp = exp.replace(tzinfo=timezone.utc)
                if exp.timestamp() > datetime.now(timezone.utc).timestamp():
                    agent_id_from_call_token = call_tok.get("agent_id")
                    overrides = call_tok.get("overrides") or {}
                    override_script = overrides.get("script")
                    override_customer_name = overrides.get("customer_name")
            except (ValueError, TypeError):
                pass

    def _fetch_agent_by_id(aid: str):
        return (
            client.table("ai_agents")
            .select(AGENT_COLS)
            .eq("id", aid)
            .maybe_single()
            .execute()
        )

    def _fetch_agent_by_voicebot_token():
        return (
            client.table("ai_agents")
            .select(AGENT_COLS)
            .eq("voicebot_token", token)
            .maybe_single()
            .execute()
        )

    if agent_id_from_call_token:
        agent_result = await _run_sync(_fetch_agent_by_id, agent_id_from_call_token)
    else:
        agent_result = await _run_sync(_fetch_agent_by_voicebot_token)

    agent = _extract_supabase_data(agent_result)
    if not agent:
        raise SupabaseAgentError("agent not found", status_code=404)
    if agent.get("status") and agent.get("status") != "active":
        raise SupabaseAgentError("agent inactive", status_code=403)

    base_sys = agent.get("system_prompt") or agent.get("script") or _DEFAULT_SYSTEM_PROMPT
    base_first = _base_first_message(agent)
    system_prompt = base_sys
    first_message = base_first
    if override_script:
        system_prompt = base_sys + build_campaign_prompt_suffix(
            override_script, override_customer_name
        )
        first_message = build_campaign_first_message(
            override_script, base_first, override_customer_name
        )

    enriched = dict(agent)
    if not enriched.get("greeting_audio_base64") and not override_script:
        greeting = await _fetch_greeting_from_moontech(agent["id"])
        if greeting:
            enriched = _merge_greeting(enriched, greeting)

    fields = _agent_bundle_fields(enriched)
    fields["system_prompt"] = system_prompt
    fields["first_message"] = first_message

    return {
        "agent_id": agent["id"],
        "user_id": agent.get("user_id"),
        "token": agent.get("voicebot_token") or token,
        **fields,
        "greeting_audio_base64": enriched.get("greeting_audio_base64"),
        "greeting_audio_mime": enriched.get("greeting_audio_mime")
        or (
            "audio/L16;rate=8000"
            if enriched.get("greeting_audio_base64")
            else None
        ),
        "greeting_audio_bytes": enriched.get("greeting_audio_bytes"),
        "greeting_audio_hash": enriched.get("greeting_audio_hash"),
        "greeting_audio_generated_at": enriched.get("greeting_audio_generated_at"),
        "greeting": _greeting_object(enriched, first_message),
        "customer_name": override_customer_name,
    }


async def fetch_greeting_from_moontech(agent_id: str) -> dict[str, Any] | None:
    """
    Cold-cache greeting synth via Moontech HTTP only (no ai_agents REST).
    Returns the greeting object or None — never raises.
    """
    try:
        return await _fetch_greeting_from_moontech(agent_id)
    except Exception as exc:
        logger.warning(
            "Moontech greeting fetch failed agent_id=%s: %s",
            agent_id[:8] if agent_id else "",
            exc,
        )
        return None


def _extract_supabase_data(result: Any) -> dict[str, Any] | None:
    """Safely read ``.data`` from a Supabase execute() result."""
    if result is None:
        return None
    data = getattr(result, "data", None)
    if isinstance(data, dict):
        return data
    return None


async def fetch_agent_payload_with_greeting(agent_id: str) -> dict[str, Any]:
    if not agent_id:
        raise SupabaseAgentError("missing agent_id", status_code=400)

    client = await _get_supabase()

    def _fetch_agent():
        return (
            client.table("ai_agents")
            .select(AGENT_COLS)
            .eq("id", agent_id)
            .maybe_single()
            .execute()
        )

    agent_result = await _run_sync(_fetch_agent)
    agent = _extract_supabase_data(agent_result)
    if not agent:
        logger.warning(
            "ai_agents lookup returned no row for agent_id=%s (RLS or missing)",
            agent_id[:8],
        )
        greeting = await fetch_greeting_from_moontech(agent_id)
        if not greeting:
            raise SupabaseAgentError("agent not found", status_code=404)
        first_message = (greeting.get("text") or "").strip() or "Hello!"
        enriched = _merge_greeting({"id": agent_id}, greeting)
        fields = _agent_bundle_fields(enriched)
        return {
            "agent_id": agent_id,
            "user_id": enriched.get("user_id") or "",
            "token": enriched.get("voicebot_token") or "",
            **fields,
            "first_message": first_message,
            "greeting": _greeting_object(enriched, first_message),
        }
    if agent.get("status") and agent.get("status") != "active":
        raise SupabaseAgentError("agent inactive", status_code=403)

    enriched = dict(agent)
    if not enriched.get("greeting_audio_base64"):
        greeting = await _fetch_greeting_from_moontech(agent_id)
        if greeting:
            enriched = _merge_greeting(enriched, greeting)

    first_message = _base_first_message(agent)
    fields = _agent_bundle_fields(enriched)

    return {
        "agent_id": agent["id"],
        "user_id": agent.get("user_id"),
        "token": agent.get("voicebot_token") or "",
        "name": agent.get("name") or "AI Voice Assistant",
        "business_name": agent.get("business_name") or "our team",
        **fields,
        "first_message": first_message,
        "greeting": _greeting_object(enriched, first_message),
    }
