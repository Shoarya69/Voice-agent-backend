"""
Direct Supabase reads for agent config (replaces Moontech HTTP GET endpoints).

Mirrors the Moontech routes:
  - agent-by-number
  - agent-by-token
  - agent-with-greeting (DB read; Moontech HTTP only for cold greeting synth)

Write paths (call-log, channel release, webhooks) stay on Moontech via lovable_client.
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

_PHONE_NUMBER_COLS = (
    "id, user_id, number, default_inbound_agent_id, agent_id, fallback_agent_id, "
    "inbound_enabled"
)

_DEFAULT_VOICE_ID = "cgSgspJ2msm6clMCkdW9"
_DEFAULT_SYSTEM_PROMPT = "You are a helpful voice assistant."

_supabase_client = None
_supabase_lock = asyncio.Lock()


class SupabaseAgentError(Exception):
    """Raised when a Supabase agent lookup fails."""

    def __init__(self, message: str, *, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code


def is_configured() -> bool:
    return bool(config.SUPABASE_URL and config.SUPABASE_SERVICE_ROLE_KEY)


async def _get_supabase():
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client
    async with _supabase_lock:
        if _supabase_client is None:
            from supabase import create_client

            _supabase_client = create_client(
                config.SUPABASE_URL,
                config.SUPABASE_SERVICE_ROLE_KEY,
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
    return await asyncio.to_thread(fn, *args, **kwargs)


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


async def fetch_agent_payload_by_number(number: str) -> dict[str, Any]:
    e164 = normalize_phone(number)
    if not e164:
        raise SupabaseAgentError("invalid number", status_code=400)

    client = await _get_supabase()
    last10 = re.sub(r"\D", "", e164)[-10:]

    def _query():
        return (
            client.table("phone_numbers")
            .select(_PHONE_NUMBER_COLS)
            .or_(f"number.eq.{e164},number.ilike.%{last10}")
            .limit(1)
            .execute()
        )

    result = await _run_sync(_query)
    pn_list = result.data or []
    pn = pn_list[0] if pn_list else None
    if not pn:
        raise SupabaseAgentError(f"number_not_found: {e164}", status_code=404)
    if pn.get("inbound_enabled") is False:
        raise SupabaseAgentError("inbound_disabled", status_code=403)

    agent_id = (
        pn.get("default_inbound_agent_id")
        or pn.get("agent_id")
        or pn.get("fallback_agent_id")
    )
    user_id = pn.get("user_id")

    def _fetch_agent_by_id(aid: str):
        return (
            client.table("ai_agents")
            .select(AGENT_COLS)
            .eq("id", aid)
            .eq("user_id", user_id)
            .eq("status", "active")
            .maybe_single()
            .execute()
        )

    def _fetch_latest_agent():
        return (
            client.table("ai_agents")
            .select(AGENT_COLS)
            .eq("user_id", user_id)
            .eq("status", "active")
            .order("created_at", desc=True)
            .limit(1)
            .maybe_single()
            .execute()
        )

    agent = None
    if agent_id:
        agent_result = await _run_sync(_fetch_agent_by_id, agent_id)
        agent = agent_result.data

    if not agent:
        latest_result = await _run_sync(_fetch_latest_agent)
        agent = latest_result.data

    if not agent:
        raise SupabaseAgentError("no_agent_configured", status_code=404)

    fields = _agent_bundle_fields(agent)
    return {
        "phone_number_id": pn.get("id"),
        "number": pn.get("number"),
        "voicebot_token": agent.get("voicebot_token"),
        "token": agent.get("voicebot_token") or "",
        "agent_id": agent["id"],
        "user_id": agent.get("user_id") or user_id,
        **fields,
        "greeting": _greeting_object(agent, fields["first_message"]),
    }


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
    call_tok = call_tok_result.data
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

    agent = agent_result.data
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
    agent = agent_result.data
    if not agent:
        raise SupabaseAgentError("agent not found", status_code=404)
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
