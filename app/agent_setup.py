"""
Agent setup helpers: load MoontechPro agent bundles onto a VoiceSession.

Separated from ai_server.py so webhook prewarm and WSS prefetch share the
same resolution path without duplicating session field assignment.
"""

from __future__ import annotations

import asyncio
import logging
import time

from app import lovable_client
from app.voice_pipeline.base import VoicePipelineSession

logger = logging.getLogger(__name__)


def _token_prefix(token: str | None) -> str:
    return (token or "")[:6]


async def load_agent_bundle_into_session(
    session: VoicePipelineSession,
    connection_id: str,
    *,
    token: str | None = None,
    number: str | None = None,
) -> bool:
    """
    Resolve agent + greeting via lovable_client.resolve_agent_bundle (cached /
    singleflight) and populate session fields. Does not set config overrides.
    """
    try:
        agent_config = await lovable_client.resolve_agent_bundle(
            token=token,
            number=number,
        )
    except lovable_client.LovableClientError as exc:
        if token:
            logger.error(
                "❌ Lovable agent lookup failed for %s (token=%s...): %s",
                connection_id,
                _token_prefix(token),
                exc,
            )
        else:
            logger.error(
                "❌ Lovable agent-by-number lookup failed for %s (number=...%s): %s",
                connection_id,
                (number or "")[-4:],
                exc,
            )
        return False

    session.agent_id = agent_config.agent_id
    session.agent_config = agent_config
    session.greeting_audio_pcm = agent_config.greeting_audio_pcm
    if agent_config.token:
        session.lovable_token = agent_config.token

    if agent_config.greeting_audio_pcm:
        logger.info(
            "🎵 Pre-stored greeting audio loaded for %s agent_id=%s bytes=%s",
            connection_id,
            agent_config.agent_id,
            len(agent_config.greeting_audio_pcm),
        )

    route_label = f"token={_token_prefix(token)}" if token else f"number=...{(number or '')[-4:]}"
    logger.info(
        "🔑 Agent resolved for %s: agent_id=%s %s system_prompt_chars=%s",
        connection_id,
        agent_config.agent_id,
        route_label,
        len(getattr(agent_config, "system_prompt", "") or ""),
    )
    return True


def start_agent_prefetch(
    session: VoicePipelineSession,
    connection_id: str,
) -> None:
    """Begin bundle resolution at WSS connect (overlaps with Exotel handshake)."""
    session._agent_prefetch_task = None
    session._prefetch_started_at = time.perf_counter()
    kind = session.wss_route_kind
    value = session.wss_route_value
    if kind == "token" and value:
        logger.info("⚡ Prefetching agent bundle for %s (route=token)", connection_id)
        session._agent_prefetch_task = asyncio.create_task(
            load_agent_bundle_into_session(
                session, connection_id, token=value
            )
        )
    elif kind == "number" and value:
        logger.info("⚡ Prefetching agent bundle for %s (route=number)", connection_id)
        session._agent_prefetch_task = asyncio.create_task(
            load_agent_bundle_into_session(
                session, connection_id, number=value
            )
        )
