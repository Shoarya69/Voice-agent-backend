#!/usr/bin/env python3
"""
Optional LiveKit Cloud agent worker (Gemini via LiveKit Agents plugin).

Vobiz/Exotel calls on this repo use ``ai_server.py`` + ``VOICE_REALTIME_ENGINE``
(embedded LiveKit Google plugin without a LiveKit room).

Run this worker only when you dispatch jobs to LiveKit rooms (SIP / WebRTC):

  export LIVEKIT_URL=wss://...
  export LIVEKIT_API_KEY=...
  export LIVEKIT_API_SECRET=...
  export GOOGLE_API_KEY=...   # or portal key
  python livekit_worker.py dev

See: https://github.com/livekit/agents
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("livekit_worker")


def main() -> None:
    from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
    from livekit.plugins import google

    from app import config
    from app.engine_config import gemini_voice_or_default

    server = AgentServer()

    class PhoneAssistant(Agent):
        def __init__(self) -> None:
            super().__init__(
                instructions=os.getenv(
                    "LIVEKIT_AGENT_INSTRUCTIONS",
                    "You are a helpful voice assistant on a live phone call.",
                )
            )

    @server.rtc_session(agent_name=os.getenv("LIVEKIT_AGENT_NAME", "moontech-gemini"))
    async def entrypoint(ctx: JobContext) -> None:
        model = (
            config.LIVEKIT_GEMINI_MODEL
            or config.GEMINI_LIVE_MODEL
            or "gemini-2.5-flash-native-audio-preview-12-2025"
        )
        voice = gemini_voice_or_default(
            config.GEMINI_LIVE_VOICE,
            config.GEMINI_LIVE_VOICE or "Aoede",
        )
        api_key = config.GEMINI_API_KEY or os.getenv("GOOGLE_API_KEY", "")

        session = AgentSession(
            llm=google.realtime.RealtimeModel(
                model=model,
                voice=voice,
                api_key=api_key or None,
            ),
        )
        await session.start(room=ctx.room, agent=PhoneAssistant())
        await ctx.connect()
        await session.generate_reply(
            instructions="Greet the caller briefly in one short sentence."
        )

    cli.run_app(server)


if __name__ == "__main__":
    main()
