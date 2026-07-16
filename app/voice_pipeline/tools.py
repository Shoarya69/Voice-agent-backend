"""
Shared tool/function-calling registry for all voice pipeline providers.

Register tool definitions here once; each provider adapter maps them to its
native API (OpenAI chat tools for Compound, Realtime GA ``tools`` for Realtime).
"""

from __future__ import annotations


def build_realtime_tools() -> list:
    """
    GA Realtime API tool definitions for ``session.update`` / ``response.create``.

    Format: https://developers.openai.com/api/reference/resources/realtime/client-events
    """
    return []


def build_chat_tools() -> list:
    """OpenAI Chat Completions tool definitions for the Compound pipeline."""
    return []
