"""
Shared tool/function-calling registry for all voice pipeline providers.

Register tool definitions here once; each provider adapter maps them to its
native API (OpenAI chat tools for Compound, Realtime GA ``tools`` for Realtime).
"""

from __future__ import annotations

END_CALL_TOOL_NAME = "end_call"


def build_gemini_live_tools() -> list:
    """Gemini Live ``setup.tools`` — synchronous function calling."""
    return [
        {
            "functionDeclarations": [
                {
                    "name": END_CALL_TOOL_NAME,
                    "description": (
                        "Disconnect the live phone call. Use only after you have "
                        "spoken a short polite goodbye when the caller wants to hang up "
                        "or the conversation is naturally finished."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "Short reason, e.g. caller_goodbye",
                            }
                        },
                    },
                }
            ]
        }
    ]


def gemini_live_system_instructions(
    system_prompt: str,
    *,
    language: str = "",
    closing_message: str = "",
) -> str:
    from app import config

    start_language = (language or "").strip()
    if start_language:
        language_rule = (
            f"Start the call in {start_language}. "
            "Reply in the caller's language. If they switch language, switch with them."
        )
    else:
        language_rule = (
            "Reply in the caller's language. Match the language they are speaking."
        )
    closing = (closing_message or "").strip()
    closing_rule = f" When the call ends, you may close with: {closing}" if closing else ""
    return (
        f"{system_prompt}\n\n"
        "--- GEMINI LIVE PHONE RULES ---\n"
        "Speak like a person on a live phone call, warm, natural, and concise. "
        "CRITICAL: Never output internal reasoning, preambles, or text thoughts like 'Crafting A Response'. "
        "Respond immediately with direct spoken voice in real-time.\n"
        f"{language_rule}{closing_rule} "
        "Ignore automated carrier announcements such as 'this call is now being recorded'. "
        "Ask only one short question at a time and wait for the caller to reply.\n"
        f"{config.GEMINI_LIVE_CALL_END_INSTRUCTION}"
    )


def build_realtime_tools() -> list:
    """
    GA Realtime API tool definitions for ``session.update`` / ``response.create``.

    Format: https://developers.openai.com/api/reference/resources/realtime/client-events
    """
    return []


def build_chat_tools() -> list:
    """OpenAI Chat Completions tool definitions for the Compound pipeline."""
    return []
