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


def gemini_live_system_instructions(system_prompt: str) -> str:
    from app import config

    return (
        f"{system_prompt}\n\n"
        "--- GEMINI LIVE PHONE RULES ---\n"
        "Speak with a warm, natural, conversational Indian Hinglish/Hindi phone voice — "
        "like a real human executive talking on a call. "
        "CRITICAL LANGUAGE RULE: Always reply in Hindi/Hinglish unless the caller explicitly "
        "speaks a full sentence in English. Never switch to English or Spanish on short/unclear audio. "
        "If caller audio is unclear or just background noise, politely ask in Hindi: "
        "'Maaf kijiye, aapki aawaz clear nahi aayi, kya aap dobara bolenge?' "
        "Ignore any automated carrier announcements like 'this call is now being recorded'. "
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
