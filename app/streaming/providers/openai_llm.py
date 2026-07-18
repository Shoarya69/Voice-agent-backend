"""OpenAI chat streaming LLM provider."""

from __future__ import annotations

from typing import AsyncIterator

from app.openai_service import BrainError, openai_brain
from app.streaming.providers.base import StreamingLLMProvider


class OpenAIStreamingLLMProvider(StreamingLLMProvider):
    async def stream_reply(self, history: list[dict]) -> AsyncIterator[str]:
        async for delta in openai_brain.stream_reply(history):
            yield delta


def brain_error_type():
    return BrainError
