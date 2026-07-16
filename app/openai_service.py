"""
OpenAI integration: acts as the "brain" of the voice agent, turning the
caller's transcribed text into a natural-language reply, using the
conversation history for context.
"""

import logging
from typing import AsyncIterator

import httpx
from openai import AsyncOpenAI, OpenAIError

from app import config
from app.http_limits import default_limits, default_timeout

logger = logging.getLogger(__name__)


class BrainError(Exception):
    """Raised when the OpenAI call fails."""


class OpenAIBrain:
    def __init__(self):
        self._client: AsyncOpenAI | None = None
        if config.OPENAI_API_KEY:
            http_client = httpx.AsyncClient(
                limits=default_limits(),
                timeout=default_timeout(60.0),
            )
            self._client = AsyncOpenAI(
                api_key=config.OPENAI_API_KEY,
                http_client=http_client,
            )
            self._http_client = http_client
        else:
            self._http_client = None

    async def close(self) -> None:
        if self._http_client is not None and not self._http_client.is_closed:
            await self._http_client.aclose()

    @staticmethod
    def _messages(history: list) -> list:
        # Agent/MoontechPro system_prompt may be long; mandatory platform rules
        # are always appended last so reply-length and call-end caps win.
        system_prompt = (
            f"{config.SYSTEM_PROMPT}\n\n"
            f"{config.MANDATORY_PLATFORM_RULES_HEADER}\n"
            f"{config.REPLY_LENGTH_INSTRUCTION}\n\n"
            f"{config.CALL_BEHAVIOR_INSTRUCTION}\n\n"
            f"{config.CALL_END_INSTRUCTION}"
        )
        return [{"role": "system", "content": system_prompt}] + history

    async def get_reply(self, history: list) -> str:
        """
        Generate the assistant's next reply.

        `history` is a list of {"role": "user"|"assistant", "content": str}
        dicts (the system prompt is added automatically).
        """
        if self._client is None:
            raise BrainError("OPENAI_API_KEY is not configured")

        messages = self._messages(history)

        try:
            response = await self._client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=messages,
                temperature=config.OPENAI_TEMPERATURE,
                max_tokens=config.OPENAI_MAX_TOKENS,
            )
        except OpenAIError as exc:
            logger.error("OpenAI request failed: %s", exc)
            raise BrainError(str(exc)) from exc

        reply = response.choices[0].message.content or ""
        return reply.strip()

    async def stream_reply(self, history: list) -> AsyncIterator[str]:
        """
        Generate the assistant's next reply, yielding text deltas as they
        arrive. This lets the caller start speaking the first sentence while
        the model is still generating the rest, which is the single biggest
        lever for cutting perceived response latency on a live call.
        """
        if self._client is None:
            raise BrainError("OPENAI_API_KEY is not configured")

        messages = self._messages(history)

        try:
            stream = await self._client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=messages,
                temperature=config.OPENAI_TEMPERATURE,
                max_tokens=config.OPENAI_MAX_TOKENS,
                stream=True,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except OpenAIError as exc:
            logger.error("OpenAI streaming request failed: %s", exc)
            raise BrainError(str(exc)) from exc


openai_brain = OpenAIBrain()
