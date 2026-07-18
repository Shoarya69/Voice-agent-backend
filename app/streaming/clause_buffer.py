"""Split streaming LLM tokens into speakable clauses for early TTS."""

from __future__ import annotations

from app import config


def split_next_clause(buffer: str) -> tuple[str, str] | None:
    """
    Return (clause, remainder) when enough text exists to start TTS.

    Prefer natural boundaries (comma, semicolon, sentence end) but flush
    early when the buffer grows long enough to avoid waiting for punctuation.
    """
    boundary_chars = ",.;!?।\n"
    min_words = max(1, config.STREAMING_CLAUSE_MIN_WORDS)
    max_chars = max(16, config.STREAMING_CLAUSE_MAX_CHARS)

    for i, ch in enumerate(buffer):
        if ch in boundary_chars and i >= 2:
            clause = buffer[: i + 1].strip()
            if len(clause.split()) >= min_words or ch in ".!?।\n":
                return clause, buffer[i + 1 :]

    if len(buffer) >= max_chars:
        idx = buffer.rfind(" ", 0, max_chars)
        if idx > 4:
            return buffer[:idx].strip(), buffer[idx + 1 :]

    word_count = len(buffer.split())
    if word_count >= min_words and len(buffer) >= max_chars // 2:
        idx = buffer.rfind(" ")
        if idx > 4:
            return buffer[:idx].strip(), buffer[idx + 1 :]

    return None
