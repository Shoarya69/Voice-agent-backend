"""Parse Vobiz ``start`` and webhook fields for routing and CRM."""

from __future__ import annotations

from typing import Any


def extract_stream_id(message: dict[str, Any], start_data: dict[str, Any]) -> str | None:
    for key in ("streamId", "streamID", "StreamId", "StreamID"):
        if start_data.get(key):
            return str(start_data[key])
        if message.get(key):
            return str(message[key])
    return None


def extract_parties(message: dict[str, Any], start_data: dict[str, Any]) -> tuple[str | None, str | None]:
    """Return (from, to) from Vobiz start payload or root message."""
    from_num = (
        start_data.get("from")
        or start_data.get("From")
        or message.get("from")
        or message.get("From")
    )
    to_num = (
        start_data.get("to")
        or start_data.get("To")
        or message.get("to")
        or message.get("To")
    )
    for custom in (
        start_data.get("customParameters"),
        start_data.get("custom_parameters"),
        start_data.get("custom"),
    ):
        if isinstance(custom, dict):
            from_num = from_num or custom.get("from") or custom.get("From")
            to_num = to_num or custom.get("to") or custom.get("To")
    return (
        str(from_num) if from_num else None,
        str(to_num) if to_num else None,
    )


def extract_call_sid(message: dict[str, Any], start_data: dict[str, Any]) -> str | None:
    for key in ("callId", "callUUID", "CallUUID", "CallSid", "call_sid"):
        if start_data.get(key):
            return str(start_data[key])
        if message.get(key):
            return str(message[key])
    return None
