#!/usr/bin/env python3
"""
AI Voice Agent WebSocket smoke test.

This is different from test_connection.py:
- test_connection.py validates the old echo server and expects the same audio
  payload back.
- ai_server.py is not an echo server. It returns ElevenLabs-generated speech,
  so the payload should be different from what we send.

Usage:
    python3 test_ai_connection.py ws://localhost:8002
    python3 test_ai_connection.py ws://80.241.209.69:8002
"""

import asyncio
import json
import sys

import websockets


async def recv_until_media(websocket, timeout=8.0):
    """Read messages until a media event is received or timeout expires."""
    while True:
        message = await asyncio.wait_for(websocket.recv(), timeout=timeout)
        data = json.loads(message)
        event = data.get("event")
        print(f"📥 Received event: {event}")
        if event == "media":
            payload = data.get("media", {}).get("payload", "")
            print(f"✅ Received AI audio payload ({len(payload)} base64 chars)")
            return data


async def test_ai_server(uri):
    print("🧪 AI Voice Agent Smoke Test")
    print("=" * 40)
    print(f"Target URI: {uri}")
    print("=" * 40)

    async with websockets.connect(uri, open_timeout=10) as websocket:
        print("✅ Connected to AI voice agent")

        connected_event = {"event": "connected"}
        await websocket.send(json.dumps(connected_event))
        print("📤 Sent connected event")

        start_event = {
            "event": "start",
            "sequence_number": 1,
            "stream_sid": "test_stream_ai",
            "start": {
                "stream_sid": "test_stream_ai",
                "call_sid": "test_call_ai",
                "account_sid": "test_account_ai",
                "from": "+1234567890",
                "to": "+0987654321",
                "media_format": {
                    "encoding": "raw/slin",
                    "sample_rate": "8000",
                    "bit_rate": "16",
                },
            },
        }
        await websocket.send(json.dumps(start_event))
        print("📤 Sent start event")

        # If ENABLE_WELCOME_MESSAGE=true, the AI server should answer with
        # generated TTS audio shortly after START.
        media_response = await recv_until_media(websocket)
        if media_response.get("stream_sid") != "test_stream_ai":
            print("⚠️  Media stream_sid mismatch, but server did send audio")

        stop_event = {
            "event": "stop",
            "sequence_number": 2,
            "stream_sid": "test_stream_ai",
            "stop": {
                "call_sid": "test_call_ai",
                "account_sid": "test_account_ai",
                "reason": "test_completed",
            },
        }
        await websocket.send(json.dumps(stop_event))
        print("📤 Sent stop event")

    print("\n🎉 AI server smoke test passed.")
    print("📋 WebSocket, START handling, ElevenLabs TTS response are working.")


async def main():
    uri = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8002"
    try:
        await test_ai_server(uri)
    except Exception as exc:
        print(f"\n❌ AI smoke test failed: {type(exc).__name__}: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
