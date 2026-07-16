#!/usr/bin/env python3
"""
Test Client for Voice Bot Echo Server
=====================================

This script tests the WebSocket echo server by sending sample
Exotel voice streaming events and verifying responses.

Usage:
    python3 test_connection.py [ws://localhost:5000]
"""

import asyncio
import websockets
import json
import sys
import base64
from datetime import datetime

# Sample test data that mimics Exotel's voice streaming protocol
TEST_EVENTS = [
    {
        "event": "connected",
        "timestamp": datetime.now().isoformat()
    },
    {
        "event": "start",
        "sequence_number": 1,
        "stream_sid": "test_stream_12345",
        "start": {
            "stream_sid": "test_stream_12345",
            "call_sid": "test_call_67890",
            "account_sid": "test_account_11111",
            "from": "+1234567890",
            "to": "+0987654321",
            "custom_parameters": {
                "test_mode": "true",
                "echo_test": "active"
            },
            "media_format": {
                "encoding": "raw/slin",
                "sample_rate": "8000",
                "bit_rate": "16"
            }
        }
    },
    {
        "event": "media",
        "sequence_number": 2,
        "stream_sid": "test_stream_12345",
        "media": {
            "chunk": 1,
            "timestamp": "100",
            "payload": base64.b64encode(b"test_audio_data_chunk_1").decode()
        }
    },
    {
        "event": "media",
        "sequence_number": 3,
        "stream_sid": "test_stream_12345",
        "media": {
            "chunk": 2,
            "timestamp": "200",
            "payload": base64.b64encode(b"test_audio_data_chunk_2").decode()
        }
    },
    {
        "event": "dtmf",
        "sequence_number": 4,
        "stream_sid": "test_stream_12345",
        "dtmf": {
            "duration": "100",
            "digit": "1"
        }
    },
    {
        "event": "stop",
        "sequence_number": 5,
        "stream_sid": "test_stream_12345",
        "stop": {
            "call_sid": "test_call_67890",
            "account_sid": "test_account_11111",
            "reason": "callended"
        }
    }
]

async def test_echo_server(uri):
    """
    Test the echo server by sending sample events and checking responses.
    """
    print(f"🔗 Connecting to: {uri}")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to echo server!")
            
            media_responses = 0
            
            # Send test events
            for i, event in enumerate(TEST_EVENTS, 1):
                print(f"\n📤 Sending event {i}: {event['event']}")
                await websocket.send(json.dumps(event))
                
                # For media events, expect an echo response
                if event['event'] == 'media':
                    try:
                        # Wait for echo response
                        response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                        response_data = json.loads(response)
                        
                        if response_data.get('event') == 'media':
                            media_responses += 1
                            print(f"✅ Received media echo #{media_responses}")
                            
                            # Verify the echoed data matches
                            original_payload = event['media']['payload']
                            echoed_payload = response_data.get('media', {}).get('payload', '')
                            
                            if original_payload == echoed_payload:
                                print(f"✅ Payload verification passed")
                            else:
                                print(f"❌ Payload mismatch!")
                        else:
                            print(f"❓ Unexpected response: {response_data.get('event')}")
                            
                    except asyncio.TimeoutError:
                        print(f"⏰ No response received within timeout")
                    except Exception as e:
                        print(f"❌ Error processing response: {e}")
                
                # Small delay between events
                await asyncio.sleep(0.5)
            
            print(f"\n🎯 Test Summary:")
            print(f"   📤 Sent events: {len(TEST_EVENTS)}")
            print(f"   📨 Media events sent: {sum(1 for e in TEST_EVENTS if e['event'] == 'media')}")
            print(f"   🔊 Media echoes received: {media_responses}")
            
            if media_responses == sum(1 for e in TEST_EVENTS if e['event'] == 'media'):
                print(f"✅ All tests passed! Echo server is working correctly.")
            else:
                print(f"⚠️  Some echo responses were missing.")
                
    except (websockets.exceptions.ConnectionClosed, ConnectionRefusedError, OSError) as e:
        print(f"❌ Connection error: {e}")
        print(f"💡 Is the server running? Try: python3 simple_server.py")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print(f"Error type: {type(e)}")
        return False
    
    return True

async def main():
    # Get WebSocket URI from command line or use default
    uri = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:5000"
    
    print("🧪 Voice Bot Echo Server Test")
    print("=" * 40)
    print(f"Target URI: {uri}")
    print(f"Test events: {len(TEST_EVENTS)}")
    print("=" * 40)
    
    success = await test_echo_server(uri)
    
    if success:
        print("\n🎉 Test completed successfully!")
        print("📋 Your echo server is ready for Exotel integration!")
    else:
        print("\n❌ Test failed. Please check server status and try again.")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user.")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        sys.exit(1) 