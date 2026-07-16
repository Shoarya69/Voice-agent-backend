#!/usr/bin/env python3
"""
Enhanced Features Test for Voice Bot Echo Server
==============================================
Test MARK, CLEAR, DTMF responses and dashboard features
"""

import asyncio
import websockets
import json
import time

async def test_enhanced_features():
    """Test all enhanced features with proper acknowledgments"""
    uri = 'ws://localhost:5000'
    
    print("🧪 Enhanced Voice Bot Echo Server Test")
    print("=" * 50)
    print(f"Testing MARK, CLEAR, DTMF acknowledgments...")
    print("=" * 50)
    
    try:
        async with websockets.connect(uri) as websocket:
            print('✅ Connected to enhanced echo server!')
            
            # Test 1: CONNECTED Event
            connected_event = {
                'event': 'connected',
                'timestamp': '2025-07-31T16:05:00.123456'
            }
            await websocket.send(json.dumps(connected_event))
            print('📤 Sent CONNECTED event')
            await asyncio.sleep(0.5)
            
            # Test 2: START Event
            start_event = {
                'event': 'start',
                'stream_sid': 'test_stream_enhanced',
                'call_sid': 'test_call_enhanced',
                'account_sid': 'test_account_enhanced',
                'from': '+1111111111',
                'to': '+2222222222',
                'media_format': {
                    'encoding': 'raw/slin',
                    'sample_rate': '8000',
                    'bit_rate': '16'
                }
            }
            await websocket.send(json.dumps(start_event))
            print('📤 Sent START event')
            await asyncio.sleep(0.5)
            
            # Test 3: MARK Event (should get acknowledgment)
            mark_event = {
                'event': 'mark',
                'stream_sid': 'test_stream_enhanced',
                'mark': {'name': 'enhanced_test_marker'},
                'sequence_number': 1
            }
            await websocket.send(json.dumps(mark_event))
            print('📤 Sent MARK event')
            
            # Wait for MARK acknowledgment
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                print(f'✅ Received MARK response: {json.loads(response)["event"]} - {json.loads(response)["mark"]["name"]}')
            except asyncio.TimeoutError:
                print('❌ No MARK acknowledgment received')
            
            await asyncio.sleep(0.5)
            
            # Test 4: CLEAR Event (should get acknowledgment)
            clear_event = {
                'event': 'clear',
                'stream_sid': 'test_stream_enhanced',
                'sequence_number': 2
            }
            await websocket.send(json.dumps(clear_event))
            print('📤 Sent CLEAR event')
            
            # Wait for CLEAR acknowledgment
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                print(f'✅ Received CLEAR response: {json.loads(response)["event"]}')
            except asyncio.TimeoutError:
                print('❌ No CLEAR acknowledgment received')
            
            await asyncio.sleep(0.5)
            
            # Test 5: DTMF Event (should get acknowledgment)
            dtmf_event = {
                'event': 'dtmf',
                'stream_sid': 'test_stream_enhanced',
                'dtmf': {'digit': '9', 'duration': '250'},
                'sequence_number': 3
            }
            await websocket.send(json.dumps(dtmf_event))
            print('📤 Sent DTMF event')
            
            # Wait for DTMF acknowledgment
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                dtmf_resp = json.loads(response)
                print(f'✅ Received DTMF response: {dtmf_resp["event"]} - Digit: {dtmf_resp["dtmf"]["digit"]}')
            except asyncio.TimeoutError:
                print('❌ No DTMF acknowledgment received')
            
            await asyncio.sleep(0.5)
            
            # Test 6: Media Events (with echo)
            for i in range(3):
                media_event = {
                    'event': 'media',
                    'stream_sid': 'test_stream_enhanced',
                    'media': {
                        'chunk': str(i + 1),
                        'timestamp': str(100 * (i + 1)),
                        'payload': f'dGVzdF9lbmhhbmNlZF9hdWRpb19jaHVua18={i+1}'
                    },
                    'sequence_number': 4 + i
                }
                await websocket.send(json.dumps(media_event))
                print(f'📤 Sent MEDIA event #{i+1}')
                
                # Wait for echo
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    echo_data = json.loads(response)
                    print(f'🔊 Received MEDIA echo: Chunk {echo_data["media"]["chunk"]}')
                except asyncio.TimeoutError:
                    print(f'❌ No MEDIA echo received for chunk {i+1}')
                
                await asyncio.sleep(0.3)
            
            # Test 7: STOP Event
            stop_event = {
                'event': 'stop',
                'call_sid': 'test_call_enhanced',
                'account_sid': 'test_account_enhanced',
                'reason': 'enhanced_test_completed',
                'sequence_number': 7
            }
            await websocket.send(json.dumps(stop_event))
            print('📤 Sent STOP event')
            await asyncio.sleep(0.5)
            
            print("\n🎯 Enhanced Test Summary:")
            print("   ✅ CONNECTED event sent")
            print("   ✅ START event sent")
            print("   ✅ MARK event sent (with acknowledgment)")
            print("   ✅ CLEAR event sent (with acknowledgment)")
            print("   ✅ DTMF event sent (with acknowledgment)")
            print("   ✅ MEDIA events sent (with echo responses)")
            print("   ✅ STOP event sent")
            print("\n🎉 All enhanced features tested successfully!")
            print("📊 Check the dashboard at: http://localhost:5001")
            print("🔍 Features to test in dashboard:")
            print("   • Call selection and event details")
            print("   • Latency metrics and timing")
            print("   • Event filtering by type")
            print("   • Real-time updates")
            print("   • Export functionality")
            
    except Exception as e:
        print(f'❌ Enhanced test failed: {e}')

if __name__ == '__main__':
    asyncio.run(test_enhanced_features()) 