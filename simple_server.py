#!/usr/bin/env python3
"""
Simple WebSocket Echo Server for Exotel Voice Streaming
======================================================

A minimal WebSocket server that echoes audio packets back to Exotel,
perfect for testing bidirectional voice streaming functionality.

Features:
- Handles all Exotel WebSocket events (connected, start, media, dtmf, stop)
- Echoes media packets back for bidirectional testing
- Comprehensive logging with timestamps
- Connection tracking and error handling
- Ready for production deployment

Usage:
    python3 simple_server.py

Requirements:
    - websockets>=12.0
    - Python 3.8+
"""

import asyncio
import websockets
import json
import logging
import os
from datetime import datetime, timezone
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Add file handler for persistent logging
file_handler = logging.FileHandler('logs/voice_bot_echo.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

class VoiceSession:
    """Manages individual voice call sessions with buffering and conversation flow"""
    
    def __init__(self, connection_id, websocket):
        self.connection_id = connection_id
        self.websocket = websocket
        self.audio_buffer = []
        self.is_listening = True
        self.is_speaking = False
        self.should_stop = False
        self.stream_sid = None
        self.call_sid = None
        self.silence_timer = None
        self.silence_threshold = 2.0  # 2 seconds of silence before responding
        
    def add_audio_chunk(self, media_data):
        """Add audio chunk to buffer during listening phase"""
        if self.is_listening and not self.should_stop:
            self.audio_buffer.append(media_data)
            logger.info(f"🎧 BUFFERING - Added chunk {media_data.get('chunk', 'N/A')} for {self.connection_id}")
            
            # Choose echo mode based on configuration
            if IMMEDIATE_ECHO_MODE:
                # IMMEDIATE ECHO: For traditional testing and compatibility
                asyncio.create_task(self.immediate_echo(media_data))
            else:
                # CONVERSATIONAL MODE: For AI-like behavior with silence detection
                # Reset silence timer
                if self.silence_timer:
                    self.silence_timer.cancel()
                
                # Start new silence timer
                self.silence_timer = asyncio.create_task(self.handle_silence())
    
    async def immediate_echo(self, media_data):
        """Immediate echo for testing purposes"""
        try:
            echo_response = {
                'event': 'media',
                'stream_sid': self.stream_sid,
                'media': media_data
            }
            await self.websocket.send(json.dumps(echo_response))
            logger.info(f"🔊 IMMEDIATE ECHO - Sent chunk {media_data.get('chunk', 'N/A')} back to {self.connection_id}")
        except Exception as e:
            logger.error(f"❌ Echo error for {self.connection_id}: {e}")
    
    async def handle_silence(self):
        """Handle silence detection - start responding after silence threshold"""
        try:
            await asyncio.sleep(self.silence_threshold)
            if self.is_listening and self.audio_buffer and not self.should_stop:
                logger.info(f"🤔 SILENCE DETECTED - Starting response for {self.connection_id}")
                await self.start_response()
        except asyncio.CancelledError:
            # Timer was cancelled due to new audio
            pass
    
    async def start_response(self):
        """Start responding with buffered audio"""
        if self.should_stop:
            return
            
        self.is_listening = False
        self.is_speaking = True
        
        logger.info(f"🗣️ SPEAKING - Starting response with {len(self.audio_buffer)} chunks for {self.connection_id}")
        
        # Send buffered audio back with slight delay for natural conversation
        for i, media_data in enumerate(self.audio_buffer):
            if self.should_stop:
                logger.info(f"⏹️ INTERRUPTED - Stopping response for {self.connection_id}")
                break
                
            # Send echo with conversational delay
            echo_response = {
                'event': 'media',
                'stream_sid': self.stream_sid,
                'media': media_data
            }
            
            await self.websocket.send(json.dumps(echo_response))
            logger.info(f"🔊 ECHO RESPONSE - Sent buffered chunk {i+1}/{len(self.audio_buffer)} for {self.connection_id}")
            
            # Small delay between chunks for natural speech
            await asyncio.sleep(0.1)
        
        # Reset for next conversation turn
        await self.reset_for_next_turn()
    
    async def reset_for_next_turn(self):
        """Reset session for next conversation turn"""
        if not self.should_stop:
            self.audio_buffer = []
            self.is_listening = True
            self.is_speaking = False
            logger.info(f"👂 LISTENING - Ready for next turn from {self.connection_id}")
    
    async def handle_clear(self):
        """Handle clear event - stop current activity immediately"""
        self.should_stop = True
        
        if self.silence_timer:
            self.silence_timer.cancel()
        
        if self.is_speaking:
            logger.info(f"🛑 CLEAR - Interrupting speech for {self.connection_id}")
        elif self.is_listening:
            logger.info(f"🧹 CLEAR - Clearing audio buffer for {self.connection_id}")
        
        # Clear buffer and reset
        self.audio_buffer = []
        self.is_listening = True
        self.is_speaking = False
        self.should_stop = False
        
        # Send clear acknowledgment
        clear_response = {
            'event': 'clear',
            'stream_sid': self.stream_sid
        }
        await self.websocket.send(json.dumps(clear_response))
        logger.info(f"✅ CLEAR ACKNOWLEDGED - Session reset for {self.connection_id}")

# Configuration: Set to True for immediate echo (testing), False for conversational mode
IMMEDIATE_ECHO_MODE = True  # Change to False for conversational AI behavior

# Global session storage
active_sessions = {}

async def handle_websocket(websocket):
    """Enhanced WebSocket handler with conversation flow management"""
    connection_id = f"conn_{int(datetime.now().timestamp() * 1000)}"
    logger.info(f"🔗 New WebSocket connection established: {connection_id}")
    
    # Create voice session
    session = VoiceSession(connection_id, websocket)
    active_sessions[connection_id] = session
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                event_type = data.get('event')
                
                logger.info(f"📨 Received event from {connection_id}: {event_type}")
                
                if event_type == 'connected':
                    logger.info(f"🎉 CONNECTED EVENT - Call connected for {connection_id}")
                    logger.info(f"   📊 Full connected data: {json.dumps(data, indent=2)}")
                    
                    # Log call connection
                    call_log_entry = {
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'connection_id': connection_id,
                        'event': 'connected'
                    }
                    
                    with open('logs/calls.log', 'a') as call_log:
                        call_log.write(json.dumps(call_log_entry) + '\n')
                
                elif event_type == 'start':
                    start_data = data.get('start', {})
                    session.stream_sid = start_data.get('stream_sid', data.get('stream_sid', 'N/A'))
                    session.call_sid = start_data.get('call_sid', 'N/A')
                    account_sid = start_data.get('account_sid', 'N/A')
                    from_number = start_data.get('from', 'N/A')
                    to_number = start_data.get('to', 'N/A')
                    media_format = start_data.get('media_format', {})
                    
                    logger.info(f"🚀 START EVENT - Stream started for {connection_id}")
                    logger.info(f"   📞 Call SID: {session.call_sid}")
                    logger.info(f"   📡 Stream SID: {session.stream_sid}")
                    logger.info(f"   🏢 Account SID: {account_sid}")
                    logger.info(f"   📲 From: {from_number} → To: {to_number}")
                    logger.info(f"   🎵 Media Format: {json.dumps(media_format)}")
                    logger.info(f"   📊 Full start data: {json.dumps(start_data, indent=2)}")
                    logger.info(f"👂 SESSION READY - Listening for audio from {connection_id}")
                
                elif event_type == 'media':
                    # Enhanced media handling with buffering
                    media_data = data.get('media', {})
                    payload = media_data.get('payload', '')
                    chunk = media_data.get('chunk', 'N/A')
                    timestamp = media_data.get('timestamp', 'N/A')
                    sequence_number = data.get('sequence_number', 'N/A')
                    stream_sid = data.get('stream_sid', session.stream_sid)
                    
                    payload_size = len(payload)
                    logger.info(f"🎵 MEDIA EVENT - Audio packet received from {connection_id}")
                    logger.info(f"   📦 Chunk: {chunk}, Timestamp: {timestamp}, Sequence: {sequence_number}")
                    logger.info(f"   📏 Payload size: {payload_size} bytes")
                    logger.info(f"   📡 Stream SID: {stream_sid}")
                    if payload_size > 0:
                        logger.info(f"   🔤 Payload preview: {payload[:50]}{'...' if len(payload) > 50 else ''}")
                    
                    # Add to session buffer instead of immediate echo
                    session.add_audio_chunk(media_data)
                
                elif event_type == 'clear':
                    # Enhanced clear handling
                    sequence_number = data.get('sequence_number', 'N/A')
                    stream_sid = data.get('stream_sid', session.stream_sid)
                    
                    logger.info(f"🧹 CLEAR EVENT - Clear command from {connection_id}")
                    logger.info(f"   📡 Stream SID: {stream_sid}")
                    logger.info(f"   📊 Sequence: {sequence_number}")
                    logger.info(f"   🗑️  Purpose: Stop current activity and clear buffer")
                    logger.info(f"   📊 Full clear data: {json.dumps(data, indent=2)}")
                    
                    # Handle clear through session
                    await session.handle_clear()
                
                elif event_type == 'dtmf':
                    dtmf_data = data.get('dtmf', {})
                    digit = dtmf_data.get('digit', 'unknown')
                    duration = dtmf_data.get('duration', 'N/A')
                    sequence_number = data.get('sequence_number', 'N/A')
                    stream_sid = data.get('stream_sid', session.stream_sid)
                    
                    logger.info(f"🔢 DTMF EVENT - Key press from {connection_id}")
                    logger.info(f"   🎹 Digit: {digit}")
                    logger.info(f"   ⏱️  Duration: {duration}ms")
                    logger.info(f"   📊 Sequence: {sequence_number}")
                    logger.info(f"   📡 Stream SID: {stream_sid}")
                    logger.info(f"   📊 Full DTMF data: {json.dumps(dtmf_data, indent=2)}")
                    
                    # Respond to DTMF event (acknowledge key press)
                    try:
                        dtmf_response = {
                            'event': 'dtmf',
                            'stream_sid': stream_sid,
                            'dtmf': {
                                'digit': digit,
                                'duration': duration
                            }
                        }
                        await websocket.send(json.dumps(dtmf_response))
                        logger.info(f"🔢 DTMF RESPONSE - Acknowledged digit '{digit}' to {connection_id}")
                    except Exception as dtmf_error:
                        logger.error(f"❌ Failed to send DTMF response: {dtmf_error}")
                
                elif event_type == 'mark':
                    mark_data = data.get('mark', {})
                    mark_name = mark_data.get('name', 'unknown')
                    sequence_number = data.get('sequence_number', 'N/A')
                    stream_sid = data.get('stream_sid', session.stream_sid)
                    
                    logger.info(f"📍 MARK EVENT - Mark received from {connection_id}")
                    logger.info(f"   🏷️  Mark Name: {mark_name}")
                    logger.info(f"   📡 Stream SID: {stream_sid}")
                    logger.info(f"   📊 Sequence: {sequence_number}")
                    logger.info(f"   📊 Full mark data: {json.dumps(mark_data, indent=2)}")
                    
                    # Respond to mark event
                    mark_response = {
                        'event': 'mark',
                        'stream_sid': stream_sid,
                        'mark': {
                            'name': mark_name
                        }
                    }
                    await websocket.send(json.dumps(mark_response))
                    logger.info(f"📍 MARK RESPONSE - Acknowledged mark '{mark_name}' to {connection_id}")
                
                elif event_type == 'stop':
                    stop_data = data.get('stop', {})
                    reason = stop_data.get('reason', 'unknown')
                    call_sid = stop_data.get('call_sid', session.call_sid)
                    account_sid = stop_data.get('account_sid', 'N/A')
                    sequence_number = data.get('sequence_number', 'N/A')
                    
                    logger.info(f"🛑 STOP EVENT - Stream ended for {connection_id}")
                    logger.info(f"   🔚 Reason: {reason}")
                    logger.info(f"   📞 Call SID: {call_sid}")
                    logger.info(f"   🏢 Account SID: {account_sid}")
                    logger.info(f"   📊 Sequence: {sequence_number}")
                    logger.info(f"   📊 Full stop data: {json.dumps(stop_data, indent=2)}")
                    
                    # Stop session
                    session.should_stop = True
                
                else:
                    logger.warning(f"❓ UNKNOWN EVENT - Unrecognized event from {connection_id}")
                    logger.warning(f"   🔤 Event type: {event_type}")
                    logger.warning(f"   📊 Full data: {json.dumps(data, indent=2)}")
                    
            except json.JSONDecodeError as e:
                logger.error(f"❌ Invalid JSON from {connection_id}: {e}")
            except Exception as e:
                logger.error(f"❌ Error processing message from {connection_id}: {e}")
                
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"👋 Connection closed normally: {connection_id}")
    except Exception as e:
        logger.error(f"❌ Connection error for {connection_id}: {e}")
    finally:
        # Cleanup session
        if connection_id in active_sessions:
            session = active_sessions[connection_id]
            session.should_stop = True
            if session.silence_timer:
                session.silence_timer.cancel()
            del active_sessions[connection_id]
        
        logger.info(f"🔚 Connection ended: {connection_id}")

async def main():
    """
    Main server function that starts the WebSocket server.
    """
    port = 5000
    host = "0.0.0.0"
    
    logger.info(f"🚀 Starting Voice Bot Echo Server...")
    logger.info(f"📡 Server will listen on {host}:{port}")
    logger.info(f"🌐 WebSocket endpoint: ws://{host}:{port}")
    logger.info(f"📝 Logs will be saved to: logs/voice_bot_echo.log")
    
    server = await websockets.serve(
        handle_websocket,
        host,
        port,
        ping_interval=30,  # Send ping every 30 seconds
        ping_timeout=10    # Wait 10 seconds for pong response
    )
    
    logger.info(f"✅ Voice Bot Echo Server running at ws://{host}:{port}")
    logger.info(f"🎯 Ready to receive Exotel voice streaming connections!")
    
    try:
        await server.wait_closed()
    except KeyboardInterrupt:
        logger.info("🛑 Server shutdown requested by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Voice Bot Echo Server shutting down...")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}") 