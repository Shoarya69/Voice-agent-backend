#!/usr/bin/env python3
"""
AgentStream Dashboard
====================
Real-time monitoring dashboard with latency tracking and advanced analytics
"""

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import json
import os
import threading
import time
from datetime import datetime, timezone
import re
from collections import defaultdict, deque
import subprocess

app = Flask(__name__)
app.config['SECRET_KEY'] = 'voicebot_echo_dashboard'
socketio = SocketIO(app, cors_allowed_origins="*")

# Enhanced global data storage
live_stats = {
    'total_calls': 0,
    'active_connections': 0,
    'total_media_packets': 0,
    'total_events': 0,
    'calls_per_hour': 0,
    'last_call_time': None,
    'server_uptime': datetime.now(timezone.utc),
    'avg_latency': 0,
    'first_media_latency': 0,
    'end_to_end_latency': 0
}

recent_events = deque(maxlen=200)  # Keep last 200 events
call_history = []
connection_stats = defaultdict(int)
event_counts = defaultdict(int)
call_sessions = {}  # Track individual call sessions with events
latency_data = []

class CallSession:
    def __init__(self, connection_id):
        self.connection_id = connection_id
        self.events = []
        self.start_time = None
        self.first_media_time = None
        self.end_time = None
        self.latencies = []
        
    def add_event(self, event):
        """Add event and calculate latencies"""
        current_time = datetime.now(timezone.utc)
        event['received_time'] = current_time
        
        # Calculate inter-event latency
        if self.events:
            last_event_time = self.events[-1]['received_time']
            inter_event_latency = (current_time - last_event_time).total_seconds() * 1000
            event['inter_event_latency_ms'] = round(inter_event_latency, 2)
            self.latencies.append(inter_event_latency)
        else:
            event['inter_event_latency_ms'] = 0
            self.start_time = current_time
            
        # Track first media event latency
        if event['type'] == 'media' and not self.first_media_time:
            self.first_media_time = current_time
            if self.start_time:
                first_media_latency = (current_time - self.start_time).total_seconds() * 1000
                event['first_media_latency_ms'] = round(first_media_latency, 2)
                live_stats['first_media_latency'] = round(first_media_latency, 2)
                
        # Calculate end-to-end latency for stop events
        if event['type'] == 'stop' and self.start_time:
            self.end_time = current_time
            end_to_end_latency = (current_time - self.start_time).total_seconds() * 1000
            event['end_to_end_latency_ms'] = round(end_to_end_latency, 2)
            live_stats['end_to_end_latency'] = round(end_to_end_latency, 2)
            
        self.events.append(event)
        
        # Update average latency
        if self.latencies:
            live_stats['avg_latency'] = round(sum(self.latencies) / len(self.latencies), 2)

def tail_logs():
    """Enhanced log monitoring with multiline parsing"""
    log_file = 'logs/voice_bot_echo.log'
    line_buffer = []
    
    while True:
        try:
            if os.path.exists(log_file):
                process = subprocess.Popen(['tail', '-f', log_file], 
                                         stdout=subprocess.PIPE, 
                                         stderr=subprocess.PIPE, 
                                         universal_newlines=True)
                
                for line in process.stdout:
                    if line.strip():
                        line_buffer.append(line.strip())
                        
                        # Process buffer when we hit a timestamp line (start of new event)
                        if re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}', line):
                            # Process previous buffered lines if any
                            if len(line_buffer) > 1:
                                event = parse_multiline_log(line_buffer[:-1])  # Exclude current line
                                if event:
                                    emit_event(event)
                            
                            # Keep only current line in buffer
                            line_buffer = [line.strip()]
                        
                        # Also try to parse current line as single event
                        event = parse_log_line(line.strip(), line_buffer)
                        if event:
                            emit_event(event)
                            
            else:
                time.sleep(1)
        except Exception as e:
            print(f"Error monitoring logs: {e}")
            time.sleep(5)

def parse_multiline_log(lines):
    """Parse multiple log lines together for multiline events"""
    if not lines:
        return None
        
    # Find the main event line (with timestamp)
    main_line = None
    context_lines = []
    
    for line in lines:
        if re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}', line):
            main_line = line
        else:
            context_lines.append(line)
    
    if not main_line:
        return None
        
    return parse_log_line(main_line, lines)

def emit_event(event):
    """Helper function to emit events and update statistics"""
    try:
        # Track call sessions for latency calculations
        connection_id = event.get('connection_id', 'unknown')
        if connection_id not in call_sessions:
            call_sessions[connection_id] = CallSession(connection_id)
        
        # Add event to session and calculate latencies
        call_sessions[connection_id].add_event(event)
        
        # Update stats
        event_counts[event['type']] += 1
        if event['type'] == 'connected':
            live_stats['total_calls'] += 1
            live_stats['last_call_time'] = event['timestamp_str']
        elif event['type'] == 'media':
            live_stats['total_media_packets'] += 1
        
        live_stats['total_events'] += 1
        
        # Add to recent events
        recent_events.append(event)
        
        # Emit to all connected clients (serialize datetime objects)
        event_to_emit = {k: v.isoformat() if isinstance(v, datetime) else v for k, v in event.items()}
        socketio.emit('new_event', event_to_emit)
        
        # Convert datetime objects for stats
        stats_to_emit = {k: v.isoformat() if isinstance(v, datetime) else v for k, v in live_stats.items()}
        socketio.emit('stats_update', stats_to_emit)
        socketio.emit('latency_update', {
            'avg_latency': live_stats['avg_latency'],
            'first_media_latency': live_stats['first_media_latency'],
            'end_to_end_latency': live_stats['end_to_end_latency']
        })
    except Exception as e:
        print(f"Error emitting event: {e}")

def parse_log_line(line, context_lines=None):
    """Enhanced log parsing with better timestamp handling and multiline context"""
    if context_lines is None:
        context_lines = [line]
    
    try:
        # Extract timestamp with timezone awareness
        timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})', line)
        if not timestamp_match:
            return None
            
        timestamp_str = timestamp_match.group(1)
        # Convert to proper datetime with timezone
        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f').replace(tzinfo=timezone.utc)
        
        # Extract connection ID
        connection_match = re.search(r'(conn_\w+)', line)
        connection_id = connection_match.group(1) if connection_match else 'unknown'
        
        # Extract stream SID and call SID from context
        stream_sid = extract_from_context(context_lines, r'Stream SID: ([^\s,\n]+)')
        call_sid = extract_from_context(context_lines, r'Call SID: ([^\s,\n]+)')
        
        # Parse different event types with enhanced multiline support
        if '🎉 CONNECTED EVENT' in line:
            return {
                'timestamp': timestamp,
                'timestamp_str': timestamp_str,
                'type': 'connected',
                'event': 'CONNECTED',
                'description': 'New call connected',
                'icon': '🎉',
                'connection_id': connection_id,
                'stream_sid': stream_sid,
                'call_sid': call_sid,
                'severity': 'success'
            }
        
        elif '🚀 START EVENT' in line:
            return {
                'timestamp': timestamp,
                'timestamp_str': timestamp_str,
                'type': 'start',
                'event': 'START',
                'description': 'Stream started',
                'icon': '🚀',
                'connection_id': connection_id,
                'stream_sid': stream_sid,
                'call_sid': call_sid,
                'severity': 'info'
            }
        
        elif '🎵 MEDIA EVENT' in line:
            chunk = extract_from_context(context_lines, r'Chunk: (\d+)')
            size = extract_from_context(context_lines, r'size: (\d+) bytes')
            sequence = extract_from_context(context_lines, r'Sequence: (\d+)')
            
            return {
                'timestamp': timestamp,
                'timestamp_str': timestamp_str,
                'type': 'media',
                'event': 'MEDIA',
                'description': f"Audio chunk {chunk or 'N/A'} ({size or 'N/A'} bytes)",
                'icon': '🎵',
                'connection_id': connection_id,
                'stream_sid': stream_sid,
                'call_sid': call_sid,
                'chunk': chunk,
                'size': size,
                'sequence': sequence,
                'severity': 'primary'
            }
        
        elif '🔢 DTMF EVENT' in line:
            # Extract digit and duration from context lines (multiline parsing)
            digit = extract_from_context(context_lines, r'🎹 Digit: ([^\s\n]+)')
            duration = extract_from_context(context_lines, r'⏱️\s+Duration: ([^\s\n]+)ms')
            
            return {
                'timestamp': timestamp,
                'timestamp_str': timestamp_str,
                'type': 'dtmf',
                'event': 'DTMF',
                'description': f"Key pressed: {digit or 'Unknown'}",
                'icon': '🔢',
                'connection_id': connection_id,
                'stream_sid': stream_sid,
                'call_sid': call_sid,
                'digit': digit,
                'duration': duration,
                'severity': 'info'
            }
        
        elif '🔢 DTMF RESPONSE' in line:
            digit = extract_from_context([line], r"Acknowledged digit '([^']+)'")
            return {
                'timestamp': timestamp,
                'timestamp_str': timestamp_str,
                'type': 'dtmf_response',
                'event': 'DTMF_ACK',
                'description': f"DTMF acknowledged: {digit or 'Unknown'}",
                'icon': '✅',
                'connection_id': connection_id,
                'stream_sid': stream_sid,
                'call_sid': call_sid,
                'digit': digit,
                'severity': 'success'
            }
        
        elif '📍 MARK EVENT' in line:
            mark_name = extract_from_context(context_lines, r'🏷️\s+Mark Name: ([^\s\n]+)')
            return {
                'timestamp': timestamp,
                'timestamp_str': timestamp_str,
                'type': 'mark',
                'event': 'MARK',
                'description': f"Mark: {mark_name or 'Unknown'}",
                'icon': '📍',
                'connection_id': connection_id,
                'stream_sid': stream_sid,
                'call_sid': call_sid,
                'mark_name': mark_name,
                'severity': 'warning'
            }
        
        elif '📍 MARK RESPONSE' in line:
            mark_name = extract_from_context([line], r"Acknowledged mark '([^']+)'")
            return {
                'timestamp': timestamp,
                'timestamp_str': timestamp_str,
                'type': 'mark_response',
                'event': 'MARK_ACK',
                'description': f"Mark acknowledged: {mark_name or 'Unknown'}",
                'icon': '✅',
                'connection_id': connection_id,
                'stream_sid': stream_sid,
                'call_sid': call_sid,
                'mark_name': mark_name,
                'severity': 'success'
            }
        
        elif '🧹 CLEAR EVENT' in line:
            return {
                'timestamp': timestamp,
                'timestamp_str': timestamp_str,
                'type': 'clear',
                'event': 'CLEAR',
                'description': 'Clear command received',
                'icon': '🧹',
                'connection_id': connection_id,
                'stream_sid': stream_sid,
                'call_sid': call_sid,
                'severity': 'warning'
            }
        
        elif '🧹 CLEAR RESPONSE' in line:
            return {
                'timestamp': timestamp,
                'timestamp_str': timestamp_str,
                'type': 'clear_response',
                'event': 'CLEAR_ACK',
                'description': 'Clear command acknowledged',
                'icon': '✅',
                'connection_id': connection_id,
                'stream_sid': stream_sid,
                'call_sid': call_sid,
                'severity': 'success'
            }
        
        elif '🛑 STOP EVENT' in line:
            reason = extract_from_context(context_lines, r'🔚 Reason: ([^\s\n]+)')
            return {
                'timestamp': timestamp,
                'timestamp_str': timestamp_str,
                'type': 'stop',
                'event': 'STOP',
                'description': f"Call ended: {reason or 'Unknown'}",
                'icon': '🛑',
                'connection_id': connection_id,
                'stream_sid': stream_sid,
                'call_sid': call_sid,
                'reason': reason,
                'severity': 'danger'
            }
        elif 'Call SID:' in line and 'connection open' not in line:
            call_sid_match = re.search(r'Call SID: (.+?)(?:\n|$)', line)
            if call_sid_match:
                return {
                    'timestamp': timestamp,
                    'timestamp_str': timestamp_str,
                    'type': 'call_info',
                    'event': 'CALL_INFO',
                    'description': f"Call SID: {call_sid_match.group(1)}",
                    'icon': '📞',
                    'connection_id': connection_id,
                    'stream_sid': stream_sid,
                    'call_sid': call_sid_match.group(1),
                    'severity': 'info'
                }
                
    except Exception as e:
        print(f"Error parsing log line: {e}")
    
    return None

def extract_from_context(lines, pattern):
    """Extract a value from context lines using regex pattern"""
    for line in lines:
        match = re.search(pattern, line)
        if match:
            return match.group(1).strip()
    return None

def load_historical_data():
    """Load historical data from existing logs"""
    log_file = 'logs/voice_bot_echo.log'
    calls_file = 'logs/calls.log'
    
    try:
        # Load call history
        if os.path.exists(calls_file):
            with open(calls_file, 'r') as f:
                for line in f:
                    try:
                        call_data = json.loads(line.strip())
                        call_history.append(call_data)
                    except:
                        pass
        
        # Count existing events and build sessions
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                for line in f:
                    event = parse_log_line(line.strip())
                    if event:
                        event_counts[event['type']] += 1
                        
                        # Build historical sessions for latency data
                        connection_id = event.get('connection_id', 'unknown')
                        if connection_id not in call_sessions:
                            call_sessions[connection_id] = CallSession(connection_id)
                        call_sessions[connection_id].add_event(event)
            
            live_stats['total_calls'] = event_counts.get('connected', 0)
            live_stats['total_media_packets'] = event_counts.get('media', 0)
            live_stats['total_events'] = sum(event_counts.values())
            
    except Exception as e:
        print(f"Error loading historical data: {e}")

@app.route('/')
def dashboard():
    """Main enhanced dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/stats')
def get_stats():
    """Get current statistics with latency data"""
    return jsonify({
        'live_stats': live_stats,
        'event_counts': dict(event_counts),
        'recent_events': list(recent_events)[-50:],  # Last 50 events
        'call_history': call_history[-20:],  # Last 20 calls
        'call_sessions': {k: {
            'connection_id': v.connection_id,
            'start_time': v.start_time.isoformat() if v.start_time else None,
            'end_time': v.end_time.isoformat() if v.end_time else None,
            'event_count': len(v.events),
            'avg_latency': round(sum(v.latencies) / len(v.latencies), 2) if v.latencies else 0
        } for k, v in call_sessions.items()}
    })

@app.route('/api/call/<connection_id>')
def get_call_details(connection_id):
    """Get detailed events for a specific call"""
    if connection_id in call_sessions:
        session = call_sessions[connection_id]
        return jsonify({
            'connection_id': connection_id,
            'events': [
                {
                    **event,
                    'timestamp': event['timestamp'].isoformat(),
                    'received_time': event['received_time'].isoformat()
                } for event in session.events
            ],
            'latencies': session.latencies,
            'avg_latency': round(sum(session.latencies) / len(session.latencies), 2) if session.latencies else 0
        })
    return jsonify({'error': 'Call not found'}), 404

@app.route('/api/clear_logs', methods=['POST'])
def clear_logs():
    """Clear all logs and reset data"""
    try:
        # Clear in-memory data
        recent_events.clear()
        call_history.clear()
        call_sessions.clear()
        event_counts.clear()
        latency_data.clear()
        
        # Reset stats
        live_stats.update({
            'total_calls': 0,
            'total_media_packets': 0,
            'total_events': 0,
            'last_call_time': None,
            'avg_latency': 0,
            'first_media_latency': 0,
            'end_to_end_latency': 0
        })
        
        # Clear log files
        log_files = ['logs/voice_bot_echo.log', 'logs/calls.log']
        for log_file in log_files:
            if os.path.exists(log_file):
                open(log_file, 'w').close()
        
        # Emit reset to all clients
        socketio.emit('logs_cleared')
        
        return jsonify({'success': True, 'message': 'All logs cleared successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@socketio.on('connect')
def handle_connect(auth):
    """Handle client connection"""
    # Convert datetime objects before emitting
    stats_to_emit = {k: v.isoformat() if isinstance(v, datetime) else v for k, v in live_stats.items()}
    emit('stats_update', stats_to_emit)
    emit('event_counts_update', dict(event_counts))
    # Send recent events with datetime conversion
    for event in list(recent_events)[-20:]:
        event_to_emit = {k: v.isoformat() if isinstance(v, datetime) else v for k, v in event.items()}
        emit('new_event', event_to_emit)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    pass

if __name__ == '__main__':
    # Create templates directory
    os.makedirs('templates', exist_ok=True)
    
    # Load historical data
    load_historical_data()
    
    # Start log monitoring in background
    log_thread = threading.Thread(target=tail_logs, daemon=True)
    log_thread.start()
    
    print("🚀 AgentStream Dashboard starting...")
    print("📊 Dashboard URL: http://localhost:5001")
    print("📈 Real-time monitoring with latency tracking active!")
    print("🎯 Features: Timestamps, Latency Analysis, Call Selection, Log Management")
    
    # Start the web server
    socketio.run(app, host='0.0.0.0', port=5001, debug=False) 