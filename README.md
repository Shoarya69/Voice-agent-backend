# 🤖 AI Voice Agent for Exotel (OpenAI + ElevenLabs)

This repo now ships a real AI phone agent (`ai_server.py`) in addition to the
original raw echo server (`simple_server.py`, kept for connectivity testing):

- **Brain**: OpenAI (`gpt-4o-mini` by default) decides what to say.
- **Ears**: ElevenLabs Speech-to-Text transcribes the caller's audio.
- **Voice**: ElevenLabs Text-to-Speech speaks the reply back, generated
  directly at 8kHz/16-bit PCM to match Exotel's `raw/slin` stream format
  (no audio conversion needed).

## 🚀 Quick Start (AI Agent)

```bash
# 1. Install dependencies
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Configure your API keys
cp .env.example .env
# then edit .env and set OPENAI_API_KEY and ELEVENLABS_API_KEY

# 3. Run the AI agent
python3 ai_server.py
```

The server listens on `ws://0.0.0.0:8002` by default (configurable via `.env`) - point
your Exotel Voicebot Applet's WebSocket URL (via ngrok or your public
domain) at it, same as described below for the echo server.

Conversation flow: caller speaks → WebRTC VAD + dynamic noise-floor endpointing
accepts the turn → audio sent to ElevenLabs STT → transcript + history sent to
OpenAI, which is **streamed token-by-token**. As soon as the model finishes a
sentence, that sentence is immediately sent to ElevenLabs streaming TTS and
played to the caller *while the model keeps generating the rest of the
reply* - this is what keeps the perceived response latency low instead of
waiting for the entire reply before speaking a word.

Barge-in defaults to off (`ENABLE_BARGE_IN=false`) because many telephony
lines echo the bot's own voice back into the inbound stream. Instead, while
the bot is speaking (and for a short `ECHO_GUARD_MS` window after it stops),
inbound audio is ignored entirely and noise-floor calibration is paused, so
the bot's own echoed voice can't inflate the noise floor and drown out the
caller's next sentence.

See `.env.example` for every configurable option (model, voice ID, system
prompt, welcome message, silence threshold, etc).

## 🐳 Docker Deployment

The included `Dockerfile` runs both the AI agent (port 8002) and the
monitoring dashboard (port 5001):

```bash
# Configure your keys first
cp .env.example .env   # then edit .env

# Build and run with docker compose (recommended)
docker compose up -d --build

# ...or with plain docker
docker build -t ai-voice-agent .
docker run -d --name ai-voice-agent --env-file .env \
  -p 8002:8002 -p 5001:5001 -v $(pwd)/logs:/app/logs ai-voice-agent
```

On your server, put this behind a reverse proxy (nginx/Caddy) that
terminates TLS so Exotel can connect over `wss://`.

## Voice Detection Tuning

The agent uses WebRTC VAD plus a per-call dynamic RMS threshold. Useful VPS
settings:

```env
VAD_MODE=2
NOISE_CALIBRATION_MS=800
NOISE_FLOOR_MULTIPLIER=5.0
DYNAMIC_RMS_MIN=400
NOISE_FLOOR_MAX_RMS=600
VAD_START_MS=150
VAD_END_SILENCE_MS=800
MIN_VAD_SPEECH_MS=250
VAD_CANDIDATE_TOLERANCE_MS=120
ECHO_GUARD_MS=250
STT_MIN_AVG_WORD_CONFIDENCE=0.35
STT_MIN_LANGUAGE_PROBABILITY=0.45
ENABLE_BARGE_IN=false
TTS_VOLUME=1.0
OPENAI_MAX_TOKENS=120
```

If real speech is missed, lower `DYNAMIC_RMS_MIN` slightly or use
`VAD_MODE=1`. If line noise still starts turns, raise `VAD_MODE` to `3` or
increase `NOISE_FLOOR_MULTIPLIER`.

Notes on the settings that were previously causing missed/garbled speech and
high latency (fixed in this pass):

- **`NOISE_FLOOR_MAX_RMS`** caps how high the adaptive noise floor can drift.
  Without it, the bot's own greeting echoing into the mic during the first
  ~800ms could spike the noise floor to 800+ and the dynamic threshold to
  4000+, making the agent deaf to normal speech for the rest of the call.
- **`VAD_CANDIDATE_TOLERANCE_MS`** tolerates a brief (≤120ms) dip below the
  speech threshold while a candidate utterance is building, instead of
  discarding the whole candidate buffer. This was clipping the first
  syllable of real words and feeding STT a truncated, garbled clip.
- **`MIN_VAD_SPEECH_MS`** was lowered from 350ms to 250ms because short but
  valid utterances (e.g. quick replies) were being rejected outright in
  production logs (`vad_speech_ms=340 < 350`).
- **`OPENAI_MAX_TOKENS`** was lowered to keep replies short, and the LLM
  reply is now streamed sentence-by-sentence into TTS (see above), which
  together cut the time-to-first-audio from several seconds down to
  roughly STT time + time-to-first-LLM-sentence + TTS first-byte time.
- **`TTS_VOLUME`** defaults to `1.0` now; only lower it if you specifically
  hear clipping on your phone line, since scaling down every sample can add
  quantization noise/dullness.

---

# 🤖 Enhanced Voice Bot Echo Server for Exotel

A **comprehensive, intelligent WebSocket echo server** with **conversational AI behavior** and **real-time monitoring dashboard** specifically designed for testing Exotel's voice streaming functionality. Features advanced audio buffering, silence detection, and interactive analytics.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![WebSockets](https://img.shields.io/badge/websockets-v12.0+-green.svg)
![Flask](https://img.shields.io/badge/flask-v2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-production--ready-brightgreen.svg)

## ✨ What This Does

🧠 **Conversational AI Echo**: Listens first, detects silence, then responds naturally - no more immediate echo interruptions

🎧 **Audio Buffering**: Intelligently buffers incoming audio and responds after silence detection

🛑 **Smart Interruption**: Handles CLEAR events to stop speaking and reset conversation state

📊 **Real-time Dashboard**: Live monitoring with interactive latency metrics and event visualization

🔍 **Advanced Protocol Testing**: Handles all Exotel WebSocket events with enhanced logging and response acknowledgments

⚡ **Production Ready**: Robust error handling, session management, and multiline log parsing

🚀 **Easy Setup**: One-command installation with automated dependency management

## 🆕 **Enhanced Features**

### **🎯 Conversational Echo Bot**
- **Listen → Silence → Respond**: Natural conversation flow instead of immediate echo
- **Audio Buffering**: Collects audio chunks during listening phase
- **Silence Detection**: 2-second silence threshold before responding
- **Clear Interruption**: Instantly stops and resets on CLEAR events
- **Session Management**: Per-call state tracking and cleanup

### **📊 AgentStream Dashboard** *(Sample Reference)*
- **Real-time Event Feed**: Live stream of all voice bot activities
- **Latency Analytics**: Inter-event, first media, and end-to-end latency tracking
- **Interactive Tooltips**: Hover explanations for all metrics
- **Call Session Tracking**: Detailed per-call event analysis
- **Multiline Log Parsing**: Accurate DTMF digit extraction and event correlation

> **Note**: The dashboard is provided as a sample reference implementation. While it provides valuable insights into system behavior, data accuracy may vary depending on log timing and parsing complexity. Use it for monitoring and debugging purposes.

## 🏃‍♂️ Quick Start (2 Minutes)

### **Prerequisites**
- Python 3.8+ 
- Ports 5000 (server) and 5001 (dashboard) available
- Internet connection

### **Installation**

```bash
# Clone the repository
git clone https://github.com/exotel/Agent-Stream-echobot.git
cd Agent-Stream-echobot

# One-command setup
chmod +x setup.sh && ./setup.sh
```

### **Start the Services**

```bash
# Start both server and dashboard
./start.sh

# Or start individually:
# Enhanced Echo Server (port 5000)
source venv/bin/activate && python3 simple_server.py &

# AgentStream Dashboard (port 5001)  
source venv/bin/activate && python3 dashboard.py &
```

**Access Points:**
- 🤖 **Echo Server**: `ws://localhost:5000`
- 📊 **Dashboard**: `http://localhost:5001`

## 🌐 Public Access with ngrok

To test with Exotel, you need a public WSS URL:

```bash
# Install ngrok (if not already installed)
brew install ngrok  # macOS
# or download from https://ngrok.com/

# Configure your ngrok authtoken
ngrok config add-authtoken YOUR_NGROK_TOKEN

# Make your server public
ngrok http 5000
```

Use the `wss://` URL from ngrok in your Exotel configuration.

## 🧪 Testing

### **Basic Connection Test**
```bash
# Test local server
python3 test_connection.py

# Test public ngrok URL
python3 test_connection.py wss://your-ngrok-url.ngrok.io
```

### **Enhanced Features Test**
```bash
# Test conversational behavior and all features
python3 test_enhanced_features.py
```

## 📋 Exotel Configuration

### **For Bidirectional Streaming (Voicebot Applet)**

1. **URL**: `wss://your-ngrok-url.ngrok.io`
2. **Custom Parameters**: Optional (will be logged)
3. **Record**: Enable if you want call recordings
4. **Next Applet**: Configure your next flow step

### **For Unidirectional Streaming (Stream Applet)**

1. **Action**: Start
2. **URL**: `wss://your-ngrok-url.ngrok.io`
3. **Next Applet**: Configure your next flow step

## 🎯 Enhanced Call Flow

### **Traditional Echo Flow** ❌
```
User speaks → Immediate echo → Interruption → Poor UX
```

### **Enhanced Conversational Flow** ✅
```
1. 🎧 LISTENING: User speaks → Audio buffering
2. 🤔 SILENCE: 2s silence detected → Prepare response  
3. 🗣️ SPEAKING: Send buffered audio naturally
4. 🛑 CLEAR: Handle interruptions gracefully
5. 👂 RESET: Ready for next turn
```

## 📊 Monitoring & Analytics

### **Real-time Dashboard Features**
- **📈 Live Metrics**: Calls, media packets, events with tooltips
- **⏱️ Latency Tracking**: 
  - **Avg Latency**: Time between consecutive events
  - **First Media**: Connection to first audio packet
  - **End-to-End**: Complete call duration
- **🎯 Event Feed**: Real-time activity stream with filtering
- **📱 Call Sessions**: Interactive call selection and analysis
- **🧹 Log Management**: Clear logs and export functionality

### **Enhanced Logging**
The server creates comprehensive logs in the `logs/` directory:

- **`voice_bot_echo.log`**: Enhanced server activity with conversation flow
- **`calls.log`**: Individual call details in JSON format

### **Log Monitoring Commands**
```bash
# Watch enhanced server logs
tail -f logs/voice_bot_echo.log

# Monitor conversational behavior
grep -E "(LISTENING|BUFFERING|SILENCE|SPEAKING|CLEAR)" logs/voice_bot_echo.log

# Watch specific events
grep "DTMF EVENT" logs/voice_bot_echo.log
```

## 📁 Enhanced File Structure

```
Agent-Stream-echobot/
├── ai_server.py               # AI Voice Agent entrypoint (OpenAI + ElevenLabs)
├── app/                       # AI agent modules
│   ├── config.py               # Env-based configuration
│   ├── audio_utils.py          # PCM/WAV helpers for Exotel's raw/slin format
│   ├── elevenlabs_service.py   # ElevenLabs STT + TTS client
│   ├── openai_service.py       # OpenAI "brain" client
│   └── voice_session.py        # Per-call turn-taking state machine
├── simple_server.py           # Raw echo server (connectivity testing only)
├── dashboard.py                # Real-time monitoring dashboard
├── test_connection.py          # Basic connection testing
├── test_enhanced_features.py   # Comprehensive feature testing
├── requirements.txt             # Python dependencies
├── .env.example                 # Copy to .env and fill in API keys
├── Dockerfile                   # Container image (AI agent + dashboard)
├── docker-compose.yml           # One-command deployment
├── setup.sh                     # Automated local setup script
├── start.sh                     # Multi-service startup script
├── templates/
│   └── dashboard.html          # Interactive dashboard UI
├── logs/                       # Log files (created at runtime)
│   ├── ai_voice_agent.log      # AI agent logs
│   ├── voice_bot_echo.log      # Echo server logs
│   └── calls.log               # Call session data
└── venv/                       # Virtual environment
```

## 🔧 Advanced Configuration

### **Echo Mode Configuration**

Choose between immediate echo (traditional) or conversational AI mode:

```python
# In simple_server.py - Line ~153
IMMEDIATE_ECHO_MODE = True   # Traditional immediate echo for testing
IMMEDIATE_ECHO_MODE = False  # Conversational AI with silence detection
```

### **Conversation Parameters** (when IMMEDIATE_ECHO_MODE = False)

```python
class VoiceSession:
    def __init__(self, connection_id, websocket):
        # Customize these parameters
        self.silence_threshold = 2.0    # Seconds before responding
        self.response_delay = 0.1       # Delay between audio chunks
```

### **Dashboard Customization**

Edit `dashboard.py` for custom analytics:

```python
# Modify latency calculations
live_stats = {
    'custom_metric': your_calculation,
    'threshold_alerts': custom_thresholds
}
```

### **Custom Port Configuration**

```bash
# Set custom ports via environment variables
export ECHO_PORT=8080
export DASHBOARD_PORT=8081

# Or edit the files directly
```

## 🛠️ Troubleshooting

### **Enhanced Server Issues**

```bash
# Check server status
curl -f http://localhost:5000 || echo "Server not responding"

# Monitor conversation flow
grep -E "(LISTENING|SPEAKING)" logs/voice_bot_echo.log | tail -10

# Check session cleanup
grep "Connection ended" logs/voice_bot_echo.log | tail -5
```

### **Dashboard Issues**

```bash
# Verify dashboard
curl -f http://localhost:5001 || echo "Dashboard not accessible"

# Check log parsing
grep "Error parsing" logs/* 

# Monitor WebSocket connections
grep "connected\|disconnected" logs/voice_bot_echo.log
```

### **Latency Issues**

1. **High Inter-event Latency**: Check network connection and server load
2. **Poor First Media**: Verify Exotel connection establishment
3. **Long End-to-End**: Review call flow and timeout configurations

## 🎨 Customization Examples

### **Add Custom Audio Processing**

```python
async def start_response(self):
    """Enhanced response with custom processing"""
    for i, media_data in enumerate(self.audio_buffer):
        # Your custom audio processing
        processed_audio = your_audio_processor(media_data)
        
        # Send enhanced response
        echo_response = {
            'event': 'media',
            'stream_sid': self.stream_sid,
            'media': processed_audio
        }
        await self.websocket.send(json.dumps(echo_response))
```

### **Custom Dashboard Metrics**

```python
# Add custom analytics
def calculate_custom_metrics(events):
    return {
        'speech_to_silence_ratio': calculate_ratio(events),
        'interruption_frequency': count_clears(events),
        'conversation_turns': count_turns(events)
    }
```

## 🚀 Production Deployment

### **Docker Deployment**

See the "🐳 Docker Deployment" section near the top of this README - the
included `Dockerfile`/`docker-compose.yml` run the AI agent (`ai_server.py`)
and dashboard together and are ready to deploy as-is.

### **Cloud Deployment**

```bash
# On your cloud server
git clone https://github.com/exotel/Agent-Stream-echobot.git
cd Agent-Stream-echobot
./setup.sh

# Use reverse proxy for HTTPS
nginx -t && systemctl reload nginx
```

### **Environment Variables**

```bash
# Production configuration
export LOG_LEVEL=INFO
export ECHO_PORT=5000
export DASHBOARD_PORT=5001
export SILENCE_THRESHOLD=1.5
export ENABLE_DASHBOARD=true
```

## 📈 Performance Metrics

### **Latency Benchmarks**
- **Inter-event Latency**: < 50ms (excellent), < 100ms (good)
- **First Media Latency**: < 200ms (excellent), < 500ms (good)  
- **End-to-End Latency**: Depends on call duration
- **Silence Detection**: 2s threshold (configurable)

### **Scalability**
- **Concurrent Calls**: 100+ (single instance)
- **Memory Usage**: ~50MB base + ~1MB per active call
- **CPU Usage**: < 5% (idle), < 20% (active calls)

## 🧪 Testing Scenarios

### **Conversation Flow Testing**
1. **Normal Flow**: Speak → Wait → Hear response
2. **Interruption**: Speak → Send CLEAR → Verify stop
3. **Multiple Turns**: Alternate speaking/listening
4. **DTMF Integration**: Test keypress during conversation

### **Load Testing**
```bash
# Simulate multiple concurrent calls
for i in {1..10}; do
    python3 test_enhanced_features.py &
done
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### **Development Setup**
```bash
# Development mode with hot reload
export FLASK_ENV=development
python3 dashboard.py

# Test with verbose logging
export LOG_LEVEL=DEBUG
python3 simple_server.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support & Documentation

- **🐛 Issues**: [GitHub Issues](https://github.com/exotel/Agent-Stream-echobot/issues)
- **📚 Exotel Docs**: [Voice Streaming Guide](https://developer.exotel.com/api/voice-streaming)
- **🔧 WebSockets**: [Python websockets library](https://websockets.readthedocs.io/)
- **📊 Flask-SocketIO**: [Real-time documentation](https://flask-socketio.readthedocs.io/)

## 🎯 Use Cases

- **🧪 Testing**: Validate Exotel voice streaming with realistic conversation flow
- **🔍 Debugging**: Analyze audio latency and protocol behavior  
- **🎓 Learning**: Study conversational AI and WebSocket telephony
- **🚀 Foundation**: Starting point for building production voice bots
- **📊 Monitoring**: Real-time analytics for voice streaming performance
- **🤖 AI Development**: Test natural conversation patterns and interruption handling

## 🔮 Future Enhancements

- **🧠 AI Integration**: Real conversational AI responses
- **📊 Advanced Analytics**: Call quality metrics and insights
- **🌍 Multi-language**: Support for different audio formats
- **☁️ Cloud Integration**: Direct cloud deployment templates
- **📱 Mobile Dashboard**: Responsive mobile monitoring interface

---

**🚀 Ready to experience natural voice conversations with Exotel? This enhanced echo server brings AI-like behavior to voice testing!**

Made with ❤️ for the Exotel developer community | **Enhanced with Conversational AI & Real-time Analytics** 