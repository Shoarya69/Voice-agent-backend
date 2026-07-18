# Voice pipeline providers

Switch the active provider with a single environment variable — no application code changes.

```env
VOICE_PIPELINE=compound          # default (production)
# VOICE_PIPELINE=openai_realtime
```

## Architecture

```
Exotel WSS (ai_server.py)
        │
        ▼
create_voice_pipeline_session()  ← reads VOICE_PIPELINE
        │
        ├── compound ──► CompoundPipelineSession ──► VoiceSession
        │                  (ElevenLabs Realtime STT WS → OpenAI stream → ElevenLabs TTS WS)
        │
        └── openai_realtime ──► OpenAIRealtimePipelineSession
                               (OpenAI Realtime WebSocket, speech-to-speech)
```

Shared across all providers (`VoicePipelineSession` in `base.py`):

- MoontechPro agent lookup / prefetch
- CRM call logs
- Conversation `history` for transcripts
- Exotel `stream_sid` / caller metadata

## Adding a new provider (Deepgram, Gemini Live, Azure, …)

1. Create `app/voice_pipeline/<name>/session.py`
2. Subclass `VoicePipelineSession` and implement:
   - `add_audio_chunk()`
   - `speak_welcome()`
   - `handle_clear()`
   - `close()`
   - Optional: `on_agent_ready()` for provider-specific setup after agent config loads
3. Register in `app/voice_pipeline/factory.py`:

   ```python
   _REGISTERED_PIPELINES["deepgram"] = "app.voice_pipeline.deepgram.session.DeepgramPipelineSession"
   ```

4. Add provider-specific env vars to `app/config.py` and `.env.example`
5. Set `VOICE_PIPELINE=deepgram` and redeploy

No changes required in `ai_server.py`, `agent_setup.py`, or `lovable_client.py`.

## Rollback

Set `VOICE_PIPELINE=compound` (or remove the variable) and restart:

```bash
docker compose up -d --build
```

The compound pipeline is unchanged — `CompoundPipelineSession` extends the existing `VoiceSession`.
