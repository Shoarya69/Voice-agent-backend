"""
Centralized configuration for the AI Voice Agent.

All values are read from environment variables (loaded from a `.env` file
via python-dotenv if present). See `.env.example` for the full list of
supported variables and sane defaults.
"""

import contextvars
import os
from dotenv import load_dotenv

# Load variables from a .env file in the current working directory (if any).
# This must run before anything below reads os.environ.
load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
HOST = os.getenv("HOST", "0.0.0.0")
PORT = _get_int("PORT", 5000)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
# Separate plain-HTTP port for Exotel's telephony webhooks (Status Callback /
# conversation applet), relayed to Lovable. Independent from PORT (the
# WebSocket Voicebot Applet port) so it can be exposed/proxied separately.
WEBHOOK_PORT = _get_int("WEBHOOK_PORT", 8081)
# When true, Exotel status callbacks (ringing/in-progress) prewarm agent data
# for the dialed Exophone so the WSS handler finds a warm cache on `start`.
PREWARM_ON_EXOTEL_STATUS = _get_bool("PREWARM_ON_EXOTEL_STATUS", True)

# ---------------------------------------------------------------------------
# Concurrency / capacity (tuned for ~6 vCPU / 12 GB — target 100–150 sessions)
# ---------------------------------------------------------------------------
MAX_CONCURRENT_SESSIONS = _get_int("MAX_CONCURRENT_SESSIONS", 150)
# ElevenLabs STT/TTS in-flight caps (per process).
MAX_CONCURRENT_STT = _get_int("MAX_CONCURRENT_STT", 80)
MAX_CONCURRENT_TTS = _get_int("MAX_CONCURRENT_TTS", 80)
# Thread pool for webrtcvad + RMS (keep <= CPU cores; 4–6 on a 6-core VPS).
VAD_WORKER_THREADS = _get_int("VAD_WORKER_THREADS", 4)
# Drop per-frame VAD info logs in production (set true only for debugging).
ENABLE_VAD_FRAME_LOGS = _get_bool("ENABLE_VAD_FRAME_LOGS", False)
VAD_FRAME_LOG_EVERY = _get_int("VAD_FRAME_LOG_EVERY", 500)
# Cap a single utterance buffer (~30s @ 8kHz mono 16-bit ≈ 480 KB).
MAX_TURN_AUDIO_MS = _get_int("MAX_TURN_AUDIO_MS", 30_000)
# Outbound HTTP pool sizing (ElevenLabs + MoontechPro share per-client pools).
HTTP_MAX_CONNECTIONS = _get_int("HTTP_MAX_CONNECTIONS", 200)
HTTP_MAX_KEEPALIVE = _get_int("HTTP_MAX_KEEPALIVE", 60)
HTTP_KEEPALIVE_EXPIRY = _get_float("HTTP_KEEPALIVE_EXPIRY", 30.0)

# ---------------------------------------------------------------------------
# Voice pipeline provider (switch via .env — no code changes)
# ---------------------------------------------------------------------------
# compound         — ElevenLabs Realtime STT → OpenAI stream → ElevenLabs WS TTS (default)
# openai_realtime  — OpenAI Realtime API (speech-to-speech)
VOICE_PIPELINE = os.getenv("VOICE_PIPELINE", "compound").strip().lower()

# OpenAI Realtime (only when VOICE_PIPELINE=openai_realtime)
OPENAI_REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-2.1")
OPENAI_REALTIME_BASE_URL = os.getenv(
    "OPENAI_REALTIME_BASE_URL", "wss://api.openai.com/v1"
)
OPENAI_REALTIME_VOICE = os.getenv("OPENAI_REALTIME_VOICE", "alloy")
OPENAI_REALTIME_TRANSCRIPTION_MODEL = os.getenv(
    "OPENAI_REALTIME_TRANSCRIPTION_MODEL", "gpt-4o-mini-transcribe"
)
OPENAI_REALTIME_VAD_THRESHOLD = _get_float("OPENAI_REALTIME_VAD_THRESHOLD", 0.5)
OPENAI_REALTIME_PREFIX_PADDING_MS = _get_int("OPENAI_REALTIME_PREFIX_PADDING_MS", 300)
OPENAI_REALTIME_SILENCE_DURATION_MS = _get_int(
    "OPENAI_REALTIME_SILENCE_DURATION_MS", 500
)

# ---------------------------------------------------------------------------
# OpenAI (the "brain" of the agent)
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
# NOTE: OPENAI_TEMPERATURE below is also overridable per-call by Lovable agent
# config. OPENAI_MAX_TOKENS is NOT overridable (platform cost cap). Reply word
# limits (REPLY_WORD_*) are enforced in openai_service + voice_session code.
_OPENAI_TEMPERATURE_DEFAULT = _get_float("OPENAI_TEMPERATURE", 0.6)
# Kept short on purpose: shorter replies are faster to generate AND faster to
# speak, which matters a lot for perceived latency and TTS cost on a live call.
_OPENAI_MAX_TOKENS_DEFAULT = _get_int("OPENAI_MAX_TOKENS", 45)
# Platform-enforced cap: NOT overridable by per-agent MoontechPro/Lovable config.
OPENAI_MAX_TOKENS = _OPENAI_MAX_TOKENS_DEFAULT
REPLY_WORD_TARGET = _get_int("REPLY_WORD_TARGET", 10)
REPLY_WORD_MAX = _get_int("REPLY_WORD_MAX", 30)
_SYSTEM_PROMPT_DEFAULT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a friendly Hindi/Hinglish voice assistant on a live phone call. "
    "Reply in simple Hindi/Hinglish unless the caller clearly asks for another language. "
    "Use the shortest possible words. One short sentence only. No lists or filler.",
)
REPLY_LENGTH_INSTRUCTION = (
    f"MANDATORY platform rule (overrides ANY conflicting instruction above, "
    f"including longer or detailed replies): default to {REPLY_WORD_TARGET} words or fewer. "
    f"Hard maximum {REPLY_WORD_MAX} words — never exceed this under any circumstance. "
    "If more detail is needed, give only the single most important point."
)
CALL_END_MARKER = os.getenv("CALL_END_MARKER", "[[END_CALL]]")
CALL_END_INSTRUCTION = (
    "MANDATORY call-ending rule (overrides conflicting instructions above): "
    "if the caller clearly says bye/goodbye, says they do not "
    "need anything else, thanks you to end the conversation, or the task is fully "
    "complete and it is natural to hang up, say one short polite closing sentence "
    f"within the word limit and append {CALL_END_MARKER} at the very end. "
    "Never speak or explain this marker."
)
CALL_BEHAVIOR_INSTRUCTION = (
    "MANDATORY (overrides conflicting instructions above): "
    "Answer every caller question on this live call — do not defer, deflect, or postpone. "
    "Never say you will call back, callback, follow up later, or that someone else will call "
    "unless the caller explicitly asks to be called later. "
    "When asked about products, services, or what you offer, briefly name them from the "
    "business details in your instructions above (within the word limit). "
    "If the list is long, give the top two or three items and invite one short follow-up question."
)
# Appended after every agent/system prompt in openai_service — not overridable.
MANDATORY_PLATFORM_RULES_HEADER = (
    "--- MANDATORY PLATFORM RULES (these override any conflicting instructions above) ---"
)

# ---------------------------------------------------------------------------
# ElevenLabs (Speech-to-Text + Text-to-Speech)
# ---------------------------------------------------------------------------
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
_ELEVENLABS_VOICE_ID_DEFAULT = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
ELEVENLABS_TTS_MODEL = os.getenv("ELEVENLABS_TTS_MODEL", "eleven_turbo_v2_5")
ELEVENLABS_STT_MODEL = os.getenv("ELEVENLABS_STT_MODEL", "scribe_v1")
# optional ISO code, blank = auto-detect
_ELEVENLABS_STT_LANGUAGE_DEFAULT = os.getenv("ELEVENLABS_STT_LANGUAGE", "")
ELEVENLABS_BASE_URL = os.getenv("ELEVENLABS_BASE_URL", "https://api.elevenlabs.io")

# ---------------------------------------------------------------------------
# Low-latency streaming pipeline (compound production default)
# ---------------------------------------------------------------------------
STREAMING_PIPELINE = _get_bool("STREAMING_PIPELINE", True)
# STT: elevenlabs (Scribe v2 Realtime WS) | deepgram (optional SDK) | batch (rollback)
STT_PROVIDER = os.getenv("STT_PROVIDER", "elevenlabs").strip().lower()
# TTS: elevenlabs_ws (persistent stream-input WS) | http (per-clause REST rollback)
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "elevenlabs_ws").strip().lower()
STREAMING_STT_MODEL = os.getenv("STREAMING_STT_MODEL", "scribe_v2_realtime")
STREAMING_STT_VAD_SILENCE_SECS = _get_float("STREAMING_STT_VAD_SILENCE_SECS", 0.45)
STREAMING_STT_VAD_THRESHOLD = _get_float("STREAMING_STT_VAD_THRESHOLD", 0.4)
STREAMING_STT_MIN_SPEECH_MS = _get_int("STREAMING_STT_MIN_SPEECH_MS", 100)
STREAMING_STT_MIN_SILENCE_MS = _get_int("STREAMING_STT_MIN_SILENCE_MS", 100)
STREAMING_STT_ENDPOINTING_MS = _get_int("STREAMING_STT_ENDPOINTING_MS", 450)
STREAMING_TTS_WS_INACTIVITY_TIMEOUT = _get_int("STREAMING_TTS_WS_INACTIVITY_TIMEOUT", 180)
STREAMING_TTS_CHUNK_SCHEDULE = [
    int(x.strip())
    for x in os.getenv("STREAMING_TTS_CHUNK_SCHEDULE", "120,160,250,290").split(",")
    if x.strip().isdigit()
] or [120, 160, 250, 290]
STREAMING_CLAUSE_MIN_WORDS = _get_int("STREAMING_CLAUSE_MIN_WORDS", 3)
STREAMING_CLAUSE_MAX_CHARS = _get_int("STREAMING_CLAUSE_MAX_CHARS", 48)
TURN_MONITOR_INTERVAL_MS = _get_int("TURN_MONITOR_INTERVAL_MS", 25)
# Pre-generated greeting PCM cap (8kHz slin). Moontech sometimes returns 15–20s clips.
MAX_GREETING_AUDIO_MS = _get_int("MAX_GREETING_AUDIO_MS", 8000)
GREETING_MOONTECH_TIMEOUT_SEC = _get_float("GREETING_MOONTECH_TIMEOUT_SEC", 2.5)
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
DEEPGRAM_STT_MODEL = os.getenv("DEEPGRAM_STT_MODEL", "nova-3")

# ---------------------------------------------------------------------------
# Voice / conversation behaviour
# ---------------------------------------------------------------------------
# Exotel's voice streaming media format used across this project (raw/slin, 8kHz, 16-bit mono).
SAMPLE_RATE = _get_int("SAMPLE_RATE", 8000)
SAMPLE_WIDTH = 2  # bytes (16-bit PCM)
CHANNELS = 1

SILENCE_THRESHOLD_SECONDS = _get_float("SILENCE_THRESHOLD", 1.5)
MIN_AUDIO_MS_TO_PROCESS = _get_int("MIN_AUDIO_MS_TO_PROCESS", 500)
VAD_MODE = _get_int("VAD_MODE", 2)
NOISE_CALIBRATION_MS = _get_int("NOISE_CALIBRATION_MS", 800)
NOISE_FLOOR_MULTIPLIER = _get_float("NOISE_FLOOR_MULTIPLIER", 5.0)
DYNAMIC_RMS_MIN = _get_float("DYNAMIC_RMS_MIN", 400.0)
# Cap how high the adaptive noise floor is allowed to drift, so a burst of
# echoed bot audio during calibration can't permanently deafen the session.
NOISE_FLOOR_MAX_RMS = _get_float("NOISE_FLOOR_MAX_RMS", 600.0)
MIN_VAD_SPEECH_MS = _get_int("MIN_VAD_SPEECH_MS", 250)
VAD_START_MS = _get_int("VAD_START_MS", 150)
VAD_END_SILENCE_MS = _get_int("VAD_END_SILENCE_MS", 450)
# Tolerate short dips below the speech threshold while a speech candidate is
# building, instead of discarding the whole candidate (avoids clipping the
# first syllable of real words on brief energy dips).
VAD_CANDIDATE_TOLERANCE_MS = _get_int("VAD_CANDIDATE_TOLERANCE_MS", 120)
# Ignore all inbound audio (and skip noise-floor calibration) for this long
# after the bot stops talking, since telephony lines commonly echo the bot's
# own voice back into the inbound stream for a short tail.
ECHO_GUARD_MS = _get_int("ECHO_GUARD_MS", 250)
STT_MIN_AVG_WORD_CONFIDENCE = _get_float("STT_MIN_AVG_WORD_CONFIDENCE", 0.35)
STT_MIN_LANGUAGE_PROBABILITY = _get_float("STT_MIN_LANGUAGE_PROBABILITY", 0.45)
TTS_VOLUME = _get_float("TTS_VOLUME", 1.0)
_WELCOME_MESSAGE_DEFAULT = os.getenv(
    "WELCOME_MESSAGE", "Hi! I'm your AI assistant. How can I help you today?"
)
ENABLE_WELCOME_MESSAGE = _get_bool("ENABLE_WELCOME_MESSAGE", True)
MAX_CONVERSATION_TURNS = _get_int("MAX_CONVERSATION_TURNS", 20)
# Let the caller interrupt the bot mid-sentence by speaking. Disable this if your
# telephony line echoes the bot's own voice back (causing false interruptions).
ENABLE_BARGE_IN = _get_bool("ENABLE_BARGE_IN", False)
# Detailed per-stage latency logs (LATENCY_TURN / LATENCY_CALL); logging only.
ENABLE_LATENCY_TRACE = _get_bool("ENABLE_LATENCY_TRACE", True)

# How many bytes of raw PCM correspond to one Exotel media chunk (20ms @ 8kHz/16-bit/mono = 320 bytes)
CHUNK_DURATION_MS = 20
BYTES_PER_CHUNK = int(SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS * (CHUNK_DURATION_MS / 1000))

# ---------------------------------------------------------------------------
# Supabase (direct agent reads — replaces Moontech HTTP GET on live-call path)
# ---------------------------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL", "") or os.getenv("VITE_SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
# Anon/publishable key (client key) — works only if Moontech RLS allows public SELECT.
# Accepts common names from frontend .env files.
SUPABASE_ANON_KEY = (
    os.getenv("SUPABASE_ANON_KEY", "")
    or os.getenv("SUPABASE_PUBLISHABLE_KEY", "")
    or os.getenv("VITE_SUPABASE_PUBLISHABLE_KEY", "")
)
# Moontech base URL for cold greeting synth fallback only (write-side on Moontech).
MOONTECH_BASE_URL = os.getenv("MOONTECH_BASE_URL", "") or os.getenv("LOVABLE_APP_URL", "")

# ---------------------------------------------------------------------------
# Lovable control plane (per-agent config lookup + call logging)
# ---------------------------------------------------------------------------
LOVABLE_APP_URL = os.getenv("LOVABLE_APP_URL", "")
# Fallback auth only: sent as a Bearer token on call-log requests for backwards
# compatibility. The primary auth for agent lookup is the per-agent token
# passed in the WSS URL query string, handled by app/lovable_client.py.
LOVABLE_API_SECRET = os.getenv("LOVABLE_API_SECRET", "")


# ---------------------------------------------------------------------------
# Per-call agent overrides (Lovable)
# ---------------------------------------------------------------------------
# A handful of settings can be overridden per-call by the agent config fetched
# from Lovable (see app/lovable_client.py). Rather than threading an "agent
# config" object through voice_session.py / openai_service.py /
# elevenlabs_service.py (which all just read `config.SOME_SETTING` today),
# we store the override for the current call in a contextvar and resolve it
# lazily via module `__getattr__` (PEP 562). This means:
#   - Every existing `config.SYSTEM_PROMPT` / `config.WELCOME_MESSAGE` / etc.
#     read, anywhere in the codebase, keeps working unchanged.
#   - Overrides are scoped to the asyncio Task of the current WebSocket
#     connection (and any child tasks it creates), so concurrent calls never
#     interfere with each other - no shared mutable global state.
_agent_overrides: "contextvars.ContextVar[dict | None]" = contextvars.ContextVar(
    "lovable_agent_overrides", default=None
)

# Maps the public config name -> the AgentConfig field that can override it.
_OVERRIDABLE_DEFAULTS = {
    "SYSTEM_PROMPT": ("system_prompt", _SYSTEM_PROMPT_DEFAULT),
    "WELCOME_MESSAGE": ("first_message", _WELCOME_MESSAGE_DEFAULT),
    "ELEVENLABS_VOICE_ID": ("voice_id", _ELEVENLABS_VOICE_ID_DEFAULT),
    "ELEVENLABS_STT_LANGUAGE": ("language", _ELEVENLABS_STT_LANGUAGE_DEFAULT),
    "OPENAI_TEMPERATURE": ("temperature", _OPENAI_TEMPERATURE_DEFAULT),
    # OPENAI_MAX_TOKENS intentionally omitted — platform cost cap, never per-agent override.
}


def set_agent_overrides(overrides: dict | None) -> None:
    """
    Set the per-call agent config overrides for the current call (i.e. the
    current asyncio Task and any tasks it spawns from here on). Pass `None`
    (or an empty dict) to fall back to the .env defaults for this call.
    """
    _agent_overrides.set(overrides or None)


def get_agent_overrides() -> dict:
    """Return the overrides active for the current call, or {} if none."""
    return _agent_overrides.get() or {}


def __getattr__(name: str):
    # Only invoked by Python when `name` isn't already a normal module
    # attribute, i.e. for the settings we deliberately did NOT assign above.
    override_info = _OVERRIDABLE_DEFAULTS.get(name)
    if override_info is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    field_name, default_value = override_info
    overrides = _agent_overrides.get()
    if overrides:
        value = overrides.get(field_name)
        if value not in (None, ""):
            return value
    return default_value


def validate() -> list:
    """Return a list of human-readable warnings for missing/invalid config."""
    warnings = []
    pipeline = VOICE_PIPELINE or "compound"
    if pipeline not in ("compound", "openai_realtime"):
        warnings.append(
            f"VOICE_PIPELINE={pipeline!r} is unknown — will fall back to compound at runtime."
        )
    if not OPENAI_API_KEY:
        warnings.append("OPENAI_API_KEY is not set - the agent will not be able to think/respond.")
    elif pipeline == "openai_realtime":
        pass  # key present — Realtime can run without ElevenLabs
    if pipeline == "compound" and not ELEVENLABS_API_KEY:
        warnings.append(
            "ELEVENLABS_API_KEY is not set - speech-to-text and text-to-speech will not work."
        )
    if (
        pipeline == "compound"
        and STREAMING_PIPELINE
        and STT_PROVIDER == "deepgram"
        and not DEEPGRAM_API_KEY
    ):
        warnings.append(
            "STT_PROVIDER=deepgram but DEEPGRAM_API_KEY is not set — streaming STT will fail."
        )
    if not LOVABLE_APP_URL:
        warnings.append("LOVABLE_APP_URL is not set - per-agent token lookup and call-log posting are disabled.")
    if not SUPABASE_URL or (not SUPABASE_SERVICE_ROLE_KEY and not SUPABASE_ANON_KEY):
        warnings.append(
            "Supabase not fully configured - agent reads fall back to Moontech HTTP. "
            "Set SUPABASE_URL plus SUPABASE_SERVICE_ROLE_KEY (preferred) or SUPABASE_ANON_KEY."
        )
    elif SUPABASE_ANON_KEY and not SUPABASE_SERVICE_ROLE_KEY:
        warnings.append(
            "Using SUPABASE_ANON_KEY (publishable) for agent reads — requires Moontech "
            "RLS policies; if lookups fail, Moontech HTTP fallback is used automatically."
        )
    return warnings
