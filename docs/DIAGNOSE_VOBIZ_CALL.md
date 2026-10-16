# Vobiz call drops / no agent voice — diagnosis

Use this when the caller hears silence and the call ends in ~5–15 seconds.

## Rule of thumb

| During a test call, do `docker compose logs -f` show **new** lines? | Likely layer |
|---------------------------------------------------------------------|--------------|
| **Nothing** after startup (no `Vobiz answer`, no `Vobiz WebSocket connected`) | **Not** AI pipeline — Vobiz → Answer URL → WSS URL → **nginx** → port 8002, or **Lovable** Answer hook |
| `Vobiz answer XML ready wss_target=...` but **no** `Vobiz WebSocket connected` | **nginx WSS** (`/ws` → 8002) or wrong host in XML |
| `Vobiz WebSocket connected` + `agent lookup failed` | **Supabase / Moontech** (`phone_numbers`, RPC, use **service role** key) |
| `Vobiz START` + `Live TTS greeting` / errors | **Backend** (ElevenLabs, streaming, μ-law) |

Your startup-only logs mean: **the media WebSocket never reached this container** until you see `Vobiz WebSocket connected`.

---

## A. Run on VPS (copy-paste)

```bash
cd /www/backend_ai_calling_agent/voice_agent/Voice-agent-backend

# 1) Services up
docker compose ps

# 2) Internal health
curl -s http://127.0.0.1:8081/health
curl -s http://127.0.0.1:5001/api/stats | head -c 200

# 3) Simulate Vobiz answer (inbound) — must return XML with wss://aicalling.nexobulkindia.com/ws/number/...
curl -s -X POST http://127.0.0.1:8081/vobiz/answer \
  -d "CallUUID=test-123&To=%2B919876543210&From=%2B919800000000&Direction=inbound" | head -20

# 4) Same via public HTTPS (nginx)
curl -s -X POST "https://aicalling.nexobulkindia.com/vobiz/answer" \
  -d "CallUUID=test-456&To=%2B919876543210&Direction=inbound" | head -20

# 5) During a REAL call — watch logs (leave running, then call)
docker compose logs -f 2>&1 | grep -E "Vobiz answer|Vobiz WebSocket|Vobiz START|agent lookup|get_inbound|Playing|Live TTS|ERROR"
```

Also tail file log:

```bash
tail -f logs/ai_voice_agent.log | grep -E "Vobiz|agent lookup|get_inbound|ERROR"
```

---

## B. Moontech / Lovable (frontend control plane)

Confirm in portal for that **Vobiz number**:

1. **Answer URL** — either:
   - `https://aicalling.nexobulkindia.com/vobiz/answer` (voice server), **or**
   - Moontech hook e.g. `https://www.moontechpro.com/api/public/hooks/vobiz-inbound?...` (Lovable)

2. If Answer goes to **Lovable**, Lovable must return XML whose `<Stream>` URL is exactly:
   - `wss://aicalling.nexobulkindia.com/ws?token=...` **or**
   - `wss://aicalling.nexobulkindia.com/ws/number/<digits>`

   Wrong host (`voice.moontechpro.com`, `localhost`, ngrok) → **your VPS logs stay empty**.

3. **Hangup URL**: `https://aicalling.nexobulkindia.com/vobiz/hangup` (or Lovable relay that forwards to voice server).

4. Supabase **`phone_numbers`**: row for that Vobiz DID, `provider=vobiz`, `agent_id`, `status=active`.

5. RPC **`get_inbound_agent_bundle(phone_number)`** returns agent for that number (last 10 digits).

Ask Lovable team to confirm: *“On vobiz-inbound / answer hook, what is the exact `wss://` string in Stream XML?”*

---

## C. aaPanel / nginx

Required:

| Path | Backend |
|------|---------|
| `/ws` | `http://127.0.0.1:8002` + WebSocket upgrade headers |
| `/vobiz/` | `http://127.0.0.1:8081` |
| `/exotel/` | `http://127.0.0.1:8081` |

**Do not** expose 8002/8081 as public HTTPS without nginx — TLS bytes on plain HTTP cause log noise and broken clients.

Test WSS locally with `websockets` client optional; minimum is: answer curl returns correct `wss://aicalling.../ws/number/...`.

---

## D. Backend `.env`

| Variable | Required |
|----------|----------|
| `PUBLIC_HOST=aicalling.nexobulkindia.com` | Yes (you fixed this) |
| `SUPABASE_SERVICE_ROLE_KEY` | Strongly recommended (anon + RLS → agent lookup fail) |
| `LOVABLE_APP_URL` | For fallback agent fetch + call-log |
| `OPENAI_API_KEY`, `ELEVENLABS_API_KEY` | Yes |

After `.env` change: `docker compose up -d --build`

---

## E. Eventlet warning

Dashboard uses Eventlet — **deprecated warning only**, does not drop phone calls.

---

## F. What to send for support

One test call + output of:

```bash
docker compose logs --since=5m 2>&1 | grep -E "Vobiz|agent|ERROR|get_inbound"
curl -s -X POST "https://aicalling.nexobulkindia.com/vobiz/answer" -d "To=CLIENT_NUMBER&CallUUID=x" 
```

Plus: Moontech screenshot of Answer/Hangup URLs and Stream URL from Lovable hook (if used).
