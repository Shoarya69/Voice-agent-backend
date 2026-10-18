# Production: `aicalling.nexobulkindia.com`

Same VPS/domain as the **existing Exotel** voice bot. Exotel + Vobiz + webhooks share one host; only paths and ports differ behind nginx.

---

## `.env` on server

```env
HOST=0.0.0.0
PORT=8002
WEBHOOK_PORT=8081

# Vobiz Stream XML + WSS hostname (no https://)
PUBLIC_HOST=aicalling.nexobulkindia.com
VOBIZ_ENABLED=true
VOBIZ_WS_PATH=/ws

ENABLE_SUPABASE_CALL_RECORDS=true
# … OPENAI_*, ELEVENLABS_*, SUPABASE_*, LOVABLE_* unchanged
```

Redeploy after edit:

```bash
docker compose up -d --build
```

---

## Public URLs (paste in consoles)

### Exotel (unchanged pattern)

| Use | URL |
|-----|-----|
| Voicebot WebSocket | `wss://aicalling.nexobulkindia.com/` (or path Exotel Flow already uses, e.g. `?token=…`) |
| Status callback | `https://aicalling.nexobulkindia.com/exotel/status` |
| Conversation (if used) | `https://aicalling.nexobulkindia.com/exotel/conversation` |

nginx must proxy **WebSocket** to `127.0.0.1:8002` and **`/exotel/*`** to `127.0.0.1:8081`.

### Vobiz (Moontech client numbers)

| Use | URL |
|-----|-----|
| Application **Answer URL** | `https://aicalling.nexobulkindia.com/vobiz/answer` |
| Application **Hangup URL** | `https://aicalling.nexobulkindia.com/vobiz/hangup` |
| Media stream (in XML / Moontech webhook) | `wss://aicalling.nexobulkindia.com/ws?token=…` or `wss://aicalling.nexobulkindia.com/ws/number/91XXXXXXXXXX` |

Inbound direct to voice server: Answer URL returns XML → WSS with `/number/{To}` → Supabase agent by dialed number.

---

## Dashboard (`templates/dashboard.html`)

Flask runs on **port 5001** inside Docker (`dashboard.py`). It is **not** on the WebSocket port.

If the browser shows plain text:

`AI Voice Agent WebSocket server is running.`

then nginx (or aaPanel) is sending **HTTP** requests to **8002** (`ai_server.py`) instead of **5001**. That line is the voice bot’s health reply for non-WebSocket GETs — not an error in Docker.

**Correct URL after nginx fix:** `https://aicalling.nexobulkindia.com/dashboard/` → `127.0.0.1:5001`

---

## nginx example (443 → 8002 / 8081 / 5001)

Adjust if you already have a working Exotel config — **add** the `/ws`, `/dashboard/`, and `/vobiz/` blocks without breaking existing Exotel WSS `location /`.

```nginx
map $http_upgrade $connection_upgrade {
    default upgrade;
    ''      close;
}

server {
    listen 443 ssl http2;
    server_name aicalling.nexobulkindia.com;

    ssl_certificate     /etc/letsencrypt/live/aicalling.nexobulkindia.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/aicalling.nexobulkindia.com/privkey.pem;

    # Vobiz μ-law WebSocket
    location /ws {
        proxy_pass http://127.0.0.1:8002;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $connection_upgrade;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }

    # Exotel Voicebot WSS (default location — keep if already used)
    location / {
        proxy_pass http://127.0.0.1:8002;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $connection_upgrade;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }

    # AgentStream dashboard (Flask + dashboard.html) — MUST be 5001, not 8002
    location /dashboard/ {
        proxy_pass http://127.0.0.1:5001/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $connection_upgrade;
    }

    # Webhooks (Exotel + Vobiz + prewarm)
    location ~ ^/(exotel|vobiz|voicebot|prewarm) {
        proxy_pass http://127.0.0.1:8081;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### aaPanel

In **Website → aicalling.nexobulkindia.com → Config**, add the `location /dashboard/` block **above** any broad `location /` that points to port **8002**. Reload nginx.

Quick test on the VPS:

```bash
curl -s http://127.0.0.1:5001/ | head -5          # should contain HTML / AgentStream
curl -s http://127.0.0.1:8002/                    # "AI Voice Agent WebSocket server is running."
```

```bash
sudo nginx -t && sudo systemctl reload nginx
```

Docker on the same machine:

```bash
cd /path/to/Agent-Stream-echobot
docker compose up -d --build
docker compose logs -f --tail=100 ai-voice-agent
```

---

## Quick checks

```bash
curl -s http://127.0.0.1:8081/health
curl -s http://127.0.0.1:5001/api/stats
curl -sI https://aicalling.nexobulkindia.com/vobiz/answer   # POST body from Vobiz; GET may still return XML or 405
```

Logs: `logs/ai_voice_agent.log` — look for `Vobiz media WebSocket path: /ws (wss://aicalling.nexobulkindia.com/ws)`.

---

## Moontech portal

- Exotel Stream URL host stays **`aicalling.nexobulkindia.com`** (same as before).
- Vobiz Stream URL: **`wss://aicalling.nexobulkindia.com/ws`** (+ token or number path).
- Do **not** set `PUBLIC_HOST` to a different domain unless DNS and nginx match.
