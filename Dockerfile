# AI Voice Agent - Production Docker image
# Build:  docker build -t ai-voice-agent .
# Run:    docker run --env-file .env -p 5000:5000 -p 5001:5001 ai-voice-agent

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (curl for the healthcheck, gosu to safely drop root privileges at runtime,
# build-essential for native packages such as webrtcvad)
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl gosu build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p logs \
    && useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app \
    && chmod +x entrypoint.sh

# NOTE: we intentionally stay root here (not `USER appuser`) because the
# ./logs directory is bind-mounted from the host and its ownership can vary.
# entrypoint.sh fixes permissions at container start, then drops to appuser.

EXPOSE 8002 5001

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:5001/api/stats || exit 1

ENTRYPOINT ["./entrypoint.sh"]
