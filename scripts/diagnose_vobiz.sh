#!/usr/bin/env bash
# Quick Vobiz + voice server checks (run on VPS in repo root)
set -euo pipefail

HOST="${PUBLIC_HOST:-aicalling.nexobulkindia.com}"
TO="${1:-919876543210}"

echo "=== Docker ==="
docker compose ps 2>/dev/null || docker-compose ps

echo ""
echo "=== Health ==="
curl -sf http://127.0.0.1:8081/health && echo " webhook OK" || echo " webhook FAIL"
curl -sf http://127.0.0.1:5001/api/stats >/dev/null && echo " dashboard OK" || echo " dashboard FAIL"

echo ""
echo "=== Vobiz answer (local 8081) ==="
curl -s -X POST "http://127.0.0.1:8081/vobiz/answer" \
  -d "CallUUID=diag-local&To=${TO}&From=919800000000&Direction=inbound" | head -15

echo ""
echo "=== Vobiz answer (public https://${HOST}) ==="
curl -s -X POST "https://${HOST}/vobiz/answer" \
  -d "CallUUID=diag-public&To=${TO}&From=919800000000&Direction=inbound" | head -15

echo ""
echo "=== Recent Vobiz lines in container log (5m) ==="
docker compose logs --since=5m 2>&1 | grep -E "Vobiz|agent lookup|get_inbound|ERROR" || echo "(none — no call hit backend recently)"

echo ""
echo "=== Tip: during a live call run: docker compose logs -f | grep Vobiz ==="
