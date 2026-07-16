#!/bin/bash
set -e

# The ./logs folder is usually bind-mounted from the host (see docker-compose.yml)
# and ends up owned by whatever user created it on the host (often root). Fix
# ownership here (while we're still root) so the unprivileged app user can write
# to it, then drop down to that user to actually run the app.
mkdir -p /app/logs
chown -R appuser:appuser /app/logs

exec gosu appuser bash -c "python3 ai_server.py & python3 dashboard.py & wait -n"
