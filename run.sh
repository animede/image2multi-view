#!/usr/bin/env bash
# Start image2multi-view. Uses flux3-klein-studio's .venv if it is next to this folder, else ./.venv.
set -euo pipefail
cd "$(dirname "$0")"
PY=""
for cand in ".venv/bin/python" "../flux3-klein-studio/.venv/bin/python"; do
  if [ -x "$cand" ] && "$cand" -c "import fastapi, uvicorn, httpx, PIL, multipart" 2>/dev/null; then PY="$cand"; break; fi
done
if [ -z "$PY" ]; then
  echo "依存ライブラリが見つかりません。python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi
export KLEIN_URL="${KLEIN_URL:-http://127.0.0.1:7864}"
HOST="${MV_HOST:-127.0.0.1}"
PORT="${MV_PORT:-7870}"
echo "image2multi-view -> http://${HOST}:${PORT}/  (KLEIN_URL=${KLEIN_URL}, python=${PY})"
exec "$PY" -m uvicorn app:app --host "$HOST" --port "$PORT" --workers 1
