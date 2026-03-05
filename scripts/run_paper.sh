#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ "${PAPER_TRADING:-true}" != "true" ]]; then
  echo "Refusing to start: PAPER_TRADING must be true"
  exit 1
fi

mkdir -p logs
SESSION="arb-paper"
LOG_FILE="logs/paper_$(date +%Y%m%d).log"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required."
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session $SESSION already exists. Attach with: tmux attach -t $SESSION"
  exit 0
fi

CMD="while true; do echo \"[\$(date -u +%FT%TZ)] starting main.py\" >> $LOG_FILE; .venv/bin/python main.py >> $LOG_FILE 2>&1 || true; echo \"[\$(date -u +%FT%TZ)] crashed, restarting in 30s\" >> $LOG_FILE; sleep 30; done"
tmux new-session -d -s "$SESSION" "$CMD"
echo "Started tmux session: $SESSION"
echo "Log file: $LOG_FILE"
