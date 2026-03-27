#!/usr/bin/env bash
# Run the agent. Activate venv first if needed.
# Usage: ./run.sh [max_steps]
#
# Examples:
#   ./run.sh           # 500 steps with settings from config.json
#   ./run.sh 100       # 100 steps
#   MAX_STEPS=50 LOG_PROMPTS=1 ./run.sh

set -e
cd "$(dirname "$0")"

if [ ! -f venv/bin/activate ]; then
  echo "Creating virtual environment..."
  python3 -m venv venv
  source venv/bin/activate
  pip install -q -r requirements.txt
else
  source venv/bin/activate
fi

MAX_STEPS="${1:-${MAX_STEPS:-500}}"
export MAX_STEPS

PYTHONPATH=src python -m src.core.main
