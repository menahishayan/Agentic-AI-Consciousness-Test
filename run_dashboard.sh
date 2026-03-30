#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if [ -d "venv" ]; then
  source venv/bin/activate
fi

PYTHONPATH=. python -m dashboard.server "$@"
