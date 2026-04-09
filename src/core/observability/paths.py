"""Run directory management."""
from __future__ import annotations

import time
import uuid
from pathlib import Path


def make_run_dir(log_root: str) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_id = uuid.uuid4().hex[:8]
    run_dir = Path(log_root) / f"{ts}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
