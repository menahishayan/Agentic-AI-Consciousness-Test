"""
Scans the logs directory for run directories and tracks their state.
"""

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dashboard.config import LOGS_ROOT, RUN_DIR_PATTERN


@dataclass
class RunInfo:
    run_id: str
    run_dir: Path
    metadata: dict
    is_complete: bool
    is_cancelled: bool = False
    step_count: int = 0
    first_seen_ts: float = field(default_factory=time.time)


class RunRegistry:
    def __init__(self, logs_root: Path = LOGS_ROOT) -> None:
        self._logs_root = logs_root
        self._runs: dict[str, RunInfo] = {}
        self._pattern = re.compile(RUN_DIR_PATTERN)

    def scan(self) -> list[str]:
        """
        Scan logs_root for run directories.
        Returns list of newly discovered run_ids.
        Also updates step_count and is_complete for known active runs.
        """
        if not self._logs_root.exists():
            return []

        new_ids: list[str] = []
        try:
            entries = list(self._logs_root.iterdir())
        except OSError:
            return []

        for entry in entries:
            if not entry.is_dir():
                continue
            if not self._pattern.match(entry.name):
                continue
            run_id = entry.name
            if run_id not in self._runs:
                is_complete = self._check_complete(entry)
                is_cancelled = False if is_complete else self._check_cancelled(entry)
                info = RunInfo(
                    run_id=run_id,
                    run_dir=entry,
                    metadata=self._load_metadata(entry),
                    is_complete=is_complete,
                    is_cancelled=is_cancelled,
                    step_count=self._count_steps(entry),
                )
                self._runs[run_id] = info
                new_ids.append(run_id)
            else:
                info = self._runs[run_id]
                # refresh metadata if not yet loaded
                if not info.metadata:
                    info.metadata = self._load_metadata(entry)
                # update step count and completion for active runs
                if not info.is_complete and not info.is_cancelled:
                    info.step_count = self._count_steps(entry)
                    info.is_complete = self._check_complete(entry)
                    if not info.is_complete:
                        info.is_cancelled = self._check_cancelled(entry)

        return new_ids

    def get(self, run_id: str) -> Optional[RunInfo]:
        return self._runs.get(run_id)

    def all_runs(self) -> list[RunInfo]:
        """All known runs sorted by start_time descending (newest first)."""
        runs = list(self._runs.values())
        runs.sort(key=lambda r: r.metadata.get("start_time", r.first_seen_ts), reverse=True)
        return runs

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_metadata(self, run_dir: Path) -> dict:
        run_json = run_dir / "run.json"
        if not run_json.exists():
            return {}
        try:
            return json.loads(run_json.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def _check_complete(self, run_dir: Path) -> bool:
        """Check if episode_complete event exists by reading the tail of events.jsonl."""
        return self._events_tail_contains(run_dir, "episode_complete")

    def _check_cancelled(self, run_dir: Path) -> bool:
        """Check if episode_cancelled event exists by reading the tail of events.jsonl."""
        return self._events_tail_contains(run_dir, "episode_cancelled")

    def _events_tail_contains(self, run_dir: Path, keyword: str) -> bool:
        events_path = run_dir / "events.jsonl"
        if not events_path.exists():
            return False
        try:
            size = events_path.stat().st_size
            if size == 0:
                return False
            read_size = min(512, size)
            with open(events_path, "rb") as f:
                f.seek(max(0, size - read_size))
                tail = f.read().decode("utf-8", errors="ignore")
            return keyword in tail
        except OSError:
            return False

    def _count_steps(self, run_dir: Path) -> int:
        """Count lines in metrics.jsonl as a proxy for step count."""
        metrics_path = run_dir / "metrics.jsonl"
        if not metrics_path.exists():
            return 0
        try:
            with open(metrics_path, "rb") as f:
                return sum(1 for line in f if line.strip())
        except OSError:
            return 0
