"""
LongTermMemory — JSON-persisted policy performance history.

Survives restarts. Provides PolicyGenerator with historical success rates
for each policy_id across all prior episodes.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.models.memory_records import LongTermPolicyRecord

log = logging.getLogger(__name__)


class LongTermMemory:
    def __init__(self, config: Dict[str, Any]) -> None:
        path_str = config.get("long_term_memory_path", "data/long_term_memory")
        self._dir = Path(path_str)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._file = self._dir / "policies.json"
        self._max_score_history: int = int(config.get("max_score_history", 200))
        self._max_outcome_history: int = int(config.get("max_outcome_history", 200))

        self._records: Dict[str, LongTermPolicyRecord] = {}
        self._load()

    def record_outcome(
        self,
        policy_id: str,
        score: float,
        outcome: str,  # "success" | "failure" | "partial"
    ) -> None:
        if policy_id not in self._records:
            self._records[policy_id] = LongTermPolicyRecord(policy_id=policy_id)

        rec = self._records[policy_id]
        rec.score_history.append(score)
        rec.outcome_history.append(outcome)
        rec.total_selections += 1
        if outcome == "success":
            rec.total_successes += 1

        # Trim history
        if len(rec.score_history) > self._max_score_history:
            rec.score_history = rec.score_history[-self._max_score_history:]
        if len(rec.outcome_history) > self._max_outcome_history:
            rec.outcome_history = rec.outcome_history[-self._max_outcome_history:]

        self._save()

    def get_success_rate(self, policy_id: str) -> float:
        rec = self._records.get(policy_id)
        if rec is None:
            return 0.5  # Neutral prior for unseen policies
        return rec.success_rate

    def get_all_records(self) -> Dict[str, LongTermPolicyRecord]:
        return dict(self._records)

    def _load(self) -> None:
        if not self._file.exists():
            return
        try:
            data = json.loads(self._file.read_text())
            for pid, rec_dict in data.items():
                self._records[pid] = LongTermPolicyRecord(
                    policy_id=pid,
                    score_history=rec_dict.get("score_history", []),
                    outcome_history=rec_dict.get("outcome_history", []),
                    total_selections=rec_dict.get("total_selections", 0),
                    total_successes=rec_dict.get("total_successes", 0),
                )
            log.info("LongTermMemory: loaded %d policy records", len(self._records))
        except Exception as exc:
            log.warning("LongTermMemory: failed to load from %s: %s", self._file, exc)

    def _save(self) -> None:
        try:
            data = {
                pid: {
                    "score_history": rec.score_history,
                    "outcome_history": rec.outcome_history,
                    "total_selections": rec.total_selections,
                    "total_successes": rec.total_successes,
                }
                for pid, rec in self._records.items()
            }
            self._file.write_text(json.dumps(data, indent=2))
        except Exception as exc:
            log.warning("LongTermMemory: failed to save: %s", exc)
