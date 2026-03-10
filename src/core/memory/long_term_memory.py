from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


def _utc_ts() -> str:
    return datetime.utcnow().isoformat() + "Z"


class LongTermMemory:
    def __init__(
        self,
        path: str = "data/long_term_memory/policies.json",
        max_score_history: int = 200,
        max_outcome_history: int = 200,
    ) -> None:
        self.path = Path(path)
        self.max_score_history = int(max_score_history)
        self.max_outcome_history = int(max_outcome_history)
        self._data: Dict[str, Any] = {"version": 1, "policies": {}}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def upsert_policies(self, policies: Iterable[Mapping[str, Any]]) -> None:
        now = _utc_ts()
        changed = False
        for policy in policies:
            policy_id = str(policy.get("policy_id") or "").strip()
            if not policy_id:
                continue
            record = self._ensure_policy_record(policy_id)

            for key in ("adapter_folder", "callable_name", "source", "signature", "description"):
                value = policy.get(key)
                if value is not None:
                    record[key] = str(value)

            tags = self._normalize_tags(policy.get("tags"))
            if tags:
                record["tags"] = tags

            record["last_seen_at"] = now
            if not record.get("discovered_at"):
                record["discovered_at"] = now
            changed = True

        if changed:
            self._persist()

    def record_policy_selection(
        self,
        policy_id: str,
        score: float,
        components: Any,
        step: Optional[int],
    ) -> None:
        record = self._ensure_policy_record(policy_id)
        now = _utc_ts()
        record["selected_count"] = int(record.get("selected_count", 0)) + 1
        record["last_selected_at"] = now
        record["last_score"] = float(score)

        history = record.setdefault("score_history", [])
        if isinstance(history, list):
            history.append(
                {
                    "ts": now,
                    "score": float(score),
                    "components": components,
                    "step": step,
                }
            )
            if len(history) > self.max_score_history:
                del history[:-self.max_score_history]

        self._persist()

    def record_policy_outcome(
        self,
        policy_id: str,
        reward: Any,
        done: Any,
        step: Optional[int],
    ) -> None:
        record = self._ensure_policy_record(policy_id)
        now = _utc_ts()
        reward_value = self._as_float_or_none(reward)
        if reward_value is not None and reward_value > 0:
            record["success_count"] = int(record.get("success_count", 0)) + 1

        outcomes = record.setdefault("outcome_history", [])
        if isinstance(outcomes, list):
            outcomes.append(
                {
                    "ts": now,
                    "reward": reward,
                    "done": bool(done),
                    "step": step,
                }
            )
            if len(outcomes) > self.max_outcome_history:
                del outcomes[:-self.max_outcome_history]

        self._persist()

    def get_policies(self, adapter_folder: Optional[str] = None) -> List[Dict[str, Any]]:
        policies = self._policies()
        out = []
        for policy in policies.values():
            if not isinstance(policy, dict):
                continue
            if adapter_folder is not None and policy.get("adapter_folder") != adapter_folder:
                continue
            out.append(dict(policy))
        out.sort(key=lambda item: str(item.get("policy_id", "")))
        return out

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = self.path.read_text(encoding="utf-8")
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("Invalid long-term-memory payload.")
            policies = parsed.get("policies")
            if not isinstance(policies, dict):
                raise ValueError("Invalid long-term-memory policies map.")
            self._data = {"version": int(parsed.get("version", 1)), "policies": policies}
        except Exception:
            backup_name = f"{self.path.name}.corrupt.{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            backup = self.path.with_name(backup_name)
            try:
                self.path.rename(backup)
            except Exception:
                pass
            self._data = {"version": 1, "policies": {}}
            self._persist()

    def _persist(self) -> None:
        payload = json.dumps(self._data, ensure_ascii=True, indent=2)
        self.path.write_text(payload, encoding="utf-8")

    def _policies(self) -> Dict[str, Dict[str, Any]]:
        policies = self._data.setdefault("policies", {})
        if not isinstance(policies, dict):
            self._data["policies"] = {}
            return self._data["policies"]
        return policies

    def _ensure_policy_record(self, policy_id: str) -> Dict[str, Any]:
        policies = self._policies()
        record = policies.get(policy_id)
        now = _utc_ts()
        if not isinstance(record, dict):
            record = {
                "policy_id": policy_id,
                "adapter_folder": "",
                "callable_name": "",
                "source": "unknown",
                "signature": "",
                "tags": [],
                "description": "",
                "discovered_at": now,
                "last_seen_at": now,
                "selected_count": 0,
                "success_count": 0,
                "last_selected_at": "",
                "last_score": None,
                "score_history": [],
                "outcome_history": [],
            }
            policies[policy_id] = record
        return record

    @staticmethod
    def _normalize_tags(raw: Any) -> List[str]:
        if not isinstance(raw, Iterable) or isinstance(raw, (str, bytes)):
            return []
        out: List[str] = []
        seen = set()
        for value in raw:
            tag = str(value).strip().lower()
            if not tag or tag in seen:
                continue
            out.append(tag)
            seen.add(tag)
        return out

    @staticmethod
    def _as_float_or_none(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
