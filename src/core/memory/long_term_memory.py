from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from urllib.parse import quote
from typing import Any, Dict, Iterable, List, Mapping, Optional


def _utc_ts() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


class LongTermMemory:
    _LAYOUT = "ltm_sharded_v1"

    def __init__(
        self,
        path: str = "data/long_term_memory",
        max_score_history: int = 200,
        max_outcome_history: int = 200,
    ) -> None:
        self.path = Path(path)
        self.max_score_history = int(max_score_history)
        self.max_outcome_history = int(max_outcome_history)
        self.root = self._resolve_root(self.path)
        self.manifest_path = self.root / "manifest.json"
        self.policies_dir = self.root / "policies"
        self._data: Dict[str, Dict[str, Any]] = {"policies": {}}
        self._manifest_index: Dict[str, str] = {}
        self.root.mkdir(parents=True, exist_ok=True)
        self.policies_dir.mkdir(parents=True, exist_ok=True)
        self._load()

    def upsert_policies(self, policies: Iterable[Mapping[str, Any]]) -> None:
        now = _utc_ts()
        changed_ids: List[str] = []
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
            changed_ids.append(policy_id)

        if changed_ids:
            self._persist_policies(changed_ids)
            self._persist_manifest()

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

        self._persist_policy(policy_id)
        self._persist_manifest()

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

        self._persist_policy(policy_id)
        self._persist_manifest()

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
        if not self.manifest_path.exists():
            return

        try:
            raw = self.manifest_path.read_text(encoding="utf-8")
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("Invalid long-term-memory manifest payload.")
            if parsed.get("layout") != self._LAYOUT:
                raise ValueError("Unsupported long-term-memory layout.")
            policies_index = parsed.get("policies")
            if not isinstance(policies_index, dict):
                raise ValueError("Invalid long-term-memory manifest index.")
        except Exception:
            self._backup_corrupt(self.manifest_path)
            self._data = {"policies": {}}
            self._manifest_index = {}
            self._persist_manifest()
            return

        policies: Dict[str, Dict[str, Any]] = {}
        clean_index: Dict[str, str] = {}

        for raw_policy_id, raw_rel_path in policies_index.items():
            policy_id = str(raw_policy_id).strip()
            rel_path = str(raw_rel_path).strip()
            if not policy_id or not rel_path:
                continue

            shard_path = self.root / rel_path
            if not shard_path.exists():
                continue
            try:
                raw = shard_path.read_text(encoding="utf-8")
                parsed_policy = json.loads(raw)
                if not isinstance(parsed_policy, dict):
                    raise ValueError("Invalid policy shard payload.")
            except Exception:
                self._backup_corrupt(shard_path)
                continue

            parsed_policy["policy_id"] = policy_id
            policies[policy_id] = parsed_policy
            clean_index[policy_id] = rel_path

        self._data = {"policies": policies}
        self._manifest_index = clean_index
        self._persist_manifest()

    def _persist_policies(self, policy_ids: Iterable[str]) -> None:
        seen = set()
        for policy_id in policy_ids:
            text = str(policy_id).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            self._persist_policy(text)

    def _persist_policy(self, policy_id: str) -> None:
        record = self._policies().get(policy_id)
        if not isinstance(record, dict):
            return
        rel_path = self._manifest_index.get(policy_id)
        if not isinstance(rel_path, str) or not rel_path:
            rel_path = str(Path("policies") / self._policy_filename(policy_id))
            self._manifest_index[policy_id] = rel_path
        shard_path = self.root / rel_path
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(record, ensure_ascii=True, indent=2)
        shard_path.write_text(payload, encoding="utf-8")

    def _persist_manifest(self) -> None:
        policies = self._policies()
        compact_index: Dict[str, str] = {}
        for policy_id in sorted(policies):
            rel_path = self._manifest_index.get(policy_id)
            if not isinstance(rel_path, str) or not rel_path:
                rel_path = str(Path("policies") / self._policy_filename(policy_id))
                self._manifest_index[policy_id] = rel_path
            compact_index[policy_id] = rel_path

        payload = json.dumps(
            {
                "layout": self._LAYOUT,
                "version": 1,
                "policies": compact_index,
            },
            ensure_ascii=True,
            indent=2,
        )
        self.manifest_path.write_text(payload, encoding="utf-8")

    def _policies(self) -> Dict[str, Dict[str, Any]]:
        policies = self._data.get("policies", {})
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
    def _resolve_root(path: Path) -> Path:
        if path.suffix.lower() == ".json":
            return path.parent
        return path

    @staticmethod
    def _policy_filename(policy_id: str) -> str:
        encoded = quote(str(policy_id), safe="-_.~")
        return f"{encoded}.json"

    def _backup_corrupt(self, path: Path) -> None:
        if not path.exists():
            return
        backup = path.with_name(f"{path.name}.corrupt.{_stamp()}")
        try:
            path.rename(backup)
        except Exception:
            pass

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
