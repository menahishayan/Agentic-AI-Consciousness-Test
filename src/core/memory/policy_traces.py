from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional


class PolicyTraces:
    def __init__(self) -> None:
        self.records: List[Any] = []

    def record(self, trace: Any) -> None:
        self.records.append(trace)

    def query(self, query: Any) -> Any:
        if not isinstance(query, Mapping):
            return list(self.records)

        policy_id = query.get("policy_id")
        limit = query.get("limit")
        out: List[Any] = []
        for record in self.records:
            if policy_id is not None:
                rec_policy_id: Optional[str] = None
                if isinstance(record, dict):
                    rec_policy_id = record.get("policy_id")
                elif hasattr(record, "policy_id"):
                    rec_policy_id = getattr(record, "policy_id")
                if rec_policy_id != policy_id:
                    continue
            out.append(record)

        if isinstance(limit, int) and limit >= 0:
            return out[-limit:]
        return out
