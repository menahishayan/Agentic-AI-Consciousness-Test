from __future__ import annotations

from copy import deepcopy
from typing import Any, List, Mapping, Optional


class SelfStateMemory:
    def __init__(self, max_records: int = 1000) -> None:
        self.max_records = max(1, int(max_records))
        self.records: List[Any] = []

    def record(self, snapshot: Any) -> None:
        self.records.append(deepcopy(snapshot))
        if len(self.records) > self.max_records:
            del self.records[:-self.max_records]

    def query(self, query: Any) -> Any:
        if not isinstance(query, Mapping):
            return deepcopy(self.records)

        step = query.get("step")
        phase = query.get("phase")
        limit = query.get("limit")

        out = list(self.records)
        if step is not None:
            out = [
                item
                for item in out
                if self._field(item, "step") == step
            ]
        if phase is not None:
            out = [
                item
                for item in out
                if self._field(item, "phase") == phase
            ]

        if isinstance(limit, int) and limit >= 0:
            out = out[-limit:]
        return deepcopy(out)

    @staticmethod
    def _field(item: Any, key: str) -> Optional[Any]:
        if isinstance(item, Mapping):
            return item.get(key)
        if hasattr(item, key):
            return getattr(item, key)
        return None
