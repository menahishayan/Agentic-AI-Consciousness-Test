from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Mapping, Optional


class PredictionErrorHistory:
    def __init__(self) -> None:
        self.records: List[Any] = []
        self._area_stats: Dict[str, Dict[str, float]] = {}

    def record(self, *args: Any) -> None:
        if len(args) == 0:
            return
        if len(args) == 1:
            area_id = self._extract_area_id(args[0])
            payload = args[0]
        else:
            area_id = self._normalize_area_id(args[0])
            payload = self._record_payload(area_id, args[1])

        self.records.append(payload)
        magnitude = self._extract_magnitude(payload)
        if area_id is not None and magnitude is not None:
            stats = self._area_stats.setdefault(area_id, {"sum": 0.0, "count": 0.0})
            stats["sum"] += abs(float(magnitude))
            stats["count"] += 1.0

    def get_area_familiarity(self, area_id: str) -> float:
        key = self._normalize_area_id(area_id)
        if key is None:
            return 0.5
        stats = self._area_stats.get(key)
        if not isinstance(stats, dict):
            return 0.5
        count = float(stats.get("count", 0.0))
        if count <= 0.0:
            return 0.5
        avg_magnitude = float(stats.get("sum", 0.0)) / count
        familiarity = 1.0 - avg_magnitude
        return max(0.0, min(1.0, familiarity))

    def query(self, query: Any) -> Any:
        if not isinstance(query, Mapping):
            return list(self.records)

        policy_id = query.get("policy_id")
        area_id = self._normalize_area_id(query.get("area_id"))
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
            if area_id is not None:
                rec_area_id = self._extract_area_id(record)
                if rec_area_id != area_id:
                    continue
            out.append(record)

        if isinstance(limit, int) and limit >= 0:
            return out[-limit:]
        return out

    def _record_payload(self, area_id: Optional[str], error: Any) -> Any:
        if isinstance(error, Mapping):
            payload = dict(error)
            if area_id is not None and "area_id" not in payload:
                payload["area_id"] = area_id
            return payload

        if is_dataclass(error):
            payload = asdict(error)
            if area_id is not None:
                payload["area_id"] = area_id
            return payload

        return {
            "area_id": area_id,
            "error": error,
            "magnitude": self._extract_magnitude(error),
        }

    def _extract_area_id(self, record: Any) -> Optional[str]:
        if isinstance(record, Mapping):
            return self._normalize_area_id(record.get("area_id"))
        if hasattr(record, "area_id"):
            return self._normalize_area_id(getattr(record, "area_id"))
        return None

    @staticmethod
    def _normalize_area_id(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _extract_magnitude(self, record: Any) -> Optional[float]:
        if isinstance(record, Mapping):
            value = record.get("magnitude")
            if isinstance(value, (int, float)):
                return float(value)
            nested = record.get("error")
            if isinstance(nested, Mapping):
                nested_value = nested.get("magnitude")
                if isinstance(nested_value, (int, float)):
                    return float(nested_value)

        if hasattr(record, "magnitude"):
            value = getattr(record, "magnitude")
            if isinstance(value, (int, float)):
                return float(value)
        return None
