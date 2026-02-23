from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional


class VitalStateMonitor:
    def __init__(self, expected_vitals: Optional[Iterable[str]] = None) -> None:
        self._expected_vitals: List[str] = []
        self._last_state: Dict[str, Any] = {}
        if expected_vitals is not None:
            self.set_expected_vitals(expected_vitals)

    def set_expected_vitals(self, vitals: Iterable[str]) -> None:
        ordered_unique: List[str] = []
        seen = set()
        for name in vitals:
            key = str(name).strip()
            if not key or key in seen:
                continue
            ordered_unique.append(key)
            seen.add(key)
        self._expected_vitals = ordered_unique

    def expected_vitals(self) -> List[str]:
        return list(self._expected_vitals)

    def update(self, payload: Any) -> Dict[str, Any]:
        source = self._coerce_mapping(payload)
        if not self._expected_vitals:
            self.set_expected_vitals(source.keys())
        snapshot = {key: source.get(key) for key in self._expected_vitals}
        self._last_state = snapshot
        return dict(snapshot)

    def last_state(self) -> Dict[str, Any]:
        return dict(self._last_state)

    def missing_vitals(self) -> List[str]:
        return [key for key, value in self._last_state.items() if value is None]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expected_vitals": self.expected_vitals(),
            "state": self.last_state(),
            "missing": self.missing_vitals(),
        }

    @staticmethod
    def _coerce_mapping(payload: Any) -> Mapping[str, Any]:
        if isinstance(payload, Mapping):
            return payload

        if hasattr(payload, "to_dict") and callable(payload.to_dict):
            mapped = payload.to_dict()
            if isinstance(mapped, Mapping):
                return mapped

        if hasattr(payload, "__dict__"):
            mapped = vars(payload)
            if isinstance(mapped, Mapping):
                return mapped

        raise TypeError("VitalStateMonitor.update expects mapping-like payload data.")
