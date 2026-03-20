from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional


@dataclass
class AreaStats:
    area_id: str
    pe_mean: float
    pe_variance: float
    threat_mean: float
    count: int
    last_tick: int


class PredictionErrorHistory:
    def __init__(self, min_observations: int = 5, ema_alpha: float = 0.1) -> None:
        self.min_observations = max(1, int(min_observations))
        self.ema_alpha = max(1e-6, min(1.0, float(ema_alpha)))
        self._stats: Dict[str, AreaStats] = {}
        self._records: List[Dict[str, Any]] = []
        self._welford_mean: Dict[str, float] = {}
        self._welford_m2: Dict[str, float] = {}
        self._threat_seen: set[str] = set()

    def record(self, area_id: str, error: Any) -> None:
        normalized_area = self._normalize_area_id(area_id)
        if normalized_area is None:
            return

        magnitude = self._extract_magnitude(error)
        if magnitude is None:
            return

        magnitude = self._clip01(abs(float(magnitude)))
        source = self._extract_source(error)
        tick = self._extract_tick(error)
        policy_id = self._extract_policy_id(error)
        channel = self._extract_channel(error)

        self._records.append(
            {
                "area_id": normalized_area,
                "policy_id": policy_id,
                "channel": channel,
                "magnitude": magnitude,
                "source": source,
                "tick": tick,
            }
        )

        stats = self._stats.get(normalized_area)
        if stats is None:
            stats = AreaStats(
                area_id=normalized_area,
                pe_mean=magnitude,
                pe_variance=0.0,
                threat_mean=0.0,
                count=0,
                last_tick=0 if tick is None else int(tick),
            )
            self._stats[normalized_area] = stats

        # EMA mean for familiarity/threat estimates.
        if stats.count == 0:
            stats.pe_mean = magnitude
        else:
            stats.pe_mean = self.ema_alpha * magnitude + (1.0 - self.ema_alpha) * stats.pe_mean

        # Welford online variance for PE magnitudes.
        prev_count = int(stats.count)
        new_count = prev_count + 1
        mean = self._welford_mean.get(normalized_area, 0.0)
        m2 = self._welford_m2.get(normalized_area, 0.0)
        delta = magnitude - mean
        mean += delta / float(new_count)
        delta2 = magnitude - mean
        m2 += delta * delta2
        variance = m2 / float(new_count)
        self._welford_mean[normalized_area] = mean
        self._welford_m2[normalized_area] = m2

        if source == "threat":
            if normalized_area in self._threat_seen:
                stats.threat_mean = (
                    self.ema_alpha * magnitude + (1.0 - self.ema_alpha) * stats.threat_mean
                )
            else:
                stats.threat_mean = magnitude
                self._threat_seen.add(normalized_area)

        stats.pe_variance = max(0.0, float(variance))
        stats.count = new_count
        if tick is not None:
            stats.last_tick = int(tick)

    def get_area_familiarity(self, area_id: str) -> float:
        normalized_area = self._normalize_area_id(area_id)
        if normalized_area is None:
            return 0.0
        stats = self._stats.get(normalized_area)
        if stats is None or int(stats.count) < self.min_observations:
            return 0.0
        return self._clip01(1.0 - self._clip01(stats.pe_mean))

    def get_area_threat_prior(self, area_id: str) -> float:
        normalized_area = self._normalize_area_id(area_id)
        if normalized_area is None:
            return 0.0
        stats = self._stats.get(normalized_area)
        if stats is None or normalized_area not in self._threat_seen:
            return 0.0
        return self._clip01(stats.threat_mean)

    def query(self, query: Any) -> List[Dict[str, Any]]:
        if not isinstance(query, Mapping):
            return list(self._records)

        policy_id = self._normalize_optional_text(query.get("policy_id"))
        area_id = self._normalize_optional_text(query.get("area_id"))
        channel = self._normalize_optional_text(query.get("channel"))
        source = self._normalize_optional_text(query.get("source"))
        limit = query.get("limit")

        out: List[Dict[str, Any]] = []
        for record in self._records:
            if policy_id is not None and self._normalize_optional_text(record.get("policy_id")) != policy_id:
                continue
            if area_id is not None and self._normalize_optional_text(record.get("area_id")) != area_id:
                continue
            if channel is not None and self._normalize_optional_text(record.get("channel")) != channel:
                continue
            if source is not None and self._normalize_optional_text(record.get("source")) != source:
                continue
            out.append(dict(record))

        if isinstance(limit, int) and limit >= 0:
            return out[-limit:]
        return out

    def clear(self) -> None:
        self._stats.clear()
        self._records.clear()
        self._welford_mean.clear()
        self._welford_m2.clear()
        self._threat_seen.clear()

    @staticmethod
    def _normalize_area_id(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return text

    @staticmethod
    def _extract_magnitude(error: Any) -> Optional[float]:
        if isinstance(error, Mapping):
            value = error.get("magnitude")
            if isinstance(value, (int, float)):
                return float(value)
            nested = error.get("error")
            if isinstance(nested, Mapping):
                nested_value = nested.get("magnitude")
                if isinstance(nested_value, (int, float)):
                    return float(nested_value)
        if hasattr(error, "magnitude"):
            value = getattr(error, "magnitude")
            if isinstance(value, (int, float)):
                return float(value)
        return None

    @staticmethod
    def _extract_source(error: Any) -> Optional[str]:
        if isinstance(error, Mapping):
            source = error.get("source")
            if isinstance(source, str):
                return source.strip().lower() or None
            nested = error.get("error")
            if isinstance(nested, Mapping):
                nested_source = nested.get("source")
                if isinstance(nested_source, str):
                    return nested_source.strip().lower() or None
        if hasattr(error, "source"):
            source = getattr(error, "source")
            if isinstance(source, str):
                return source.strip().lower() or None
        return None

    @staticmethod
    def _extract_policy_id(error: Any) -> Optional[str]:
        if isinstance(error, Mapping):
            policy_id = error.get("policy_id")
            if isinstance(policy_id, str):
                normalized = policy_id.strip()
                return normalized or None
            nested = error.get("error")
            if isinstance(nested, Mapping):
                nested_policy_id = nested.get("policy_id")
                if isinstance(nested_policy_id, str):
                    normalized_nested = nested_policy_id.strip()
                    return normalized_nested or None
        if hasattr(error, "policy_id"):
            policy_id = getattr(error, "policy_id")
            if isinstance(policy_id, str):
                normalized_attr = policy_id.strip()
                return normalized_attr or None
        return None

    @staticmethod
    def _extract_channel(error: Any) -> Optional[str]:
        if isinstance(error, Mapping):
            channel = error.get("channel")
            if isinstance(channel, str):
                normalized = channel.strip().lower()
                return normalized or None
            nested = error.get("error")
            if isinstance(nested, Mapping):
                nested_channel = nested.get("channel")
                if isinstance(nested_channel, str):
                    normalized_nested = nested_channel.strip().lower()
                    return normalized_nested or None
        if hasattr(error, "channel"):
            channel = getattr(error, "channel")
            if isinstance(channel, str):
                normalized_attr = channel.strip().lower()
                return normalized_attr or None
        return None

    @staticmethod
    def _extract_tick(error: Any) -> Optional[int]:
        if isinstance(error, Mapping):
            tick = error.get("tick")
            if isinstance(tick, int):
                return int(tick)
            nested = error.get("error")
            if isinstance(nested, Mapping):
                nested_tick = nested.get("tick")
                if isinstance(nested_tick, int):
                    return int(nested_tick)
        if hasattr(error, "tick"):
            tick = getattr(error, "tick")
            if isinstance(tick, int):
                return int(tick)
        return None

    @staticmethod
    def _clip01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _normalize_optional_text(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return text
