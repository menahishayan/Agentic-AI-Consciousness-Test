from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


class WorldModelGenerator:
    def __init__(
        self,
        channels: Optional[Iterable[str]] = None,
        alpha: float = 0.1,
        min_precision: float = 0.3,
        confidence_threshold: int = 20,
    ) -> None:
        resolved_channels = list(channels) if channels is not None else []
        if not resolved_channels:
            resolved_channels = [
                "health",
                "hunger",
                "resource_level",
                "oxygen",
                "threat_proximity",
                "entity_density",
                "terrain_novelty",
            ]
        self.channels: Tuple[str, ...] = tuple(str(channel) for channel in resolved_channels)
        self.alpha = self._clip(float(alpha), 1e-6, 1.0)
        self.min_precision = self._clip01(float(min_precision))
        self.confidence_threshold = max(1, int(confidence_threshold))
        self._delta_ema: Dict[Tuple[str, str], float] = {}
        self._counts: Dict[Tuple[str, str], int] = {}

    def predict(self, observation: Any, action_id: Any) -> Dict[str, float]:
        channels = self._observation_channels(observation)
        action_key = self._normalize_action_id(action_id)
        out: Dict[str, float] = {}
        for channel in self.channels:
            baseline = channels.get(channel, 0.0)
            delta = self._delta_ema.get((action_key, channel), 0.0)
            out[channel] = self._clip01(baseline + delta)
        return out

    def update(self, prev_observation: Any, action_id: Any, next_observation: Any) -> None:
        prev_channels = self._observation_channels(prev_observation)
        next_channels = self._observation_channels(next_observation)
        action_key = self._normalize_action_id(action_id)

        for channel in self.channels:
            key = (action_key, channel)
            observed_delta = next_channels.get(channel, 0.0) - prev_channels.get(channel, 0.0)
            previous_delta = self._delta_ema.get(key, 0.0)
            updated_delta = previous_delta + self.alpha * (observed_delta - previous_delta)
            self._delta_ema[key] = float(updated_delta)
            self._counts[key] = int(self._counts.get(key, 0)) + 1

    def confidence(self, action_id: Any, channel: str) -> float:
        action_key = self._normalize_action_id(action_id)
        channel_key = str(channel)
        count = int(self._counts.get((action_key, channel_key), 0))
        confidence = float(count) / float(self.confidence_threshold)
        return self._clip(confidence, self.min_precision, 1.0)

    def observation_count(self, action_id: Any, channel: str) -> int:
        action_key = self._normalize_action_id(action_id)
        channel_key = str(channel)
        return int(self._counts.get((action_key, channel_key), 0))

    def delta(self, action_id: Any, channel: str) -> float:
        action_key = self._normalize_action_id(action_id)
        channel_key = str(channel)
        return float(self._delta_ema.get((action_key, channel_key), 0.0))

    def reset(self) -> None:
        self._delta_ema.clear()
        self._counts.clear()

    def _observation_channels(self, observation: Any) -> Dict[str, float]:
        if isinstance(observation, Mapping):
            source = observation
        else:
            source = vars(observation) if hasattr(observation, "__dict__") else {}
        out: Dict[str, float] = {}
        for channel in self.channels:
            out[channel] = self._clip01(self._as_float(source.get(channel), 0.0))
        return out

    @staticmethod
    def _normalize_action_id(action_id: Any) -> str:
        if action_id is None:
            return "bootstrap"
        text = str(action_id).strip()
        if not text:
            return "bootstrap"
        return text

    @staticmethod
    def _as_float(value: Any, default: float = 0.0) -> float:
        if isinstance(value, bool):
            return float(default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _clip(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, float(value)))

    @staticmethod
    def _clip01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))


class PredictionErrorCalculator:
    def __init__(self, max_expected_error: float = 1.0, window_size: int = 20) -> None:
        self.max_expected_error = max_expected_error if max_expected_error > 0 else 1.0
        self.window_size = max(1, int(window_size))

    def compute(
        self,
        policy_id: str,
        context: Any = None,
        memory_manager: Any = None,
    ) -> Dict[str, Any]:
        if memory_manager is None:
            return {
                "prediction_error_score": None,
                "reason": "prediction_history_unavailable",
                "samples": 0,
            }

        query = {"policy_id": policy_id, "limit": self.window_size}
        history = self._query_prediction_history(memory_manager=memory_manager, query=query)
        if not isinstance(history, list) or not history:
            return {
                "prediction_error_score": None,
                "reason": "no_prediction_errors",
                "samples": 0,
            }

        magnitudes = self._extract_magnitudes(history)
        if not magnitudes:
            return {
                "prediction_error_score": None,
                "reason": "no_magnitude_values",
                "samples": 0,
            }

        average = float(sum(magnitudes)) / float(len(magnitudes))
        normalized = max(0.0, min(1.0, average / self.max_expected_error))
        return {
            "prediction_error_score": normalized,
            "average_magnitude": average,
            "samples": len(magnitudes),
        }

    def _extract_magnitudes(self, history: List[Any]) -> List[float]:
        out: List[float] = []
        for item in history:
            value = self._magnitude_from_record(item)
            if value is not None:
                out.append(value)
        return out

    def _magnitude_from_record(self, record: Any) -> Optional[float]:
        if isinstance(record, Mapping):
            if isinstance(record.get("magnitude"), (int, float)):
                return abs(float(record["magnitude"]))
            error = record.get("error")
            if isinstance(error, Mapping) and isinstance(error.get("magnitude"), (int, float)):
                return abs(float(error["magnitude"]))

        if hasattr(record, "magnitude"):
            value = getattr(record, "magnitude")
            if isinstance(value, (int, float)):
                return abs(float(value))
        return None

    @staticmethod
    def _query_prediction_history(memory_manager: Any, query: Mapping[str, Any]) -> List[Any]:
        query_prediction_errors = getattr(memory_manager, "query_prediction_errors", None)
        if callable(query_prediction_errors):
            try:
                return query_prediction_errors(
                    policy_id=query.get("policy_id"),
                    area_id=query.get("area_id"),
                    limit=query.get("limit"),
                )
            except Exception:
                pass

        prediction_errors = getattr(memory_manager, "prediction_errors", None)
        if prediction_errors is not None:
            query_method = getattr(prediction_errors, "query", None)
            if callable(query_method):
                try:
                    return query_method(query)
                except Exception:
                    pass
        return []


class PrecisionWeighter:
    def weight(self, prediction: Any, observation: Any) -> Any:
        raise NotImplementedError("Precision weighting not implemented.")
