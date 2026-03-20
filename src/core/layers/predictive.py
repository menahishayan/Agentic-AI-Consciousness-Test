from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional


class WorldModelGenerator:
    def predict(self, observation: Any) -> Any:
        raise NotImplementedError("World model prediction not implemented.")


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
