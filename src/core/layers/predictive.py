from __future__ import annotations

from typing import Any


class WorldModelGenerator:
    def predict(self, observation: Any) -> Any:
        raise NotImplementedError("World model prediction not implemented.")


class PredictionErrorCalculator:
    def compute(self, prediction: Any, observation: Any) -> Any:
        raise NotImplementedError("Prediction error calculation not implemented.")


class PrecisionWeighter:
    def weight(self, prediction: Any, observation: Any) -> Any:
        raise NotImplementedError("Precision weighting not implemented.")
