from __future__ import annotations

from typing import Any


class VitalStateMonitor:
    def update(self, state: Any) -> None:
        raise NotImplementedError("Vital state monitoring not implemented.")


class AllostaticController:
    def predict_needs(self, state: Any) -> Any:
        raise NotImplementedError("Allostatic prediction not implemented.")


class ArousalValenceSystem:
    def compute(self, state: Any) -> Any:
        raise NotImplementedError("Arousal/valence computation not implemented.")
