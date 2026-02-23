from __future__ import annotations

from typing import Any

from core.layers.interoceptive.VitalStateMonitor import VitalStateMonitor


class AllostaticController:
    def predict_needs(self, state: Any) -> Any:
        raise NotImplementedError("Allostatic prediction not implemented.")


class ArousalValenceSystem:
    def compute(self, state: Any) -> Any:
        raise NotImplementedError("Arousal/valence computation not implemented.")


__all__ = ["VitalStateMonitor", "AllostaticController", "ArousalValenceSystem"]
