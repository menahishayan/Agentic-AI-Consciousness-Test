from __future__ import annotations

from typing import Any

from core.models.signals import PredictionError


class PerceptualAgent:
    def __init__(self) -> None:
        pass

    def step(self, observation: Any, memory_manager: Any) -> PredictionError:
        raise NotImplementedError("Perceptual agent step not implemented.")
