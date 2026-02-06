from __future__ import annotations

from typing import Any

from core.agents.homeostatic_agent import HomeostaticAgent
from core.agents.metacognitive_agent import MetacognitiveAgent
from core.agents.motor_agent import MotorAgent
from core.agents.perceptual_agent import PerceptualAgent


class MultiAgentCoordinator:
    def __init__(
        self,
        homeostatic: HomeostaticAgent,
        perceptual: PerceptualAgent,
        motor: MotorAgent,
        metacognitive: MetacognitiveAgent,
    ) -> None:
        self.homeostatic = homeostatic
        self.perceptual = perceptual
        self.motor = motor
        self.metacognitive = metacognitive

    def step(self, observation: Any, memory_manager: Any, workspace: Any) -> Any:
        raise NotImplementedError("Coordinator step not implemented.")
