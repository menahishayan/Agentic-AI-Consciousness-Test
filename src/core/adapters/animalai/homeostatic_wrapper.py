"""
HomeostaticWrapper — simulates physiological depletion on top of Animal AI rewards.

This is the boundary between raw game mechanics and the homeostatic state model.
The brain never sees this class; it sees only normalized AgentState values.

Design:
  - Health and saturation deplete each step at configurable rates
  - Positive Animal AI reward = food consumed → restores saturation and health
  - Negative Animal AI reward = hazard hit → penalizes health
  - Energy is a composite of saturation and health

All rates are configurable via config.json adapter_config.homeostatic.
"""
from __future__ import annotations

from typing import Any, Dict


class HomeostaticWrapper:
    """
    Maintains simulated physiological variables that deplete over time.

    Conceptually, this is a layer between the raw reward signal and the
    structured homeostatic state. It converts Animal AI's scalar reward
    into meaningful physiological changes.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        hc = config.get("homeostatic", {})

        # Depletion rates (per step)
        self.health_depletion_rate: float = float(hc.get("health_depletion_rate", 0.001))
        self.saturation_depletion_rate: float = float(hc.get("saturation_depletion_rate", 0.002))

        # Restoration on positive reward (food)
        self.food_health_restore: float = float(hc.get("food_health_restore", 0.3))
        self.food_saturation_restore: float = float(hc.get("food_saturation_restore", 0.5))

        # Penalty on negative reward (hazard / bad event)
        self.hazard_health_penalty: float = float(hc.get("hazard_health_penalty", 0.2))

        # Death threshold
        self.death_threshold: float = float(hc.get("death_threshold", 0.0))

        # Internal state
        self._health: float = 1.0
        self._saturation: float = 1.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Restore full physiological state at episode start."""
        self._health = 1.0
        self._saturation = 1.0

    def step(self, raw_reward: float, env_done: bool) -> None:
        """
        Apply one timestep of physiology given the raw Animal AI reward.

        raw_reward > 0  → food consumed, restore saturation and health
        raw_reward < 0  → hazard hit, penalize health
        raw_reward == 0 → normal step, apply passive depletion only
        """
        # Passive depletion always applies
        self._health = max(0.0, self._health - self.health_depletion_rate)
        self._saturation = max(0.0, self._saturation - self.saturation_depletion_rate)

        if raw_reward > 0.0:
            # Food consumed — scale restoration by reward magnitude (capped)
            restore_scale = min(raw_reward, 1.0)
            self._saturation = min(1.0, self._saturation + restore_scale * self.food_saturation_restore)
            self._health = min(1.0, self._health + restore_scale * self.food_health_restore)
        elif raw_reward < 0.0:
            # Hazard / penalty — health drops proportional to negative reward
            penalty_scale = min(abs(raw_reward), 1.0)
            self._health = max(0.0, self._health - penalty_scale * self.hazard_health_penalty)

    def get_state(self) -> Dict[str, float]:
        """
        Return normalized [0, 1] homeostatic values.
        energy is a composite: saturation drives most of it, health floors it.
        """
        energy = self._saturation * 0.7 + self._health * 0.3
        return {
            "health": round(self._health, 4),
            "saturation": round(self._saturation, 4),
            "energy": round(energy, 4),
        }

    @property
    def is_alive(self) -> bool:
        return self._health > self.death_threshold

    @property
    def health(self) -> float:
        return self._health

    @property
    def saturation(self) -> float:
        return self._saturation
