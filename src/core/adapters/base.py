from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from core.models.signals import DriveChannel
from core.models.state import AgentState


class AbstractEnvironmentAdapter(ABC):
    """
    The sole interface between the environment and the brain.

    Brain layers (layers/, memory/, llm/, runtime/) import ONLY this class —
    never any concrete adapter. To swap the game, change adapter_folder in
    config.json; the brain is untouched.

    Contract:
      - reset() → initial AgentState
      - step(action_id) → (next AgentState, done)
      - close() → cleanup
      - get_available_vitals() → channel names exposed
      - get_available_policies() → list of policy descriptor dicts
      - get_drive_channels() → task-specific DriveChannel definitions
      - get_task_goal() → {description, priority, task_id}
      - estimate_* methods → scalar [0,1] derived features for AgentState

    All estimate_* methods receive the current AgentState so they can compute
    derived features without needing to re-query the environment.
    """

    @abstractmethod
    def reset(self) -> AgentState:
        """Reset environment and return initial state."""

    @abstractmethod
    def step(self, action_id: str) -> Tuple[AgentState, bool]:
        """
        Execute action_id and return (next_state, done).
        done=True when the episode ends (death, timeout, success).
        """

    @abstractmethod
    def close(self) -> None:
        """Release environment resources."""

    @abstractmethod
    def get_available_vitals(self) -> List[str]:
        """
        Return list of homeostatic channel names this adapter provides.
        Example: ["health", "saturation", "energy"]
        """

    @abstractmethod
    def get_available_policies(self) -> List[Dict[str, Any]]:
        """
        Return list of policy descriptor dicts. Each must contain:
          policy_id: str       — unique identifier
          callable_name: str   — name used internally
          tags: List[str]      — semantic category tags (e.g. ["navigation"])
          drive_tags: List[str] — drive channel ids this policy addresses
        """

    @abstractmethod
    def get_drive_channels(self) -> List[DriveChannel]:
        """
        Return task-specific DriveChannel definitions.
        These are owned by the adapter, NOT hardcoded in brain layers.
        AgentLoop reads these at init and injects them into AllostaticController.
        """

    @abstractmethod
    def get_task_goal(self) -> Dict[str, Any]:
        """
        Return the top-level task goal as a dict:
          description: str   — natural language for LLM and GoalCoherenceChecker
          priority: float    — [0,1] importance
          task_id: str       — machine-readable identifier
        """

    @abstractmethod
    def estimate_resource_level(self, state: AgentState) -> float:
        """Return [0,1] estimate of available resources."""

    @abstractmethod
    def estimate_threat_proximity(self, state: AgentState) -> float:
        """Return [0,1] threat proximity (1=immediate danger)."""

    @abstractmethod
    def build_area_id(self, state: AgentState) -> str:
        """Return a stable string identifier for the agent's current spatial region."""

    @abstractmethod
    def estimate_entity_density(self, state: AgentState) -> float:
        """Return [0,1] density of notable entities in the vicinity."""

    @abstractmethod
    def estimate_terrain_novelty(self, state: AgentState) -> float:
        """Return [0,1] how novel this area is (0=very familiar, 1=never seen)."""
