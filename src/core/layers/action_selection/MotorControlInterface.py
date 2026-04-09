"""
MotorControlInterface — Layer 3, Action Selection

Translates a selected policy_id into an environment action by calling
the action_dispatcher callable. This is the only place where a policy
decision becomes a physical action in the environment.

Crucially: this class does NOT import the adapter. It receives an
action_dispatcher: Callable[[str], Tuple[AgentState, bool]] that is
created as a closure over the adapter in main.py.

This maintains the abstraction invariant: the brain never knows what
environment it's running in.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Optional, Tuple

from core.models.state import AgentState

log = logging.getLogger(__name__)


class MotorControlInterface:
    """
    Executes selected policies by dispatching to the environment via callback.

    The action_dispatcher receives a policy_id string and returns
    (next_state, done). This is injected at init — no adapter import needed.
    """

    def __init__(
        self,
        action_dispatcher: Callable[[str], Tuple[AgentState, bool]],
    ) -> None:
        self._dispatch = action_dispatcher
        self._last_policy_id: Optional[str] = None
        self._last_execution_ms: float = 0.0

    def execute(self, policy_id: str) -> Tuple[AgentState, bool]:
        """
        Execute the given policy by dispatching to the environment.

        Returns:
            (next_state, done): Next AgentState and whether episode ended
        """
        self._last_policy_id = policy_id
        t0 = time.monotonic()

        try:
            next_state, done = self._dispatch(policy_id)
        except Exception as exc:
            log.error("MotorControlInterface: dispatch failed for policy '%s': %s", policy_id, exc)
            raise

        self._last_execution_ms = (time.monotonic() - t0) * 1000.0
        return next_state, done

    @property
    def last_policy_id(self) -> Optional[str]:
        return self._last_policy_id

    @property
    def last_execution_ms(self) -> float:
        return self._last_execution_ms
