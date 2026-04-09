"""
MetacognitiveMonitor — Layer 4, Global Workspace

Integrates signals from all layers via GlobalWorkspace broadcast and
publishes urgency/uncertainty assessments. This is the "global broadcast"
in Seth's global workspace theory — the layer that integrates disparate
information streams into a unified, coherent signal.

Functions:
  1. Uncertainty estimation — how confident is the agent overall?
  2. Goal drift detection — are we still on track?
  3. Urgency broadcast — alerts all layers when critical state is reached
  4. Context assembly — builds the context dict for PolicyGenerator

No game-specific logic. No adapter imports.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from core.coordination.messages import AgentMessage
from core.coordination.workspace import GlobalWorkspace
from core.models.signals import (
    ArousalValence,
    DriveSignalBatch,
    Goal,
    PredictionErrorBatch,
)

log = logging.getLogger(__name__)


class MetacognitiveMonitor:
    """
    Global workspace integrator. Reads all messages from the workspace,
    assesses confidence and urgency, and publishes a metacognitive summary.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self._high_uncertainty_threshold = 0.7
        self._critical_health_threshold = 0.15
        self._prev_goals: List[Goal] = []

    def update(
        self,
        workspace: GlobalWorkspace,
        goals: List[Goal],
        step: int,
    ) -> Dict[str, Any]:
        """
        Integrate workspace signals and publish metacognitive assessment.

        Returns:
            Context dict consumed by PolicyGenerator
        """
        messages = workspace.broadcast()

        # Extract latest signals from workspace
        drive_batch = self._latest_payload(messages, "drive_signal", DriveSignalBatch)
        pe_batch = self._latest_payload(messages, "prediction_error", PredictionErrorBatch)
        av = self._latest_payload(messages, "arousal_valence", ArousalValence)
        vitals = self._latest_payload(messages, "vital_state", dict)

        # Compute integrated uncertainty
        uncertainty = self._estimate_uncertainty(drive_batch, pe_batch, av)

        # Detect critical states
        is_critical = self._check_critical(drive_batch, vitals)

        # Detect goal drift
        goal_changed = self._detect_goal_change(goals)
        self._prev_goals = list(goals)

        assessment = {
            "uncertainty": uncertainty,
            "is_critical": is_critical,
            "goal_changed": goal_changed,
            "max_drive_urgency": drive_batch.max_urgency if drive_batch else 0.0,
            "mean_pe": pe_batch.mean_magnitude if pe_batch else 0.0,
            "arousal": av.arousal if av else 0.0,
            "valence": av.valence if av else 0.0,
        }

        workspace.publish(AgentMessage(
            sender="MetacognitiveMonitor",
            kind="metacognitive",
            payload=assessment,
            step=step,
            priority=uncertainty if is_critical else 0.0,
        ))

        if is_critical:
            log.warning("Step %d: CRITICAL state detected (urgency=%.2f)", step,
                        drive_batch.max_urgency if drive_batch else 0.0)

        # Assemble context for PolicyGenerator
        context = {
            "drive_batch": drive_batch,
            "pe_batch": pe_batch,
            "arousal_valence": av,
            "vitals": vitals or {},
            "uncertainty": uncertainty,
            "is_critical": is_critical,
            "step": step,
        }
        return context

    def _estimate_uncertainty(
        self,
        drive_batch: Optional[DriveSignalBatch],
        pe_batch: Optional[PredictionErrorBatch],
        av: Optional[ArousalValence],
    ) -> float:
        components = []
        if drive_batch:
            components.append(drive_batch.max_urgency)
        if pe_batch:
            components.append(min(pe_batch.mean_magnitude, 1.0))
        if av:
            components.append(av.arousal)
        if not components:
            return 0.5
        return float(sum(components) / len(components))

    def _check_critical(
        self,
        drive_batch: Optional[DriveSignalBatch],
        vitals: Optional[Dict],
    ) -> bool:
        if drive_batch and drive_batch.max_urgency >= 0.9:
            return True
        if vitals:
            health = vitals.get("health", 1.0)
            if health is not None and health < self._critical_health_threshold:
                return True
        return False

    def _detect_goal_change(self, goals: List[Goal]) -> bool:
        prev_ids = {g.goal_id for g in self._prev_goals}
        curr_ids = {g.goal_id for g in goals}
        return prev_ids != curr_ids

    def _latest_payload(
        self,
        messages: List[AgentMessage],
        kind: str,
        expected_type: Any,
    ) -> Optional[Any]:
        matches = [m for m in messages if m.kind == kind]
        if not matches:
            return None
        payload = matches[-1].payload
        if isinstance(expected_type, type) and not isinstance(payload, expected_type):
            return None
        return payload
