"""
AllostaticController — Layer 1, Interoceptive Foundation

Predictive allostatic regulation: anticipates homeostatic deficits before
they become critical, computing urgency per drive channel.

Key principles (from Seth's beast machine theory):
  - Allostasis = predicting and preparing for body-state changes
  - Urgency drives action selection bias (before actual deprivation)
  - ticks_to_critical is computed from EMA depletion rate history

No game-specific logic. No adapter imports.
DriveChannel definitions are injected at init from the adapter.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from core.coordination.messages import AgentMessage
from core.coordination.workspace import GlobalWorkspace
from core.models.signals import DriveChannel, DriveSignal, DriveSignalBatch


class AllostaticController:
    """
    Computes urgency for each drive channel and publishes DriveSignalBatch
    to GlobalWorkspace each step.
    """

    def __init__(
        self,
        drive_channels: List[DriveChannel],
        config: Dict[str, Any],
    ) -> None:
        self._channels: Dict[str, DriveChannel] = {ch.channel_id: ch for ch in drive_channels}
        cfg = config.get("allostatic_controller", {})
        self._planning_horizon: int = int(cfg.get("planning_horizon", 50))
        self._history_window: int = int(cfg.get("history_window", 20))
        self._urgency_tie_epsilon: float = float(cfg.get("urgency_tie_epsilon", 0.05))
        self._recovery_weight_factor: float = float(cfg.get("recovery_weight_factor", 0.2))
        self._threat_prior_weight: float = float(cfg.get("threat_prior_weight", 0.3))
        self._min_confidence: float = float(cfg.get("min_confidence", 0.5))

        # Rolling history of (step, value) for each channel
        self._history: Dict[str, deque] = {
            ch_id: deque(maxlen=self._history_window)
            for ch_id in self._channels
        }
        # External depletion rate provided by memory (optional)
        self._memory_depletion_rates: Dict[str, Optional[float]] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(
        self,
        vitals: Dict[str, Optional[float]],
        workspace: GlobalWorkspace,
        step: int,
        memory_depletion_rates: Optional[Dict[str, float]] = None,
    ) -> DriveSignalBatch:
        """
        Compute urgency for each channel and publish to workspace.

        Args:
            vitals: channel_id → current normalized value
            workspace: GlobalWorkspace for publishing
            step: Current episode step
            memory_depletion_rates: Optional rates from MemoryManager
        """
        if memory_depletion_rates:
            self._memory_depletion_rates.update(memory_depletion_rates)

        signals: List[DriveSignal] = []

        for ch_id, channel in self._channels.items():
            current = vitals.get(ch_id)
            if current is None:
                continue

            self._history[ch_id].append((step, current))
            urgency = self._compute_urgency(channel, current, ch_id)
            ticks = self._estimate_ticks_to_critical(channel, current, ch_id)

            signals.append(DriveSignal(
                channel_id=ch_id,
                current_value=current,
                setpoint=channel.setpoint,
                urgency=urgency,
                ticks_to_critical=ticks,
                suggested_action_tags=channel.suggested_action_tags,
            ))

        batch = DriveSignalBatch(signals=signals)

        workspace.publish(AgentMessage(
            sender="AllostaticController",
            kind="drive_signal",
            payload=batch,
            step=step,
            priority=batch.max_urgency,
        ))

        return batch

    def peek_max_urgency(self, vitals: Dict[str, Optional[float]]) -> float:
        """
        Compute max urgency for given vitals using the current model state.
        No side effects — does not update history or publish to workspace.
        Used to score the post-action state for outcome_score computation.
        """
        max_urgency = 0.0
        for ch_id, channel in self._channels.items():
            current = vitals.get(ch_id)
            if current is None:
                continue
            urgency = self._compute_urgency(channel, current, ch_id)
            max_urgency = max(max_urgency, urgency)
        return float(max_urgency)

    def inject_memory_depletion_rate(self, channel_id: str, rate: float) -> None:
        """Called by AgentLoop to inject memory-corrected depletion rates."""
        self._memory_depletion_rates[channel_id] = rate

    # ------------------------------------------------------------------
    # Internal computation
    # ------------------------------------------------------------------

    def _compute_urgency(
        self,
        channel: DriveChannel,
        current: float,
        ch_id: str,
    ) -> float:
        """
        Urgency is a function of:
          1. Distance below setpoint (normalized by setpoint)
          2. Proximity to critical threshold
          3. Predicted rate of decline (allostatic anticipation)

        Returns float in [0, 1].
        """
        # Component 1: setpoint deviation
        if current >= channel.setpoint:
            setpoint_urgency = 0.0
        else:
            setpoint_urgency = (channel.setpoint - current) / channel.setpoint

        # Component 2: threshold proximity
        if current <= channel.critical_threshold:
            threshold_urgency = 1.0
        elif current >= channel.setpoint:
            # Above setpoint — no urgency from threshold proximity.
            # Without this guard, below_setpoint = setpoint - current is negative
            # and leaks a negative threshold_urgency into the raw sum.
            threshold_urgency = 0.0
        else:
            range_above = channel.setpoint - channel.critical_threshold
            if range_above <= 0:
                threshold_urgency = 0.0
            else:
                below_setpoint = channel.setpoint - current  # positive: current < setpoint
                threshold_urgency = min(1.0, below_setpoint / range_above)

        # Component 3: predictive rate component
        rate = self._estimate_depletion_rate(ch_id)
        if rate is not None and rate > 0.0:
            # Ticks before hitting critical at current rate
            remaining = current - channel.critical_threshold
            if remaining > 0:
                ticks = remaining / rate
                # Urgency ramps up as horizon shrinks
                rate_urgency = max(0.0, 1.0 - ticks / self._planning_horizon)
            else:
                rate_urgency = 1.0
        else:
            rate_urgency = 0.0

        raw = (
            setpoint_urgency * 0.4
            + threshold_urgency * 0.4
            + rate_urgency * 0.2
        ) * channel.weight

        return float(min(1.0, raw))

    def _estimate_ticks_to_critical(
        self,
        channel: DriveChannel,
        current: float,
        ch_id: str,
    ) -> Optional[int]:
        rate = self._estimate_depletion_rate(ch_id)
        if rate is None or rate <= 0.0:
            return None
        remaining = current - channel.critical_threshold
        if remaining <= 0.0:
            return 0
        return int(remaining / rate)

    def _estimate_depletion_rate(self, ch_id: str) -> Optional[float]:
        """
        Estimate per-step depletion rate from history or memory.
        Returns positive float (rate of decrease) or None if unknown.
        """
        # Prefer memory-provided rate if available
        memory_rate = self._memory_depletion_rates.get(ch_id)
        if memory_rate is not None and memory_rate > 0.0:
            return memory_rate

        hist = self._history.get(ch_id)
        if not hist or len(hist) < 2:
            return None

        items = list(hist)
        # Compute average decline per step
        total_decline = 0.0
        n = 0
        for i in range(1, len(items)):
            step_diff = items[i][0] - items[i - 1][0]
            val_diff = items[i - 1][1] - items[i][1]  # positive = decline
            if step_diff > 0:
                total_decline += val_diff / step_diff
                n += 1

        if n == 0:
            return None
        avg = total_decline / n
        return max(0.0, avg)  # Only return positive (depletion) rates
