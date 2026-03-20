from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import cmp_to_key
from math import isinf
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from core.coordination.messages import AgentMessage


@dataclass
class DriveChannel:
    id: str
    setpoint: float
    critical_threshold: float
    irreversible: bool
    recovery_cost_ticks: int
    suggested_action_tag: str = ""


@dataclass
class HomeostaticState:
    values: Dict[str, float]
    tick: int
    context_hash: str = "default"


@dataclass
class HomeostaticHistory:
    snapshots: List[HomeostaticState]
    channels: List[DriveChannel]
    tick: int


@dataclass
class DriveSignal:
    channel_id: str
    current_value: float
    projected_value: float
    ticks_to_critical: float
    urgency: float
    projection_confidence: float
    suggested_action_tag: str
    tick: int


@dataclass
class PrioritisedDriveSignals:
    signals: List[DriveSignal]
    highest_urgency: float
    tick: int


@dataclass
class AllostaticConfig:
    planning_horizon: int = 50
    history_window: int = 20
    irreversibility_bonus: float = 0.3
    recovery_weight_factor: float = 0.2
    urgency_tie_epsilon: float = 0.05
    threat_prior_weight: float = 0.3
    min_confidence: float = 0.5


class AllostaticController:
    def __init__(
        self,
        config: AllostaticConfig,
        channels: List[DriveChannel],
        self_state_memory: Optional[Any] = None,
        pe_history: Optional[Any] = None,
        policy_traces: Optional[Any] = None,
        memory_manager: Optional[Any] = None,
        message_bus: Any = None,
    ) -> None:
        self.config = self._normalize_config(config)
        self.channels = list(channels)
        self._channel_map = {channel.id: channel for channel in self.channels}
        self.memory_manager = memory_manager
        self.self_state_memory = (
            self_state_memory if self_state_memory is not None else memory_manager
        )
        self.pe_history = pe_history if pe_history is not None else memory_manager
        self.policy_traces = policy_traces if policy_traces is not None else memory_manager
        self.message_bus = message_bus
        self._last_output: Optional[PrioritisedDriveSignals] = None
        self._last_tick: int = 0

    def update(
        self,
        history: HomeostaticHistory,
        area_id: str = "unknown",
    ) -> PrioritisedDriveSignals:
        snapshots = list(history.snapshots)
        if not snapshots:
            out = PrioritisedDriveSignals(signals=[], highest_urgency=0.0, tick=int(history.tick))
            self._last_output = out
            self._publish(out)
            return out

        newest = snapshots[0]
        self._last_tick = int(history.tick)
        context_hash = self._resolve_context_hash(newest, area_id)
        active_channels = history.channels if history.channels else self.channels
        max_recovery_cost = max((max(0, int(c.recovery_cost_ticks)) for c in active_channels), default=1)

        signals: List[DriveSignal] = []
        for channel in active_channels:
            current = self._current_value(newest, channel.id)
            if current is None:
                continue

            values_oldest_first = self._values_for_channel(snapshots, channel.id)
            raw_rate = self._rate_of_change(values_oldest_first)
            corrected_rate, confidence = self._memory_corrected_rate(
                channel_id=channel.id,
                state_snapshot=newest,
                context_hash=context_hash,
                raw_rate=raw_rate,
                snapshot_count=len(snapshots),
            )

            ticks_to_critical = self._ticks_to_critical(
                current=current,
                rate=corrected_rate,
                threshold=channel.critical_threshold,
            )
            ticks_to_critical = self._apply_threat_prior(
                ticks_to_critical=ticks_to_critical,
                channel=channel,
                area_id=area_id,
            )

            if isinf(ticks_to_critical) or ticks_to_critical > float(self.config.planning_horizon):
                continue

            urgency = self._compute_urgency(
                ticks_to_critical=ticks_to_critical,
                channel=channel,
                confidence=confidence,
                max_recovery_cost=max_recovery_cost,
            )
            projected_value = self._clip01(current + corrected_rate * ticks_to_critical)

            signals.append(
                DriveSignal(
                    channel_id=channel.id,
                    current_value=current,
                    projected_value=projected_value,
                    ticks_to_critical=float(ticks_to_critical),
                    urgency=urgency,
                    projection_confidence=confidence,
                    suggested_action_tag=str(getattr(channel, "suggested_action_tag", "")),
                    tick=int(history.tick),
                )
            )

        signals.sort(key=cmp_to_key(self._compare_signals))
        highest = float(signals[0].urgency) if signals else 0.0
        out = PrioritisedDriveSignals(signals=signals, highest_urgency=highest, tick=int(history.tick))
        self._last_output = out
        self._publish(out)
        return out

    def reset(self) -> None:
        self._last_output = None

    def _compare_signals(self, left: DriveSignal, right: DriveSignal) -> int:
        delta = float(left.urgency) - float(right.urgency)
        if abs(delta) > self.config.urgency_tie_epsilon:
            return -1 if delta > 0.0 else 1

        conflict_score = self._conflict_resolution_score(left, right)
        if conflict_score > 0.0:
            return -1
        if conflict_score < 0.0:
            return 1

        left_channel = self._channel_map.get(left.channel_id)
        right_channel = self._channel_map.get(right.channel_id)
        left_cost = 0 if left_channel is None else max(0, int(left_channel.recovery_cost_ticks))
        right_cost = 0 if right_channel is None else max(0, int(right_channel.recovery_cost_ticks))
        if left_cost != right_cost:
            return -1 if left_cost > right_cost else 1

        if left.channel_id == right.channel_id:
            return 0
        return -1 if left.channel_id < right.channel_id else 1

    def _memory_corrected_rate(
        self,
        channel_id: str,
        state_snapshot: HomeostaticState,
        context_hash: str,
        raw_rate: float,
        snapshot_count: int,
    ) -> tuple[float, float]:
        getter = getattr(self.memory_manager, "get_depletion_rate", None)
        if callable(getter):
            try:
                historical_rate = getter(state_snapshot, channel_id)
            except TypeError:
                historical_rate = None
            except Exception:
                historical_rate = None
            if isinstance(historical_rate, (int, float)):
                corrected = 0.6 * raw_rate + 0.4 * float(historical_rate)
                confidence = max(self.config.min_confidence, 0.75)
                return corrected, self._clip(confidence, self.config.min_confidence, 1.0)

        getter = getattr(self.self_state_memory, "get_depletion_rate", None)
        if callable(getter):
            try:
                historical_rate = getter(channel_id, context_hash)
            except Exception:
                historical_rate = None
            if isinstance(historical_rate, (int, float)):
                corrected = 0.6 * raw_rate + 0.4 * float(historical_rate)
                confidence = max(self.config.min_confidence, 0.75)
                return corrected, self._clip(confidence, self.config.min_confidence, 1.0)

        heuristic_confidence = float(snapshot_count) / float(max(1, self.config.history_window))
        confidence = self._clip(heuristic_confidence, self.config.min_confidence, 1.0)
        return raw_rate, confidence

    def _apply_threat_prior(
        self,
        ticks_to_critical: float,
        channel: DriveChannel,
        area_id: str,
    ) -> float:
        if isinf(ticks_to_critical) or not channel.irreversible:
            return ticks_to_critical

        getter = getattr(self.memory_manager, "get_area_threat_prior", None)
        if not callable(getter):
            getter = getattr(self.pe_history, "get_area_threat_prior", None)
        if not callable(getter):
            return ticks_to_critical

        try:
            threat_prior = float(getter(str(area_id)))
        except Exception:
            return ticks_to_critical

        threat_prior = self._clip01(threat_prior)
        multiplier = 1.0 - self.config.threat_prior_weight * threat_prior
        multiplier = max(0.5, multiplier)
        return float(ticks_to_critical) * float(multiplier)

    def _compute_urgency(
        self,
        ticks_to_critical: float,
        channel: DriveChannel,
        confidence: float,
        max_recovery_cost: int,
    ) -> float:
        if isinf(ticks_to_critical):
            return 0.0

        time_pressure = 1.0 - self._clip(
            float(ticks_to_critical) / float(max(1, self.config.planning_horizon)),
            0.0,
            1.0,
        )
        irreversibility_bonus = self.config.irreversibility_bonus if channel.irreversible else 0.0
        recovery_weight = self._clip(
            float(max(0, int(channel.recovery_cost_ticks))) / float(max(max_recovery_cost, 1)),
            0.0,
            1.0,
        )
        raw_urgency = (
            time_pressure
            + irreversibility_bonus
            + self.config.recovery_weight_factor * recovery_weight
        )
        return self._clip(raw_urgency * self._clip(confidence, 0.0, 1.0), 0.0, 1.0)

    @staticmethod
    def _ticks_to_critical(current: float, rate: float, threshold: float) -> float:
        if rate >= 0.0:
            return float("inf")
        gap = float(current) - float(threshold)
        if gap <= 0.0:
            return 0.0
        return float(gap / abs(rate))

    @staticmethod
    def _rate_of_change(values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values), dtype=float)
        slope = np.polyfit(x, np.asarray(values, dtype=float), 1)[0]
        return float(slope)

    def _values_for_channel(self, snapshots_newest_first: List[HomeostaticState], channel_id: str) -> List[float]:
        values: List[float] = []
        for snapshot in reversed(snapshots_newest_first):
            value = self._current_value(snapshot, channel_id)
            if value is not None:
                values.append(value)
        if not values:
            return [0.0]
        return values

    @staticmethod
    def _current_value(snapshot: HomeostaticState, channel_id: str) -> Optional[float]:
        values = getattr(snapshot, "values", None)
        if not isinstance(values, Mapping):
            return None
        raw = values.get(channel_id)
        if not isinstance(raw, (int, float)):
            return None
        return float(np.clip(raw, 0.0, 1.0))

    @staticmethod
    def _resolve_context_hash(snapshot: HomeostaticState, area_id: str) -> str:
        raw = getattr(snapshot, "context_hash", None)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
        if isinstance(area_id, str) and area_id.strip():
            return area_id.strip()
        return "default"

    def _conflict_resolution_score(self, left: DriveSignal, right: DriveSignal) -> float:
        channel_id_a = left.channel_id
        channel_id_b = right.channel_id
        context_vector = self._build_conflict_context_vector(left, right)

        getter = getattr(self.memory_manager, "get_conflict_resolution_score", None)
        score: Any = None
        if callable(getter):
            try:
                score = getter(channel_id_a, channel_id_b, context_vector)
            except TypeError:
                score = getter(channel_id_a, channel_id_b)
            except Exception:
                score = None

        if score is None:
            getter = getattr(self.policy_traces, "get_conflict_resolution_score", None)
            if not callable(getter):
                return 0.0
            try:
                score = getter(channel_id_a, channel_id_b)
            except Exception:
                return 0.0

        if not isinstance(score, (int, float)):
            return 0.0
        return float(score)

    def _build_conflict_context_vector(self, left: DriveSignal, right: DriveSignal) -> np.ndarray:
        left_channel = self._channel_map.get(left.channel_id)
        right_channel = self._channel_map.get(right.channel_id)
        survival_weight = 1.0 if (
            bool(getattr(left_channel, "irreversible", False))
            or bool(getattr(right_channel, "irreversible", False))
        ) else 0.5
        tick_norm = self._clip01(
            float(self._last_tick % max(1, self.config.planning_horizon))
            / float(max(1, self.config.planning_horizon))
        )
        return np.asarray(
            [
                self._clip01(left.urgency),
                self._clip01(right.urgency),
                self._clip01(max(left.urgency, right.urgency)),
                self._clip(left.projected_value - right.projected_value, -1.0, 1.0),
                self._clip01(survival_weight),
                tick_norm,
            ],
            dtype=np.float32,
        )

    def _publish(self, output: PrioritisedDriveSignals) -> None:
        publish = getattr(self.message_bus, "publish", None)
        if not callable(publish):
            return

        payload = asdict(output)
        message = AgentMessage(
            sender="homeostatic",
            kind="homeostatic.drive_signals",
            payload=payload,
        )
        try:
            publish(message)
            return
        except TypeError:
            pass

        try:
            publish("homeostatic.drive_signals", payload)
        except TypeError:
            publish(topic="homeostatic.drive_signals", payload=payload)

    @staticmethod
    def _normalize_config(config: AllostaticConfig) -> AllostaticConfig:
        return AllostaticConfig(
            planning_horizon=max(1, int(getattr(config, "planning_horizon", 50))),
            history_window=max(1, int(getattr(config, "history_window", 20))),
            irreversibility_bonus=AllostaticController._clip(
                AllostaticController._as_float(getattr(config, "irreversibility_bonus", 0.3), 0.3),
                0.0,
                1.0,
            ),
            recovery_weight_factor=AllostaticController._clip(
                AllostaticController._as_float(getattr(config, "recovery_weight_factor", 0.2), 0.2),
                0.0,
                1.0,
            ),
            urgency_tie_epsilon=AllostaticController._clip(
                AllostaticController._as_float(getattr(config, "urgency_tie_epsilon", 0.05), 0.05),
                0.0,
                1.0,
            ),
            threat_prior_weight=AllostaticController._clip(
                AllostaticController._as_float(getattr(config, "threat_prior_weight", 0.3), 0.3),
                0.0,
                1.0,
            ),
            min_confidence=AllostaticController._clip(
                AllostaticController._as_float(getattr(config, "min_confidence", 0.5), 0.5),
                0.0,
                1.0,
            ),
        )

    @staticmethod
    def _clip01(value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))

    @staticmethod
    def _clip(value: float, lo: float, hi: float) -> float:
        return float(np.clip(value, lo, hi))

    @staticmethod
    def _as_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)
