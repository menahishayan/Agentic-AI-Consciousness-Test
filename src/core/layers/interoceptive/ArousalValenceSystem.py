from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional

import numpy as np

from core.coordination.messages import AgentMessage


@dataclass
class HomeostaticState:
    health: float
    hunger: float
    resource_level: float
    threat_proximity: float
    oxygen: float
    tick: int


@dataclass
class PredictionError:
    magnitude: float
    source: str
    tick: int


@dataclass
class PolicyBias:
    survival_weight: float
    exploration_weight: float
    risk_tolerance: float


@dataclass
class ArousalValenceState:
    arousal: float
    valence: float
    urgency_signal: float
    policy_bias: PolicyBias
    learning_rate_mod: float
    tick: int


@dataclass
class ArousalValenceConfig:
    w_health: float = 0.35
    w_hunger: float = 0.25
    w_threat: float = 0.30
    w_pred_err: float = 0.10
    v_health: float = 0.30
    v_hunger: float = 0.30
    v_resources: float = 0.20
    v_oxygen: float = 0.20
    decay_rate: float = 0.95
    resting_arousal: float = 0.10
    urgency_broadcast_threshold: float = 0.65


class ArousalValenceSystem:
    def __init__(
        self,
        config: Optional[Mapping[str, Any] | ArousalValenceConfig] = None,
        message_bus: Any = None,
        self_state_tracker: Any = None,
    ) -> None:
        self.config = self._normalize_config(config)
        self.message_bus = message_bus
        self.self_state_tracker = self_state_tracker
        self._last_tick: Optional[int] = None
        self._current_state = self._build_state(
            arousal=self.config.resting_arousal,
            valence=0.0,
            tick=0,
        )

    def update(
        self,
        homeostatic_state: HomeostaticState,
        prediction_error: Optional[PredictionError] = None,
    ) -> ArousalValenceState:
        tick = max(0, int(homeostatic_state.tick))
        health = self._clip01(homeostatic_state.health)
        hunger = self._clip01(homeostatic_state.hunger)
        resource_level = self._clip01(homeostatic_state.resource_level)
        threat_proximity = self._clip01(homeostatic_state.threat_proximity)
        oxygen = self._clip01(homeostatic_state.oxygen)
        pred_err_magnitude = (
            0.0 if prediction_error is None else self._clip01(prediction_error.magnitude)
        )

        raw_arousal = self._clip01(
            self.config.w_health * (1.0 - health)
            + self.config.w_hunger * (1.0 - hunger)
            + self.config.w_threat * threat_proximity
            + self.config.w_pred_err * pred_err_magnitude
        )

        valence = self._clip(
            (
                health * self.config.v_health
                + hunger * self.config.v_hunger
                + resource_level * self.config.v_resources
                + oxygen * self.config.v_oxygen
            )
            * 2.0
            - 1.0,
            -1.0,
            1.0,
        )

        # Keep arousal continuous by never dropping below the decayed prior state.
        if self._last_tick is None:
            arousal = raw_arousal
        else:
            dt = max(0, tick - self._last_tick)
            decayed_prior = (
                self._current_state.arousal
                if dt == 0
                else self._decay_value(self._current_state.arousal, dt=dt)
            )
            arousal = self._clip01(max(raw_arousal, decayed_prior))

        self._current_state = self._build_state(arousal=arousal, valence=valence, tick=tick)
        self._last_tick = tick
        self._publish(self._current_state)
        self._record_high_arousal_if_needed(self._current_state)
        return self.get_current_state()

    def apply_decay(self, dt: int) -> None:
        if dt <= 0:
            return

        tick = self._current_state.tick + int(dt)
        decayed_arousal = self._decay_value(self._current_state.arousal, dt=int(dt))
        self._current_state = self._build_state(
            arousal=decayed_arousal,
            valence=self._current_state.valence,
            tick=tick,
        )
        self._last_tick = tick

    def get_current_state(self) -> ArousalValenceState:
        return ArousalValenceState(
            arousal=float(self._current_state.arousal),
            valence=float(self._current_state.valence),
            urgency_signal=float(self._current_state.urgency_signal),
            policy_bias=PolicyBias(
                survival_weight=float(self._current_state.policy_bias.survival_weight),
                exploration_weight=float(self._current_state.policy_bias.exploration_weight),
                risk_tolerance=float(self._current_state.policy_bias.risk_tolerance),
            ),
            learning_rate_mod=float(self._current_state.learning_rate_mod),
            tick=int(self._current_state.tick),
        )

    def reset(self) -> None:
        self._current_state = self._build_state(
            arousal=self.config.resting_arousal,
            valence=0.0,
            tick=0,
        )
        self._last_tick = None

    def _build_state(self, arousal: float, valence: float, tick: int) -> ArousalValenceState:
        arousal = self._clip01(arousal)
        valence = self._clip(valence, -1.0, 1.0)

        survival_weight = self._clip01(arousal * max(0.0, -valence) + 0.1)
        exploration_weight = 1.0 - survival_weight
        risk_tolerance = self._clip01((1.0 - arousal) * 0.8)
        learning_rate_mod = self._clip(0.5 + (arousal * 1.5), 0.5, 2.0)
        urgency_signal = self._clip01(arousal * (0.5 + 0.5 * max(0.0, -valence)))

        return ArousalValenceState(
            arousal=arousal,
            valence=valence,
            urgency_signal=urgency_signal,
            policy_bias=PolicyBias(
                survival_weight=survival_weight,
                exploration_weight=exploration_weight,
                risk_tolerance=risk_tolerance,
            ),
            learning_rate_mod=learning_rate_mod,
            tick=int(tick),
        )

    def _decay_value(self, arousal: float, dt: int) -> float:
        return self._clip01(
            self.config.resting_arousal
            + (arousal - self.config.resting_arousal) * float(self.config.decay_rate ** dt)
        )

    def _publish(self, state: ArousalValenceState) -> None:
        if self.message_bus is None:
            return
        publish = getattr(self.message_bus, "publish", None)
        if not callable(publish):
            return

        payload = asdict(state)
        message = AgentMessage(
            sender="interoceptive",
            kind="homeostatic.arousal_valence",
            payload=payload,
        )
        try:
            publish(message)
            return
        except TypeError:
            pass

        try:
            publish("homeostatic.arousal_valence", payload)
        except TypeError:
            publish(topic="homeostatic.arousal_valence", payload=payload)

    def _record_high_arousal_if_needed(self, state: ArousalValenceState) -> None:
        if state.arousal <= 0.7 or self.self_state_tracker is None:
            return
        record = getattr(self.self_state_tracker, "record", None)
        if not callable(record):
            return
        record(
            {
                "tick": state.tick,
                "phase": "arousal_valence_alert",
                "topic": "homeostatic.arousal_valence",
                "arousal_valence": asdict(state),
            }
        )

    @staticmethod
    def _normalize_config(
        config: Optional[Mapping[str, Any] | ArousalValenceConfig],
    ) -> ArousalValenceConfig:
        if isinstance(config, ArousalValenceConfig):
            return config
        if not isinstance(config, Mapping):
            return ArousalValenceConfig()

        base = ArousalValenceConfig()
        values = {}
        for key in ArousalValenceConfig.__dataclass_fields__.keys():
            value = config.get(key, getattr(base, key))
            if key == "urgency_broadcast_threshold":
                values[key] = float(
                    ArousalValenceSystem._clip(
                        ArousalValenceSystem._as_float(value, base.urgency_broadcast_threshold),
                        0.0,
                        1.0,
                    )
                )
            elif key == "resting_arousal":
                values[key] = float(
                    ArousalValenceSystem._clip(
                        ArousalValenceSystem._as_float(value, base.resting_arousal),
                        0.0,
                        1.0,
                    )
                )
            elif key == "decay_rate":
                values[key] = float(
                    ArousalValenceSystem._clip(
                        ArousalValenceSystem._as_float(value, base.decay_rate),
                        0.0,
                        1.0,
                    )
                )
            else:
                values[key] = ArousalValenceSystem._as_float(value, getattr(base, key))
        return ArousalValenceConfig(**values)

    @staticmethod
    def _clip01(value: Any) -> float:
        return float(np.clip(float(value), 0.0, 1.0))

    @staticmethod
    def _clip(value: Any, lo: float, hi: float) -> float:
        return float(np.clip(float(value), lo, hi))

    @staticmethod
    def _as_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)
