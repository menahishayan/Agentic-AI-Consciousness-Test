from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from core.coordination.messages import AgentMessage
from core.layers.predictive import WorldModelGenerator


SOURCE_MAP: Dict[str, str] = {
    "health": "proprioceptive",
    "hunger": "proprioceptive",
    "resource_level": "proprioceptive",
    "oxygen": "proprioceptive",
    "threat_proximity": "threat",
    "entity_density": "visual",
    "terrain_novelty": "visual",
}

CHANNELS = tuple(SOURCE_MAP.keys())


@dataclass
class ObservationSnapshot:
    health: float
    hunger: float
    resource_level: float
    oxygen: float
    threat_proximity: float
    entity_density: float
    terrain_novelty: float
    area_id: str
    last_action: str
    tick: int


@dataclass
class PredictionError:
    magnitude: float
    raw_magnitude: float
    precision: float
    source: str
    channel: str
    predicted: float
    observed: float
    tick: int


@dataclass
class PredictionErrorBatch:
    errors: List[PredictionError]
    aggregate_magnitude: float
    dominant_source: str
    tick: int


@dataclass
class PEConfig:
    alpha: float = 0.1
    epsilon: float = 0.01
    sigma_clip: float = 3.0
    default_precision: float = 0.5
    min_precision: float = 0.3
    world_model_alpha: float = 0.1
    action_confidence_threshold: int = 20


class PredictionErrorCalculator:
    def __init__(
        self,
        config: PEConfig,
        memory_manager: Optional[Any] = None,
        pe_history: Optional[Any] = None,
        message_bus: Any = None,
    ) -> None:
        self.config = self._normalize_config(config)
        self.memory_manager = memory_manager if memory_manager is not None else pe_history
        self.message_bus = message_bus

        self._baseline: Dict[str, float] = {}
        self._variance: Dict[str, float] = {}
        self._buffered_prediction: Optional[Dict[str, float]] = None
        self._world_model = WorldModelGenerator(
            channels=CHANNELS,
            alpha=self.config.world_model_alpha,
            min_precision=self.config.min_precision,
            confidence_threshold=self.config.action_confidence_threshold,
        )

    def update(
        self,
        observation: ObservationSnapshot,
        last_action: Optional[str] = None,
    ) -> PredictionErrorBatch:
        observed_channels = self._observation_channels(observation)
        if not self._baseline:
            self._initialize_baseline(observed_channels)

        if self._buffered_prediction is None:
            self._update_baseline(observed_channels)
            return PredictionErrorBatch(
                errors=[],
                aggregate_magnitude=0.0,
                dominant_source="",
                tick=int(observation.tick),
            )

        action_id = self._resolve_action_id(last_action, observation)
        area_precision = self._estimate_area_precision(observation.area_id)
        errors: List[PredictionError] = []
        for channel in CHANNELS:
            predicted = self._buffered_prediction.get(channel, observed_channels[channel])
            observed = observed_channels[channel]
            variance = max(self._variance.get(channel, self.config.epsilon**2), self.config.epsilon**2)
            std = max(math.sqrt(variance), self.config.epsilon)
            action_precision = self._estimate_action_precision(action_id, channel)
            precision = self._combine_precision(area_precision, action_precision)

            normalized = abs(observed - predicted) / std
            # sigma_clip=3.0 maps 3-sigma deviation to approximately 1.0.
            raw_pe = self._clip01(normalized / self.config.sigma_clip)
            weighted_pe = self._clip01(raw_pe * precision)

            errors.append(
                PredictionError(
                    magnitude=weighted_pe,
                    raw_magnitude=raw_pe,
                    precision=precision,
                    source=SOURCE_MAP[channel],
                    channel=channel,
                    predicted=predicted,
                    observed=observed,
                    tick=int(observation.tick),
                )
            )

        aggregate = self._clip01(float(np.mean([err.magnitude for err in errors])) if errors else 0.0)
        dominant_source = self._dominant_source(errors)
        batch = PredictionErrorBatch(
            errors=errors,
            aggregate_magnitude=aggregate,
            dominant_source=dominant_source,
            tick=int(observation.tick),
        )

        self._publish(batch)
        self._record_errors(observation.area_id, errors)

        self._update_baseline(observed_channels)
        self._buffered_prediction = None
        return batch

    def prepare_next_prediction(self, observation: ObservationSnapshot, action_id: Any) -> None:
        observed_channels = self._observation_channels(observation)
        if not self._baseline:
            self._initialize_baseline(observed_channels)
        resolved_action = self._resolve_action_id(action_id, observation)
        self._buffered_prediction = self._world_model.predict(observed_channels, resolved_action)

    def observe_transition(
        self,
        prev_observation: ObservationSnapshot,
        action_id: Any,
        next_observation: ObservationSnapshot,
    ) -> None:
        self._world_model.update(
            prev_observation=prev_observation,
            action_id=self._resolve_action_id(action_id, prev_observation),
            next_observation=next_observation,
        )

    def get_prediction(self) -> Dict[str, float]:
        if self._buffered_prediction is None:
            return {}
        return dict(self._buffered_prediction)

    def reset(self) -> None:
        self._baseline.clear()
        self._variance.clear()
        self._buffered_prediction = None
        self._world_model.reset()

    def get_variance(self) -> Dict[str, float]:
        return dict(self._variance)

    def _initialize_baseline(self, observed_channels: Mapping[str, float]) -> None:
        self._baseline = {channel: float(observed_channels[channel]) for channel in CHANNELS}
        self._variance = {channel: self.config.epsilon**2 for channel in CHANNELS}

    def _update_baseline(self, observed_channels: Mapping[str, float]) -> None:
        alpha = self.config.alpha
        inv_alpha = 1.0 - alpha
        for channel in CHANNELS:
            previous_mean = self._baseline.get(channel, observed_channels[channel])
            observed = observed_channels[channel]
            updated_mean = alpha * observed + inv_alpha * previous_mean
            previous_var = self._variance.get(channel, self.config.epsilon**2)
            updated_var = alpha * ((observed - updated_mean) ** 2) + inv_alpha * previous_var
            self._baseline[channel] = self._clip01(updated_mean)
            self._variance[channel] = max(updated_var, self.config.epsilon**2)

    def _estimate_area_precision(self, area_id: str) -> float:
        getter = getattr(self.memory_manager, "get_area_familiarity", None)
        if not callable(getter):
            return self.config.default_precision
        try:
            familiarity = float(getter(str(area_id)))
        except Exception:
            return self.config.default_precision

        familiarity = self._clip01(familiarity)
        precision = self.config.min_precision + familiarity * (1.0 - self.config.min_precision)
        return self._clip01(precision)

    def _estimate_action_precision(self, action_id: str, channel: str) -> float:
        return self._world_model.confidence(action_id, channel)

    def _combine_precision(self, area_precision: float, action_precision: float) -> float:
        area = self._clip(area_precision, self.config.min_precision, 1.0)
        action = self._clip(action_precision, self.config.min_precision, 1.0)
        combined = math.sqrt(area * action)
        return self._clip(combined, self.config.min_precision, 1.0)

    def _record_errors(self, area_id: str, errors: List[PredictionError]) -> None:
        if self.memory_manager is None:
            return

        for error in errors:
            record_pe = getattr(self.memory_manager, "record_pe", None)
            if callable(record_pe):
                try:
                    record_pe(str(area_id), error)
                    continue
                except Exception:
                    pass

            record = getattr(self.memory_manager, "record", None)
            if not callable(record):
                continue
            try:
                record(str(area_id), error)
                continue
            except TypeError:
                pass
            try:
                record({"area_id": str(area_id), "error": asdict(error), "magnitude": error.magnitude})
            except Exception:
                continue

    def _publish(self, batch: PredictionErrorBatch) -> None:
        publish = getattr(self.message_bus, "publish", None)
        if not callable(publish):
            return

        payload = asdict(batch)
        message = AgentMessage(
            sender="perceptual",
            kind="perceptual.prediction_error",
            payload=payload,
        )
        try:
            publish(message)
            return
        except TypeError:
            pass

        try:
            publish("perceptual.prediction_error", payload)
        except TypeError:
            publish(topic="perceptual.prediction_error", payload=payload)

    @staticmethod
    def _dominant_source(errors: List[PredictionError]) -> str:
        if not errors:
            return ""
        dominant = max(errors, key=lambda item: item.magnitude)
        return dominant.source

    @staticmethod
    def _observation_channels(observation: ObservationSnapshot) -> Dict[str, float]:
        return {
            "health": PredictionErrorCalculator._clip01(observation.health),
            "hunger": PredictionErrorCalculator._clip01(observation.hunger),
            "resource_level": PredictionErrorCalculator._clip01(observation.resource_level),
            "oxygen": PredictionErrorCalculator._clip01(observation.oxygen),
            "threat_proximity": PredictionErrorCalculator._clip01(observation.threat_proximity),
            "entity_density": PredictionErrorCalculator._clip01(observation.entity_density),
            "terrain_novelty": PredictionErrorCalculator._clip01(observation.terrain_novelty),
        }

    @staticmethod
    def _normalize_config(config: PEConfig) -> PEConfig:
        alpha = PredictionErrorCalculator._clip(
            PredictionErrorCalculator._as_float(getattr(config, "alpha", 0.1), 0.1),
            1e-6,
            1.0,
        )
        epsilon = PredictionErrorCalculator._clip(
            PredictionErrorCalculator._as_float(getattr(config, "epsilon", 0.01), 0.01),
            1e-6,
            1.0,
        )
        sigma_clip = PredictionErrorCalculator._clip(
            PredictionErrorCalculator._as_float(getattr(config, "sigma_clip", 3.0), 3.0),
            1e-6,
            10.0,
        )
        default_precision = PredictionErrorCalculator._clip01(
            PredictionErrorCalculator._as_float(getattr(config, "default_precision", 0.5), 0.5)
        )
        min_precision = PredictionErrorCalculator._clip01(
            PredictionErrorCalculator._as_float(getattr(config, "min_precision", 0.3), 0.3)
        )
        world_model_alpha = PredictionErrorCalculator._clip(
            PredictionErrorCalculator._as_float(getattr(config, "world_model_alpha", 0.1), 0.1),
            1e-6,
            1.0,
        )
        action_confidence_threshold = max(
            1,
            int(
                PredictionErrorCalculator._as_float(
                    getattr(config, "action_confidence_threshold", 20),
                    20.0,
                )
            ),
        )
        return PEConfig(
            alpha=alpha,
            epsilon=epsilon,
            sigma_clip=sigma_clip,
            default_precision=default_precision,
            min_precision=min_precision,
            world_model_alpha=world_model_alpha,
            action_confidence_threshold=action_confidence_threshold,
        )

    @staticmethod
    def _resolve_action_id(last_action: Any, observation: ObservationSnapshot) -> str:
        if isinstance(last_action, str) and last_action.strip():
            return last_action.strip()
        observed_last_action = getattr(observation, "last_action", None)
        if isinstance(observed_last_action, str) and observed_last_action.strip():
            return observed_last_action.strip()
        return "bootstrap"

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
