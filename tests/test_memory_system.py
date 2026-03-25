from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Mapping, Optional, Tuple

import numpy as np
import pytest

from core.memory import (
    MemoryConfig,
    MemoryManager,
    PolicyTraces,
    PredictionErrorHistory,
    SelfStateTracking,
    WorkingMemoryBuffer,
    WorkingMemoryEntry,
)


@dataclass
class _PredictionError:
    magnitude: float
    source: str
    tick: int


class _CaptureMemoryLogger:
    def __init__(self) -> None:
        self.events: List[Tuple[Mapping[str, Any], Optional[int]]] = []

    def memory_event(self, payload: Mapping[str, Any], step: Optional[int] = None) -> None:
        self.events.append((dict(payload), step))


def _state(
    *,
    tick: int,
    health: float = 0.6,
    hunger: float = 0.4,
    resource_level: float = 0.5,
    threat_proximity: float = 0.2,
    oxygen: float = 0.8,
    time_of_day: float = 0.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        tick=tick,
        health=health,
        hunger=hunger,
        resource_level=resource_level,
        threat_proximity=threat_proximity,
        oxygen=oxygen,
        time_of_day=time_of_day,
    )


def test_working_memory_buffer_recency_and_high_priority_goal_warning(caplog: pytest.LogCaptureFixture) -> None:
    buffer = WorkingMemoryBuffer(capacity=2)
    with caplog.at_level("WARNING"):
        buffer.record(
            WorkingMemoryEntry(
                tick=1,
                entry_type="goal",
                payload={"goal_id": "survive"},
                priority=0.95,
            )
        )
        buffer.record(
            WorkingMemoryEntry(
                tick=2,
                entry_type="observation",
                payload={"area_id": "a"},
                priority=0.2,
            )
        )
        buffer.record(
            WorkingMemoryEntry(
                tick=3,
                entry_type="prediction",
                payload={"pe": 0.3},
                priority=0.1,
            )
        )

    recent = buffer.get_recent(2)
    assert [entry.tick for entry in recent] == [3, 2]
    assert buffer.get_active_goals() == []
    assert "High-priority goal evicted" in caplog.text


def test_prediction_error_history_familiarity_threat_prior_and_variance() -> None:
    history = PredictionErrorHistory(min_observations=2, ema_alpha=0.5)

    history.record("forest", _PredictionError(magnitude=0.8, source="visual", tick=1))
    assert history.get_area_familiarity("forest") == pytest.approx(0.0, abs=1e-9)

    history.record("forest", _PredictionError(magnitude=0.6, source="visual", tick=2))
    familiarity = history.get_area_familiarity("forest")
    assert 0.0 < familiarity < 1.0

    assert history.get_area_threat_prior("forest") == pytest.approx(0.0, abs=1e-9)
    history.record("forest", _PredictionError(magnitude=0.9, source="threat", tick=3))
    assert history.get_area_threat_prior("forest") > 0.0

    stats = history._stats["forest"]  # noqa: SLF001
    assert stats.count == 3
    assert stats.pe_variance > 0.0


def test_self_state_tracking_similarity_retrieval() -> None:
    memory = SelfStateTracking(k_default=5, epsilon=1e-6)
    query_state = _state(tick=10)

    values = [-0.10, -0.20, -0.15, -0.05, -0.10]
    for index, value in enumerate(values):
        memory.record(
            state=_state(tick=index + 1),
            channel_deltas={"hunger": value},
            context_tags=["area:forest"],
            arousal=0.4 + 0.1 * index,
        )

    depletion_rate = memory.get_depletion_rate(query_state, "hunger", k=5)
    assert depletion_rate is not None
    assert depletion_rate == pytest.approx(sum(values) / len(values), abs=1e-3)

    capability = memory.get_capability_estimate(query_state, "area:forest", k=5)
    assert capability is not None
    assert capability == pytest.approx(0.6, abs=1e-3)

    missing_channel = memory.get_depletion_rate(query_state, "oxygen_drop", k=5)
    assert missing_channel is None


def test_policy_traces_conflict_resolution_and_best_action() -> None:
    traces = PolicyTraces(episode_length=100, k_default=5, epsilon=1e-6)
    base_context = np.asarray([0.8, 0.6, 0.7, -0.2, 1.0, 0.0], dtype=np.float32)

    traces.record(
        channel_a_id="hunger",
        channel_b_id="oxygen",
        winner_channel_id="hunger",
        action_tag="eat",
        context_vector=base_context,
        outcome_score=0.9,
        tick=10,
    )
    traces.record(
        channel_a_id="oxygen",
        channel_b_id="hunger",
        winner_channel_id="hunger",
        action_tag="eat",
        context_vector=base_context,
        outcome_score=0.8,
        tick=11,
    )
    traces.record(
        channel_a_id="hunger",
        channel_b_id="oxygen",
        winner_channel_id="oxygen",
        action_tag="surface",
        context_vector=base_context,
        outcome_score=0.5,
        tick=12,
    )

    score = traces.get_conflict_resolution_score("hunger", "oxygen", base_context, k=5)
    assert score > 0.0

    best_action = traces.get_best_action_for_drive("hunger", base_context, k=5)
    assert best_action == "eat"


def test_policy_traces_preserves_record_and_query_feature_space() -> None:
    traces = PolicyTraces(episode_length=100, k_default=5, epsilon=1e-6)
    context = np.asarray([0.85, 0.60, 0.85, -0.10, 1.0, 0.95], dtype=np.float32)
    traces.record(
        channel_a_id="health",
        channel_b_id="hunger",
        winner_channel_id="health",
        action_tag="heal",
        context_vector=context,
        outcome_score=1.0,
        tick=10,
    )

    assert traces._metadata  # noqa: SLF001
    record = next(iter(traces._metadata.values()))  # noqa: SLF001

    expected = context / np.linalg.norm(context)
    assert np.allclose(record.context_vector, expected.astype(np.float32), atol=1e-6)
    assert traces.get_conflict_resolution_score("health", "hunger", context, k=1) > 0.9


def test_memory_manager_delegates_and_lifecycle_clears_expected_scopes() -> None:
    manager = MemoryManager(
        MemoryConfig(
            working_memory_capacity=3,
            pe_min_observations=2,
            pe_ema_alpha=0.5,
            faiss_k_default=3,
            episode_length=200,
        )
    )

    manager.record_working(
        WorkingMemoryEntry(
            tick=1,
            entry_type="goal",
            payload={"goal_id": "maintain_oxygen"},
            priority=0.9,
        )
    )
    assert len(manager.get_recent(5)) == 1

    manager.record_pe("zone_a", _PredictionError(magnitude=0.7, source="visual", tick=1))
    manager.record_pe("zone_a", _PredictionError(magnitude=0.3, source="threat", tick=2))
    assert manager.get_area_familiarity("zone_a") > 0.0
    assert manager.get_area_threat_prior("zone_a") > 0.0

    state = _state(tick=1)
    manager.record_state(
        state=state,
        channel_deltas={"hunger": -0.1},
        context_tags=["area:zone_a"],
        arousal=0.75,
    )
    assert manager.get_depletion_rate(state, "hunger") is not None
    assert manager.get_capability_estimate(state, "area:zone_a") is not None

    context = np.asarray([0.9, 0.6, 0.8, -0.3, 1.0, 0.1], dtype=np.float32)
    manager.record_trace(
        channel_a_id="hunger",
        channel_b_id="oxygen",
        winner_channel_id="hunger",
        action_tag="eat",
        context_vector=context,
        outcome_score=0.9,
        tick=10,
    )
    assert manager.get_conflict_resolution_score("hunger", "oxygen", context) > 0.0
    assert manager.get_best_action_for_drive("hunger", context) == "eat"

    manager.clear_episode()
    assert manager.get_recent(5) == []
    # Long-term stores persist across episodes.
    assert manager.get_area_familiarity("zone_a") > 0.0
    assert manager.get_depletion_rate(state, "hunger") is not None

    manager.clear_all()
    assert manager.get_area_familiarity("zone_a") == pytest.approx(0.0, abs=1e-9)
    assert manager.get_depletion_rate(state, "hunger") is None
    assert manager.get_conflict_resolution_score("hunger", "oxygen", context) == pytest.approx(0.0, abs=1e-9)


def test_memory_manager_emits_memory_events_for_core_writes(tmp_path: Path) -> None:
    logger = _CaptureMemoryLogger()
    manager = MemoryManager(
        MemoryConfig(
            working_memory_capacity=3,
            pe_min_observations=2,
            pe_ema_alpha=0.5,
            faiss_k_default=3,
            episode_length=200,
        ),
        long_term_memory_config={"path": str(tmp_path / "ltm")},
        logger=logger,
    )

    manager.set_active_policy_for_pe("dummy:policy_test")
    manager.record_pe("zone_a", _PredictionError(magnitude=0.7, source="visual", tick=1))
    manager.record_state(
        state=_state(tick=1),
        channel_deltas={"hunger": -0.1},
        context_tags=["area:zone_a"],
        arousal=0.6,
    )
    manager.snapshot_self_state({"step": 1, "phase": "pre_action", "value": 1})
    manager.record_policy_trace(
        {"policy_id": "dummy:policy_test", "step": 1, "status": "selected"}
    )
    manager.record_policy_selection(
        policy_id="dummy:policy_test",
        score=0.75,
        components={"coherence": 0.8},
        step=1,
    )
    manager.record_policy_outcome(
        policy_id="dummy:policy_test",
        reward=1.0,
        done=False,
        step=1,
    )

    operations = [event["operation"] for event, _ in logger.events]
    assert "set_active_policy_for_pe" in operations
    assert "record_pe" in operations
    assert "record_state" in operations
    assert "snapshot_self_state" in operations
    assert "record_policy_trace" in operations
    assert "record_policy_selection" in operations
    assert "record_policy_outcome" in operations
