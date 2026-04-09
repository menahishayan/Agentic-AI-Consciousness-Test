"""Tests for memory subsystems."""
import pytest
import tempfile
import os
from core.memory.working_memory_buffer import WorkingMemoryBuffer
from core.memory.long_term_memory import LongTermMemory
from core.models.state import AgentState, HomeostasisState


def _state(health=0.8, step=0):
    return AgentState(homeostasis=HomeostasisState(health=health, saturation=0.7), step=step)


def test_working_memory_capacity():
    buf = WorkingMemoryBuffer(capacity=5)
    for i in range(10):
        buf.record(_state(step=i))
    assert len(buf) == 5
    recent = buf.get_recent(3)
    assert len(recent) == 3
    assert recent[-1].step == 9


def test_long_term_memory_persists(tmp_path):
    cfg = {"long_term_memory_path": str(tmp_path), "max_score_history": 50, "max_outcome_history": 50}
    ltm = LongTermMemory(cfg)
    ltm.record_outcome("move_forward", 0.8, "success")
    ltm.record_outcome("move_forward", 0.9, "success")
    ltm.record_outcome("idle", 0.2, "failure")

    # New instance reads from disk
    ltm2 = LongTermMemory(cfg)
    assert ltm2.get_success_rate("move_forward") > 0.5
    assert ltm2.get_success_rate("idle") < 0.5
    assert ltm2.get_success_rate("unknown_policy") == pytest.approx(0.5)


def test_long_term_memory_neutral_prior(tmp_path):
    cfg = {"long_term_memory_path": str(tmp_path)}
    ltm = LongTermMemory(cfg)
    assert ltm.get_success_rate("never_seen") == pytest.approx(0.5)


def test_policy_traces_outcome_history():
    from core.memory.policy_traces import PolicyTraces
    traces = PolicyTraces({"faiss_k_default": 3})
    from core.models.memory_records import PolicyTraceRecord
    for i in range(5):
        traces.record(PolicyTraceRecord(
            step=i, policy_id="move_forward",
            context_vector=[0.5] * 10, outcome_score=0.8,
            drive_signals={},
        ))
    rate = traces.get_policy_outcome_history("move_forward")
    assert rate == pytest.approx(0.8, abs=0.01)
    assert traces.get_policy_outcome_history("unknown") == pytest.approx(0.5)
