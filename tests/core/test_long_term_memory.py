from __future__ import annotations

from pathlib import Path

from core.memory.long_term_memory import LongTermMemory


def test_long_term_memory_recovers_from_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "policies.json"
    path.write_text("{ not-json", encoding="utf-8")

    ltm = LongTermMemory(path=str(path), max_score_history=5, max_outcome_history=5)
    policies = ltm.get_policies()
    assert policies == []

    backups = list(tmp_path.glob("policies.json.corrupt.*"))
    assert backups, "Expected corrupt backup file to be created."


def test_long_term_memory_persists_policy_selection_and_outcome(tmp_path: Path) -> None:
    path = tmp_path / "policies.json"
    ltm = LongTermMemory(path=str(path), max_score_history=5, max_outcome_history=5)
    ltm.upsert_policies(
        [
            {
                "policy_id": "dummy:policy_walk",
                "adapter_folder": "dummy",
                "callable_name": "policy_walk",
                "source": "introspection",
                "signature": "()",
                "tags": ["walk"],
            }
        ]
    )
    ltm.record_policy_selection(
        policy_id="dummy:policy_walk",
        score=0.75,
        components={"coherence_score": 0.8, "prediction_error_score": 0.3},
        step=3,
    )
    ltm.record_policy_outcome(
        policy_id="dummy:policy_walk",
        reward=1.0,
        done=False,
        step=3,
    )

    reloaded = LongTermMemory(path=str(path), max_score_history=5, max_outcome_history=5)
    policies = reloaded.get_policies(adapter_folder="dummy")
    assert len(policies) == 1
    policy = policies[0]
    assert policy["policy_id"] == "dummy:policy_walk"
    assert policy["selected_count"] == 1
    assert policy["success_count"] == 1
    assert len(policy["score_history"]) == 1
    assert len(policy["outcome_history"]) == 1
