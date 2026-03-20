from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

from core.memory.long_term_memory import LongTermMemory


def test_long_term_memory_recovers_from_corrupt_manifest(tmp_path: Path) -> None:
    root = tmp_path / "long_term_memory"
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.json"
    manifest.write_text("{ not-json", encoding="utf-8")

    ltm = LongTermMemory(path=str(root), max_score_history=5, max_outcome_history=5)
    policies = ltm.get_policies()
    assert policies == []

    backups = list(root.glob("manifest.json.corrupt.*"))
    assert backups, "Expected corrupt backup file to be created."
    assert manifest.exists()


def test_long_term_memory_persists_policy_selection_and_outcome_sharded(tmp_path: Path) -> None:
    root = tmp_path / "long_term_memory"
    ltm = LongTermMemory(path=str(root), max_score_history=5, max_outcome_history=5)
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

    manifest_path = root / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest.get("layout") == "ltm_sharded_v1"
    index = manifest.get("policies", {})
    assert isinstance(index, dict)
    assert "dummy:policy_walk" in index

    shard_relative = index["dummy:policy_walk"]
    shard_path = root / str(shard_relative)
    expected_filename = f"{quote('dummy:policy_walk', safe='-_.~')}.json"
    assert shard_path.name == expected_filename
    assert shard_path.exists()

    reloaded = LongTermMemory(path=str(root), max_score_history=5, max_outcome_history=5)
    policies = reloaded.get_policies(adapter_folder="dummy")
    assert len(policies) == 1
    policy = policies[0]
    assert policy["policy_id"] == "dummy:policy_walk"
    assert policy["selected_count"] == 1
    assert policy["success_count"] == 1
    assert len(policy["score_history"]) == 1
    assert len(policy["outcome_history"]) == 1


def test_long_term_memory_accepts_legacy_file_like_path(tmp_path: Path) -> None:
    legacy_path = tmp_path / "policies.json"
    ltm = LongTermMemory(path=str(legacy_path), max_score_history=5, max_outcome_history=5)

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

    assert (tmp_path / "manifest.json").exists()
    shard_files = list((tmp_path / "policies").glob("*.json"))
    assert shard_files


def test_long_term_memory_ignores_legacy_single_file_when_manifest_absent(tmp_path: Path) -> None:
    legacy_path = tmp_path / "policies.json"
    legacy_payload = {
        "version": 1,
        "policies": {
            "dummy:legacy_policy": {
                "policy_id": "dummy:legacy_policy",
                "adapter_folder": "dummy",
            }
        },
    }
    legacy_text = json.dumps(legacy_payload, ensure_ascii=True, indent=2)
    legacy_path.write_text(legacy_text, encoding="utf-8")

    ltm = LongTermMemory(path=str(legacy_path), max_score_history=5, max_outcome_history=5)

    assert ltm.get_policies() == []
    assert not (tmp_path / "manifest.json").exists()
    assert legacy_path.read_text(encoding="utf-8") == legacy_text


def test_long_term_memory_backs_up_corrupt_policy_shard(tmp_path: Path) -> None:
    root = tmp_path / "long_term_memory"
    policies_dir = root / "policies"
    policies_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "layout": "ltm_sharded_v1",
        "version": 1,
        "policies": {"dummy:bad": "policies/dummy%3Abad.json"},
    }
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")
    (policies_dir / "dummy%3Abad.json").write_text("{ not-json", encoding="utf-8")

    ltm = LongTermMemory(path=str(root), max_score_history=5, max_outcome_history=5)

    assert ltm.get_policies() == []
    backups = list(policies_dir.glob("dummy%3Abad.json.corrupt.*"))
    assert backups
