from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from core.layers.interoceptive import DriveChannel


_DEFAULT_MAX_INVENTORY_SLOTS = 36

_DEFAULT_CHANNEL_SPECS: Tuple[Dict[str, Any], ...] = (
    {
        "id": "health",
        "setpoint": 0.9,
        "critical_threshold": 0.25,
        "irreversible": True,
        "recovery_cost_ticks": 25,
        "suggested_action_tag": "heal",
    },
    {
        "id": "hunger",
        "setpoint": 0.8,
        "critical_threshold": 0.2,
        "irreversible": False,
        "recovery_cost_ticks": 20,
        "suggested_action_tag": "eat",
    },
    {
        "id": "oxygen",
        "setpoint": 0.9,
        "critical_threshold": 0.2,
        "irreversible": True,
        "recovery_cost_ticks": 30,
        "suggested_action_tag": "surface",
    },
    {
        "id": "resource_level",
        "setpoint": 0.7,
        "critical_threshold": 0.3,
        "irreversible": False,
        "recovery_cost_ticks": 15,
        "suggested_action_tag": "gather",
    },
    {
        "id": "safety",
        "setpoint": 0.8,
        "critical_threshold": 0.35,
        "irreversible": True,
        "recovery_cost_ticks": 20,
        "suggested_action_tag": "retreat",
    },
)


def normalize_task_id(task_id: Any) -> str:
    raw = str(task_id or "").strip().lower()
    if not raw:
        return "unknown_task"
    normalized = re.sub(r"[^a-z0-9_]+", "_", raw)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "unknown_task"


def task_goal_payload(task_id: str) -> Dict[str, Any]:
    normalized = normalize_task_id(task_id)
    descriptions = {
        "harvest_milk": "Harvest milk by obtaining or using a bucket and interacting with a cow.",
    }
    return {
        "goal_id": f"task:{normalized}",
        "goal": normalized,
        "description": descriptions.get(
            normalized,
            f"Complete MineDojo task '{normalized}'.",
        ),
        "priority": 1.0,
        "task_id": normalized,
    }


def drive_channels_for_task(task_id: str) -> List[DriveChannel]:
    normalized = normalize_task_id(task_id)
    resource_suggested_action = "harvest" if normalized == "harvest_milk" else "gather"

    channels: List[DriveChannel] = []
    for spec in _DEFAULT_CHANNEL_SPECS:
        suggested = spec["suggested_action_tag"]
        if spec["id"] == "resource_level":
            suggested = resource_suggested_action
        channels.append(
            DriveChannel(
                id=str(spec["id"]),
                setpoint=float(spec["setpoint"]),
                critical_threshold=float(spec["critical_threshold"]),
                irreversible=bool(spec["irreversible"]),
                recovery_cost_ticks=max(0, int(spec["recovery_cost_ticks"])),
                suggested_action_tag=str(suggested),
            )
        )
    return channels


def task_phase_intents(
    task_id: str,
    world_facts: Optional[Mapping[str, Any]] = None,
) -> List[List[str]]:
    normalized = normalize_task_id(task_id)
    if normalized == "harvest_milk":
        facts = world_facts if isinstance(world_facts, Mapping) else {}
        has_bucket = _as_bool(facts.get("has_bucket"), default=False)
        nearby_cow = _as_bool(facts.get("nearby_cow"), default=False)
        craft_possible = _as_bool(facts.get("nearby_crafting_table"), default=False)

        craft_phase = ["bucket", "craft", "collect"]
        use_phase = ["cow", "milk", "harvest", "interact", "use"]
        explore_phase = ["explore", "move", "search"]

        if has_bucket and nearby_cow:
            # Ready to harvest now: use/interact first.
            return [use_phase, explore_phase, craft_phase]
        if has_bucket and not nearby_cow:
            # Bucket acquired, still searching for a cow.
            return [explore_phase, use_phase, craft_phase]
        if craft_possible:
            # No bucket yet but crafting is feasible in the current area.
            return [craft_phase, explore_phase, use_phase]
        # No bucket and no nearby crafting support: explore first.
        return [explore_phase, craft_phase, use_phase]

    return [
        ["explore", "move", "search"],
        ["collect", "resource", "gather"],
        ["interact", "use", "craft"],
    ]


def build_skill_plan_queue(
    *,
    task_id: str,
    policies: Sequence[Mapping[str, Any]],
    world_facts: Optional[Mapping[str, Any]] = None,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    normalized = normalize_task_id(task_id)
    intents = task_phase_intents(normalized, world_facts=world_facts)

    queue: List[str] = []
    metadata: List[Dict[str, Any]] = []
    used_policy_ids: set[str] = set()

    for index, intent_tokens in enumerate(intents):
        selected, score = _select_policy_for_intent(
            intent_tokens=intent_tokens,
            policies=policies,
            used_policy_ids=used_policy_ids,
        )
        source = "intent_match"
        if selected is None:
            selected = _fallback_policy_for_task(
                task_id=normalized,
                policies=policies,
                used_policy_ids=used_policy_ids,
                intent_tokens=intent_tokens,
            )
            score = 0.0
            source = "fallback"
        if selected is None:
            continue
        queue.append(selected)
        used_policy_ids.add(selected)
        metadata.append(
            {
                "phase_index": index,
                "intent_tokens": list(intent_tokens),
                "selected_policy_id": selected,
                "selection_source": source,
                "score": float(score),
            }
        )

    if not queue:
        single = _fallback_policy_for_task(
            task_id=normalized,
            policies=policies,
            used_policy_ids=set(),
            intent_tokens=["explore", "move", "use"],
        )
        if single is not None:
            queue = [single]
            metadata.append(
                {
                    "phase_index": 0,
                    "intent_tokens": ["explore", "move", "use"],
                    "selected_policy_id": single,
                    "selection_source": "fallback",
                    "score": 0.0,
                }
            )
    return queue, metadata


def estimate_inventory_score(
    inventory_state: Mapping[str, Any],
    max_slots: int = _DEFAULT_MAX_INVENTORY_SLOTS,
) -> Optional[float]:
    if not isinstance(inventory_state, Mapping):
        return None
    raw_inventory = inventory_state.get("inventory")
    if not isinstance(raw_inventory, list):
        return None

    valid_slots = max(1, int(max_slots))
    non_air_items = [
        item
        for item in raw_inventory
        if isinstance(item, Mapping)
        and str(item.get("name", "")).strip().lower() != "air"
        and _as_non_negative_int(item.get("quantity")) > 0
    ]
    return clip01(float(len(non_air_items)) / float(valid_slots))


def clip01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return max(0.0, min(1.0, numeric))


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _select_policy_for_intent(
    *,
    intent_tokens: Sequence[str],
    policies: Sequence[Mapping[str, Any]],
    used_policy_ids: set[str],
) -> Tuple[Optional[str], float]:
    best_policy_id: Optional[str] = None
    best_score = 0.0
    normalized_intent = {
        token.strip().lower()
        for token in intent_tokens
        if isinstance(token, str) and token.strip()
    }
    if not normalized_intent:
        return None, 0.0

    for policy in policies:
        policy_id = str(policy.get("policy_id") or "").strip()
        if not policy_id or policy_id in used_policy_ids:
            continue
        tokens = _policy_tokens(policy)
        score = _intent_score(intent=normalized_intent, tokens=tokens)
        if score > best_score:
            best_policy_id = policy_id
            best_score = score
    return best_policy_id, best_score


def _fallback_policy_for_task(
    *,
    task_id: str,
    policies: Sequence[Mapping[str, Any]],
    used_policy_ids: set[str],
    intent_tokens: Sequence[str],
) -> Optional[str]:
    fallback_hints: List[str]
    if task_id == "harvest_milk":
        fallback_hints = ["use", "craft", "interact", "move_forward", "sprint", "no_op"]
    else:
        fallback_hints = ["use", "move_forward", "sprint", "no_op"]

    fallback_hints.extend(
        token.strip().lower()
        for token in intent_tokens
        if isinstance(token, str) and token.strip()
    )

    for hint in fallback_hints:
        for policy in policies:
            policy_id = str(policy.get("policy_id") or "").strip()
            if not policy_id or policy_id in used_policy_ids:
                continue
            tokens = _policy_tokens(policy)
            if hint in tokens:
                return policy_id
    return None


def _policy_tokens(policy: Mapping[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for key in ("policy_id", "callable_name", "description"):
        value = policy.get(key)
        if value is not None:
            tokens.update(_name_tokens(str(value)))

    for key in ("tags", "drive_tags"):
        raw = policy.get(key)
        if not isinstance(raw, Iterable) or isinstance(raw, (str, bytes)):
            continue
        for value in raw:
            tokens.update(_name_tokens(str(value)))
    return tokens


def _intent_score(*, intent: set[str], tokens: set[str]) -> float:
    if not intent or not tokens:
        return 0.0
    overlap = intent.intersection(tokens)
    if not overlap:
        return 0.0
    direct = float(len(overlap))
    coverage = float(len(overlap)) / float(len(intent))
    return direct + coverage


def _name_tokens(value: str) -> List[str]:
    return [token for token in re.split(r"[^a-z0-9]+", value.lower()) if token]


def _as_non_negative_int(value: Any) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, numeric)
