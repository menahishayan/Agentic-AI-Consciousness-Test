"""
RemoteMineDojoAdapter — drop-in replacement for MineDojoAdapter that talks
to a running persistent_server.py instead of owning the Minecraft process.

Usage in config.json:
    "adapter_folder": "minedojo",          ← unchanged
    "adapter_config": {
        "task_id":    "harvest_milk",
        "image_size": [160, 256],
        "use_remote": true,                ← add this line
        "remote_host": "127.0.0.1",        ← optional, default shown
        "remote_port": 9876                ← optional, default shown
    }

If use_remote is false/absent the original MineDojoAdapter is used (starts
its own Minecraft process as before).
"""
from __future__ import annotations

import pickle
import socket
import struct
from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional, Tuple

from core.adapters.minedojo.task_profiles import (
    build_skill_plan_queue,
    drive_channels_for_task,
    estimate_inventory_score,
    normalize_task_id,
    task_goal_payload,
)

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 9876


# ---------------------------------------------------------------------------
# Wire helpers  (identical to persistent_server.py)
# ---------------------------------------------------------------------------

def _send(conn: socket.socket, obj: Any) -> None:
    data = pickle.dumps(obj, protocol=4)
    conn.sendall(struct.pack(">I", len(data)) + data)


def _recv(conn: socket.socket) -> Any:
    raw_len = _recv_exactly(conn, 4)
    if not raw_len:
        raise EOFError("connection closed")
    (length,) = struct.unpack(">I", raw_len)
    return pickle.loads(_recv_exactly(conn, length))


def _recv_exactly(conn: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise EOFError("connection closed mid-message")
        buf += chunk
    return buf


# ---------------------------------------------------------------------------
# RemoteMineDojoAdapter
# ---------------------------------------------------------------------------

class RemoteMineDojoAdapter:
    """
    Identical public interface to MineDojoAdapter.
    All env calls are forwarded to the persistent server over TCP.
    """

    _VITALS: Tuple[str, ...] = (
        "life", "armor", "food", "saturation", "xp", "air",
        "is_sleeping", "is_alive", "is_dead",
    )

    # These are the same drive tag rules as MineDojoAdapter / AgentLoop.
    _DRIVE_TAG_RULES: Dict[str, set] = {
        "health":         {"heal", "retreat", "defend", "protect", "shield", "escape"},
        "hunger":         {"eat", "collect", "cook", "food", "harvest", "fish", "drink"},
        "oxygen":         {"surface", "ascend", "air", "breath"},
        "resource_level": {
            "gather",
            "mine",
            "craft",
            "smelt",
            "chop",
            "dig",
            "harvest",
            "interact",
            "bucket",
            "milk",
        },
        "safety":         {"retreat", "avoid", "shelter", "defend", "escape"},
    }

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self._config = dict(config or {})
        self._task_id = normalize_task_id(self._config.get("task_id", "harvest_milk"))
        self._task_goal = task_goal_payload(self._task_id)
        try:
            self._max_inventory_slots = max(1, int(self._config.get("max_inventory_slots", 36)))
        except (TypeError, ValueError):
            self._max_inventory_slots = 36
        self._skill_plan_queue: List[str] = []
        self._skill_plan_metadata: List[Dict[str, Any]] = []
        self._last_skill_plan_signature: Optional[str] = None
        self._skill_plan_exhausted = False
        host = str(self._config.get("remote_host", _DEFAULT_HOST))
        port = int(self._config.get("remote_port", _DEFAULT_PORT))

        self._conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._conn.connect((host, port))

        # Get action space info from server so we can reconstruct policies
        _send(self._conn, {"cmd": "get_action_space"})
        resp = _recv(self._conn)
        if not resp.get("ok"):
            raise RuntimeError(f"Could not get action space: {resp.get('error')}")
        space_info = resp["result"]
        self._nvec: List[int] = space_info.get("nvec") or []
        self._noop_action: List[int] = space_info.get("noop") or [0] * 8

        self._last_obs: Any = None
        self._last_info: Dict[str, Any] = {}
        self._policies: List[Dict[str, Any]] = self._build_policies()

    # ------------------------------------------------------------------
    # Core env interface
    # ------------------------------------------------------------------

    def reset(self) -> Tuple[Any, Dict]:
        _send(self._conn, {"cmd": "reset"})
        resp = _recv(self._conn)
        if not resp.get("ok"):
            raise RuntimeError(f"reset() failed: {resp.get('error')}")
        obs, info = resp["result"]
        self._last_obs = obs
        self._last_info = info if isinstance(info, dict) else {}
        self._skill_plan_queue = []
        self._skill_plan_metadata = []
        self._last_skill_plan_signature = None
        self._skill_plan_exhausted = False
        return obs, self._last_info

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        if hasattr(action, "tolist"):
            action = action.tolist()
        elif not isinstance(action, list):
            action = list(action) if action is not None else self._noop_action
        _send(self._conn, {"cmd": "step", "action": action})
        resp = _recv(self._conn)
        if not resp.get("ok"):
            raise RuntimeError(f"step() failed: {resp.get('error')}")
        obs, reward, done, info = resp["result"]
        self._last_obs = obs
        self._last_info = info if isinstance(info, dict) else {}
        return obs, reward, done, self._last_info

    def close(self) -> None:
        try:
            _send(self._conn, {"cmd": "close"})
            _recv(self._conn)
        except Exception:
            pass
        try:
            self._conn.close()
        except Exception:
            pass

    def sample_action(self) -> List[int]:
        import random
        return [random.randint(0, max(0, n - 1)) for n in (self._nvec or [3, 3, 4, 25, 25, 8, 101, 9])]

    # ------------------------------------------------------------------
    # Adapter interface required by loader._validate_adapter
    # ------------------------------------------------------------------

    def get_available_vitals(self) -> List[str]:
        return list(self._VITALS)

    def get_available_policies(self) -> List[Dict[str, Any]]:
        return deepcopy(self._policies)

    def get_task_goals(self) -> List[Dict[str, Any]]:
        return [deepcopy(self._task_goal)]

    def get_drive_channels(self) -> List[Any]:
        return drive_channels_for_task(self._task_id)

    def get_skill_plan(
        self,
        goals: Any = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        _ = context
        signature = self._skill_plan_signature(goals)
        should_refresh = False
        if signature != self._last_skill_plan_signature:
            should_refresh = True
        elif self._last_skill_plan_signature is None:
            should_refresh = True
        elif not self._skill_plan_queue and not self._skill_plan_exhausted:
            should_refresh = True

        if should_refresh:
            queue, metadata = build_skill_plan_queue(
                task_id=self._task_id,
                policies=self._policies,
            )
            self._skill_plan_queue = list(queue)
            self._skill_plan_metadata = list(metadata)
            self._last_skill_plan_signature = signature
            self._skill_plan_exhausted = not bool(self._skill_plan_queue)

        return {
            "task_id": self._task_id,
            "head_policy_id": self._skill_plan_queue[0] if self._skill_plan_queue else None,
            "queue": list(self._skill_plan_queue),
            "remaining_policy_ids": list(self._skill_plan_queue),
            "remaining_count": len(self._skill_plan_queue),
            "metadata": deepcopy(self._skill_plan_metadata),
        }

    def notify_policy_selected(
        self,
        policy_id: Any,
        context: Optional[Mapping[str, Any]] = None,
    ) -> None:
        _ = context
        selected = str(policy_id or "").strip()
        if not selected or not self._skill_plan_queue:
            return
        if selected != self._skill_plan_queue[0]:
            return
        self._skill_plan_queue.pop(0)
        if not self._skill_plan_queue:
            self._skill_plan_exhausted = True

    def estimate_resource_level(self, **kwargs: Any) -> float:
        hunger = kwargs.get("hunger", 0.0)
        hunger_score = self._clip01(hunger)
        inventory_state = kwargs.get("inventory_state")
        if not isinstance(inventory_state, Mapping):
            info = kwargs.get("info") or self._last_info
            if isinstance(info, Mapping):
                inventory_state = {"inventory": info.get("inventory")}
        if not isinstance(inventory_state, Mapping):
            return hunger_score
        inventory_score = estimate_inventory_score(
            inventory_state,
            max_slots=self._max_inventory_slots,
        )
        if inventory_score is None:
            return hunger_score
        return self._clip01(0.6 * float(inventory_score) + 0.4 * hunger_score)

    def estimate_threat_proximity(self, **kwargs: Any) -> float:
        info = kwargs.get("info") or self._last_info
        if not isinstance(info, dict):
            return 0.0
        life = info.get("life")
        if life is None:
            return 0.0
        return max(0.0, 1.0 - float(life) / 20.0)

    def build_area_id(self, **kwargs: Any) -> str:
        info = kwargs.get("info") or self._last_info
        if not isinstance(info, dict):
            return "chunk_0_0"
        x = int(float(info.get("xpos", 0)) // 16)
        z = int(float(info.get("zpos", 0)) // 16)
        return f"chunk_{x}_{z}"

    def estimate_entity_density(self, **kwargs: Any) -> float:
        return self.estimate_threat_proximity(**kwargs)

    def estimate_terrain_novelty(self, **kwargs: Any) -> float:
        info = kwargs.get("info") or self._last_info
        if not isinstance(info, dict):
            return 0.0
        import math
        x = float(info.get("xpos", 0))
        z = float(info.get("zpos", 0))
        return min(1.0, math.sqrt(x ** 2 + z ** 2) / 500.0)

    # ------------------------------------------------------------------
    # Policy building  (mirrors MineDojoAdapter primitive set)
    # ------------------------------------------------------------------

    def _build_policies(self) -> List[Dict[str, Any]]:
        """
        Build the policy descriptor list AND bind a real callable method on
        this instance for each policy.  PolicyGenerator requires
        getattr(adapter, callable_name) to return a callable — without this
        every descriptor gets silently dropped and propose_action returns None.

        Each bound method accepts **kwargs (so PolicyGenerator can pass the
        full context dict) and returns the pre-built action vector.
        """
        nvec = self._nvec
        noop = list(self._noop_action)

        def vec(overrides: Dict[int, int]) -> List[int]:
            v = list(noop)
            for dim, val in overrides.items():
                if dim < len(v):
                    v[dim] = val
            return v

        specs = [
            # (policy_id,                               callable_name,   description,                      tags,                                              drive_tags,           action_vec)
            ("minedojo:action_space:no_op",       "no_op",       "No operation.",                   ["action_space", "primitive", "noop"],              [],                   noop),
            ("minedojo:action_space:move_forward","move_forward","Move forward.",                    ["action_space", "primitive", "move"],              [],                   vec({0: 1})),
            ("minedojo:action_space:move_back",   "move_back",   "Move backward.",                   ["action_space", "primitive", "move"],              [],                   vec({0: 2})),
            ("minedojo:action_space:move_left",   "move_left",   "Strafe left.",                     ["action_space", "primitive", "move"],              [],                   vec({1: 1})),
            ("minedojo:action_space:move_right",  "move_right",  "Strafe right.",                    ["action_space", "primitive", "move"],              [],                   vec({1: 2})),
            ("minedojo:action_space:jump",        "jump",        "Jump.",                            ["action_space", "primitive", "jump"],              [],                   vec({2: 1})),
            ("minedojo:action_space:sprint",      "sprint",      "Sprint forward.",                  ["action_space", "primitive", "move"],              [],                   vec({0: 1, 2: 3})),
            ("minedojo:action_space:attack",      "attack",      "Attack / mine the block ahead.",   ["action_space", "primitive", "attack", "mine"],   ["resource_level"],   vec({5: 3, 0: 1})),
            ("minedojo:action_space:use",         "use",         "Use / eat item in hand.",          ["action_space", "primitive", "use", "eat"],        ["hunger"],           vec({5: 1})),
            ("minedojo:action_space:craft",       "craft",       "Craft or smelt.",                  ["action_space", "primitive", "craft"],             ["resource_level"],   vec({5: 4})),
            ("minedojo:action_space:equip",       "equip",       "Equip item in slot 0.",            ["action_space", "primitive", "equip"],             [],                   vec({5: 5, 7: 0})),
            ("minedojo:action_space:sneak",       "sneak",       "Sneak.",                           ["action_space", "primitive", "sneak"],             [],                   vec({2: 2})),
        ]

        policies = []
        for policy_id, callable_name, description, tags, drive_tags, action_vec in specs:
            # Bind a method on self so PolicyGenerator can find it via getattr.
            # We capture action_vec by default argument to avoid closure issues.
            def _make_method(av: List[int]):
                def _method(self_inner, **kwargs: Any) -> List[int]:  # noqa: ARG001
                    return list(av)
                return _method

            method = _make_method(action_vec)
            method.__name__ = callable_name
            # Bind to instance
            import types
            setattr(self, callable_name, types.MethodType(method, self))

            policies.append({
                "policy_id": policy_id,
                "callable_name": callable_name,
                "description": description,
                "tags": tags,
                "drive_tags": drive_tags,
                "_action_vector": action_vec,
            })

        self._apply_task_policy_overrides(policies)
        return policies

    def _apply_task_policy_overrides(self, policies: List[Dict[str, Any]]) -> None:
        if self._task_id != "harvest_milk":
            return

        interaction_policy_found = False
        for policy in policies:
            policy_id = str(policy.get("policy_id") or "").strip().lower()
            tags = {
                str(tag).strip().lower()
                for tag in policy.get("tags", [])
                if isinstance(tag, str) and tag.strip()
            }
            drive_tags = {
                str(tag).strip().lower()
                for tag in policy.get("drive_tags", [])
                if isinstance(tag, str) and tag.strip()
            }

            if policy_id.endswith(":attack"):
                drive_tags.discard("resource_level")

            if "use" in tags or "interact" in tags:
                drive_tags.add("resource_level")
                interaction_policy_found = True

            policy["drive_tags"] = sorted(drive_tags)

        if interaction_policy_found:
            return

        for policy in policies:
            policy_id = str(policy.get("policy_id") or "").strip().lower()
            if policy_id.endswith(":use"):
                drive_tags = {
                    str(tag).strip().lower()
                    for tag in policy.get("drive_tags", [])
                    if isinstance(tag, str) and tag.strip()
                }
                drive_tags.add("resource_level")
                policy["drive_tags"] = sorted(drive_tags)
                break

    def get_action_for_policy(self, policy_id: str) -> Optional[List[int]]:
        for p in self._policies:
            if p["policy_id"] == policy_id:
                return list(p["_action_vector"])
        return None

    @staticmethod
    def _clip01(value: Any) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 0.0
        return max(0.0, min(1.0, numeric))

    def _skill_plan_signature(self, goals: Any) -> str:
        goal_tokens: List[str] = []
        if isinstance(goals, list):
            for goal in goals:
                if isinstance(goal, Mapping):
                    raw = goal.get("goal_id") or goal.get("description") or goal.get("goal")
                else:
                    raw = goal
                token = str(raw or "").strip().lower()
                if token:
                    goal_tokens.append(token)
        elif goals is not None:
            goal_tokens.append(str(goals).strip().lower())
        deduped = list(dict.fromkeys(sorted(goal_tokens)))
        return f"{self._task_id}|{'|'.join(deduped)}"


# ---------------------------------------------------------------------------
# create_adapter  — called by loader.build_adapter
# ---------------------------------------------------------------------------

def create_adapter(config: Optional[Mapping[str, Any]] = None) -> Any:
    cfg = dict(config or {})

    # If use_remote is not set, fall back to local MineDojoAdapter
    if not cfg.get("use_remote", False):
        try:
            import minedojo  # type: ignore
        except Exception as exc:
            raise ImportError("minedojo package is required") from exc
        from core.adapters.minedojo.env_adapter import MineDojoAdapter
        task_id = cfg.get("task_id", "harvest_milk")
        image_size_value = cfg.get("image_size", (160, 256))
        if isinstance(image_size_value, (list, tuple)) and len(image_size_value) == 2:
            image_size = (int(image_size_value[0]), int(image_size_value[1]))
        else:
            image_size = (160, 256)
        env = minedojo.make(task_id=task_id, image_size=image_size)
        return MineDojoAdapter(env, config=cfg)

    return RemoteMineDojoAdapter(cfg)
