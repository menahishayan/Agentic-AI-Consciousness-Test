"""
Action Mapper — translates abstract policy IDs to Animal AI action vectors.

Animal AI uses a discrete action space with two branches:
  Branch 0 (movement): 0=no-op, 1=forward, 2=backward
  Branch 1 (rotation): 0=no-op, 1=turn-left, 2=turn-right

The brain selects policy IDs (strings). This mapper converts them to the
numpy array format expected by AnimalAI.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# Action vector layout: [movement_branch, rotation_branch]
# Branch values: 0=no-op for that branch
_POLICY_TO_ACTION: Dict[str, List[int]] = {
    "move_forward":  [1, 0],
    "move_backward": [2, 0],
    "turn_left":     [0, 1],
    "turn_right":    [0, 2],
    "idle":          [0, 0],
}

_DEFAULT_ACTION = [0, 0]  # idle


def policy_id_to_action(policy_id: str) -> np.ndarray:
    """
    Convert a policy_id string to an Animal AI action array.

    Returns shape (2,) int32 array matching AnimalAI's discrete action space.
    Unknown policy_ids fall back to idle.
    """
    raw = _POLICY_TO_ACTION.get(policy_id, _DEFAULT_ACTION)
    return np.array(raw, dtype=np.int32)


def get_policy_descriptors() -> List[Dict]:
    """
    Return policy descriptor dicts for all available actions.
    These are consumed by AnimalAIAdapter.get_available_policies().
    """
    return [
        {
            "policy_id": "move_forward",
            "callable_name": "move_forward",
            "tags": ["navigation", "exploration"],
            "drive_tags": ["energy", "resource_level"],
            "description": "Move forward to explore and find food",
        },
        {
            "policy_id": "move_backward",
            "callable_name": "move_backward",
            "tags": ["navigation", "avoidance"],
            "drive_tags": ["safety"],
            "description": "Move backward to avoid hazards",
        },
        {
            "policy_id": "turn_left",
            "callable_name": "turn_left",
            "tags": ["navigation", "orientation"],
            "drive_tags": [],
            "description": "Turn left to change direction",
        },
        {
            "policy_id": "turn_right",
            "callable_name": "turn_right",
            "tags": ["navigation", "orientation"],
            "drive_tags": [],
            "description": "Turn right to change direction",
        },
        {
            "policy_id": "idle",
            "callable_name": "idle",
            "tags": ["rest"],
            "drive_tags": [],
            "description": "Stay still and observe the environment",
        },
    ]
