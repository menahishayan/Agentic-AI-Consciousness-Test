from __future__ import annotations

from typing import Any

from core.models.signals import ActionProposal


def map_action(action_proposal: ActionProposal) -> Any:
    raise NotImplementedError("Action mapping not implemented.")
