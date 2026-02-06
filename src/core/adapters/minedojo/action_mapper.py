from __future__ import annotations

from typing import Any

from core.models.signals import ActionProposal


def map_action(action_proposal: ActionProposal) -> Any:
    if action_proposal is None:
        return None
    if hasattr(action_proposal, "action") and action_proposal.action is not None:
        return action_proposal.action
    return action_proposal
