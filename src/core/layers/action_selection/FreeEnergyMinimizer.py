from __future__ import annotations

from typing import Any, Iterable, Optional


class FreeEnergyMinimizer:
    def select_action(self, proposals: Any) -> Optional[Any]:
        if proposals is None:
            return None
        if not isinstance(proposals, Iterable) or isinstance(proposals, (str, bytes, dict)):
            return proposals

        best = None
        best_cost = None
        for proposal in proposals:
            cost = getattr(proposal, "cost", None)
            if not isinstance(cost, (int, float)):
                continue
            if best is None or cost < best_cost:
                best = proposal
                best_cost = float(cost)

        if best is not None:
            return best

        for proposal in proposals:
            return proposal
        return None
