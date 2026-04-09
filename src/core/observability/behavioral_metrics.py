"""
BehavioralMetrics — offline behavioral probe extractor.

Reads a completed run's metrics.jsonl and computes the four probes that
target the theoretical claims of the interoceptive/predictive architecture,
rather than raw survival score.

The four probes:

  Probe 1 — Drive-Urgency Behavioral Coupling (DBC)
    Pearson r between drive_urgency and food-seeking action rate in rolling
    20-step windows. Full system: r > 0.4. llm_only / no_interoceptive: near 0.

  Probe 2 — Arousal-Diversity Index (ADI)
    Pearson r between mean arousal and Shannon entropy of the action
    distribution in non-overlapping 10-step windows. Full system: positive
    correlation. no_arousal: near 0.

  Probe 3 — Satiation Reorientation Latency (SRL)
    Steps from the drive-injection event until the agent issues its first
    non-food-directed action. Full system: low (interoceptive layer reacts
    immediately). llm_only: higher (must infer from raw value).

  Probe 4 — Urgency-Quartile Food Collection Rate (UQCR)
    Food collection rate (positive saturation delta) per drive-urgency
    quartile. Full system: Q4 >> Q1. F-test across conditions.

Usage:
    from pathlib import Path
    from core.observability.behavioral_metrics import BehavioralMetrics

    bm = BehavioralMetrics(run_dir=Path("src/logs/runs/20260409_abc123"),
                           injection_step=200)
    print(bm.summary())
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class BehavioralMetrics:
    """
    Offline behavioral probe extractor.

    Reads a completed run directory and computes four theory-targeted probes.
    Does not add runtime overhead — runs post-hoc against stored metrics.jsonl.
    """

    def __init__(
        self,
        run_dir: Path,
        injection_step: Optional[int] = None,
    ) -> None:
        self._run_dir = Path(run_dir)
        self._rows: List[Dict[str, Any]] = self._load_metrics()
        # Prefer explicitly passed injection_step; fall back to events.jsonl
        self._injection_step: Optional[int] = (
            injection_step if injection_step is not None
            else self._find_injection_step()
        )

    # ------------------------------------------------------------------
    # Public probe methods
    # ------------------------------------------------------------------

    def drive_behavior_coupling(self) -> float:
        """
        Probe 1 — DBC: Pearson r between drive urgency and food-seeking rate.

        Food-seeking action = move_forward (primary pragmatic food approach).
        Computed over rolling 20-step windows; each window contributes one
        (urgency_mean, food_rate) pair to the correlation.

        Full system theoretical prediction: r > 0.4
        llm_only / no_interoceptive prediction: r ≈ 0 (no urgency signal)
        """
        if len(self._rows) < 20:
            return float("nan")

        urgency_vals = [float(r.get("drive_urgency", 0.0)) for r in self._rows]
        policy_ids = [r.get("policy_id", "idle") for r in self._rows]

        window = 20
        urgency_means: List[float] = []
        food_rates: List[float] = []

        for i in range(len(self._rows) - window + 1):
            w_urgency = urgency_vals[i : i + window]
            w_policies = policy_ids[i : i + window]
            urgency_means.append(float(np.mean(w_urgency)))
            food_rate = sum(1 for p in w_policies if p == "move_forward") / window
            food_rates.append(food_rate)

        if len(urgency_means) < 5:
            return float("nan")

        r = np.corrcoef(urgency_means, food_rates)[0, 1]
        return float(r) if not math.isnan(float(r)) else float("nan")

    def arousal_diversity_index(self) -> float:
        """
        Probe 2 — ADI: Pearson r between arousal and action entropy.

        Action entropy = Shannon entropy of the policy_id distribution in a
        non-overlapping 10-step window. Correlated with mean arousal per window.

        Full system theoretical prediction: positive r (high arousal → exploration)
        no_arousal prediction: near 0 (no arousal signal)

        Implements the LC-NE pathway claim: arousal modulates behavioral
        diversity, not just action selection confidence.
        """
        if len(self._rows) < 10:
            return float("nan")

        window = 10
        arousal_means: List[float] = []
        entropies: List[float] = []

        for i in range(0, len(self._rows) - window + 1, window):
            chunk = self._rows[i : i + window]
            if len(chunk) < window:
                break

            mean_arousal = float(np.mean([float(r.get("arousal", 0.0)) for r in chunk]))
            policies = [r.get("policy_id", "idle") for r in chunk]

            counts: Dict[str, int] = {}
            for p in policies:
                counts[p] = counts.get(p, 0) + 1
            total = len(policies)
            entropy = 0.0
            for count in counts.values():
                p_prob = count / total
                if p_prob > 0.0:
                    entropy -= p_prob * math.log2(p_prob)

            arousal_means.append(mean_arousal)
            entropies.append(entropy)

        if len(arousal_means) < 5:
            return float("nan")

        r = np.corrcoef(arousal_means, entropies)[0, 1]
        return float(r) if not math.isnan(float(r)) else float("nan")

    def satiation_reorientation_latency(self) -> int:
        """
        Probe 3 — SRL: steps from drive injection until first non-food-seeking action.

        Food-directed action = move_forward.
        Non-food-directed = any other action (turn, idle, backward).

        Full system prediction: low SRL (urgency drops immediately, EFE shifts)
        llm_only prediction: higher SRL (must infer from raw saturation value)

        Returns -1 if no injection step configured or agent reoriented immediately.
        Returns len(post_injection_rows) if agent never reoriented (all food-seeking
        until episode end — treat as worst-case latency).
        """
        if self._injection_step is None:
            return -1

        post_injection = [
            r for r in self._rows
            if r.get("step", 0) >= self._injection_step
        ]
        if not post_injection:
            return -1

        for latency, r in enumerate(post_injection):
            if r.get("policy_id", "idle") != "move_forward":
                return latency

        # Never reoriented — return total post-injection steps as worst-case
        return len(post_injection)

    def urgency_quartile_collection_rate(self) -> Dict[str, float]:
        """
        Probe 4 — UQCR: food collection rate per drive-urgency quartile.

        Food collection = positive saturation delta (> 0.01 threshold to exclude
        passive depletion noise). Each step is assigned to a urgency quartile;
        rate = fraction of steps in that quartile with a collection event.

        Full system prediction: Q4 >> Q1 (urgency gates food-seeking behavior)
        llm_only prediction: weaker Q4/Q1 ratio (raw value read, no urgency gating)
        """
        if len(self._rows) < 4:
            return {"q1": 0.0, "q2": 0.0, "q3": 0.0, "q4": 0.0}

        urgency_vals = [float(r.get("drive_urgency", 0.0)) for r in self._rows]
        q25 = float(np.percentile(urgency_vals, 25))
        q50 = float(np.percentile(urgency_vals, 50))
        q75 = float(np.percentile(urgency_vals, 75))

        quartile_collections: Dict[str, List[float]] = {
            "q1": [], "q2": [], "q3": [], "q4": [],
        }

        for i in range(1, len(self._rows)):
            prev_sat = float(self._rows[i - 1].get("saturation", 0.0))
            curr_sat = float(self._rows[i].get("saturation", 0.0))
            delta = curr_sat - prev_sat
            # Threshold: exclude passive depletion noise (< 0.01), flag food events
            collected = 1.0 if delta > 0.01 else 0.0

            urgency = urgency_vals[i]
            if urgency <= q25:
                quartile_collections["q1"].append(collected)
            elif urgency <= q50:
                quartile_collections["q2"].append(collected)
            elif urgency <= q75:
                quartile_collections["q3"].append(collected)
            else:
                quartile_collections["q4"].append(collected)

        return {
            q: float(np.mean(vals)) if vals else 0.0
            for q, vals in quartile_collections.items()
        }

    def summary(self) -> Dict[str, Any]:
        """
        All four probes plus episode-level survival stats.

        Keys:
          dbc          — Drive-Urgency Behavioral Coupling (Pearson r)
          adi          — Arousal-Diversity Index (Pearson r)
          srl          — Satiation Reorientation Latency (steps; -1 = no injection)
          uqcr_q1      — Food collection rate in urgency Q1 (low urgency)
          uqcr_q4      — Food collection rate in urgency Q4 (high urgency)
          uqcr_q4_q1   — Q4/Q1 ratio (nan if Q1=0)
          steps_survived — total episode steps logged
        """
        uqcr = self.urgency_quartile_collection_rate()
        q1 = uqcr.get("q1", 0.0)
        q4 = uqcr.get("q4", 0.0)
        q4_q1 = q4 / q1 if q1 > 0.0 else float("nan")

        return {
            "dbc": self.drive_behavior_coupling(),
            "adi": self.arousal_diversity_index(),
            "srl": self.satiation_reorientation_latency(),
            "uqcr_q1": q1,
            "uqcr_q4": q4,
            "uqcr_q4_q1": q4_q1,
            "steps_survived": len(self._rows),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_metrics(self) -> List[Dict[str, Any]]:
        path = self._run_dir / "metrics.jsonl"
        if not path.exists():
            return []
        rows: List[Dict[str, Any]] = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return sorted(rows, key=lambda r: int(r.get("step", 0)))

    def _find_injection_step(self) -> Optional[int]:
        """Scan events.jsonl for a drive_injection event and return its step."""
        path = self._run_dir / "events.jsonl"
        if not path.exists():
            return None
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
                if ev.get("event") == "drive_injection":
                    return int(ev.get("step", 0))
            except (json.JSONDecodeError, ValueError):
                pass
        return None
