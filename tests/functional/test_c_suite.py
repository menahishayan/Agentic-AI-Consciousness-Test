"""
Seth Functional Validation Suite — Tier C: Epistemic Exploration & Goal Persistence
=====================================================================================

C1  EFE Epistemic Decomposition
C2  Precision Weighting Adaptation (in-episode prior)
C3  Drive-Conditioned Goal Persistence (C3a + C3b)

Architecture Notes
------------------
These tests probe claims about the epistemic (information-gain) component of EFE
and the generative model's goal-persistence under food removal.

Known architectural limitations that cause legitimate test failures:

C1: The FEM's epistemic term drives SCANNING (turn_left/turn_right) rather than
    LOCOMOTION (move_forward).  In the test harness pe_batch=None, so pe_mean=0.
    For turns: epistemic = area_novelty + 0.60.
    For move_forward: epistemic = pe_mean + 0.60 = 0.60 (when pe_batch absent).
    At area_novelty > 0, turns always outscore move_forward epistemically.
    The agent scans in place; arena coverage stays near 0 in the full condition.
    NO_EPISTEMIC ablation zeroes all active-action scores; idle dominates.
    Both conditions produce low coverage, but for different reasons:
      full = turns dominate (epistemic scanning, no locomotion)
      ablated = idle dominates (no epistemic, no pragmatic at satiation)

C2: The FEM last_food_angle buffer has no Bayesian confidence parameter.
    Both high-prior (food visible 15+ steps) and low-prior (food visible 2 steps)
    conditions produce the same 30-step buffer decay after food disappears.
    Persistence (continued move_forward after removal) is statistically identical.
    This is an architectural limitation: the buffer strength does not scale with
    observation count.

C3a: Should pass.  Food memory bonus (sat_urgency × mem_bonus) biases
     move_forward for up to food_memory_decay_steps=30 steps after food removal
     when sat_urgency ≥ 0.30.  No-memory baseline (decay=1) terminates immediately.

C3b: Should pass.  At saturation=1.0 (above setpoint 0.70), sat_urgency=0 and
     the food-proximity bonus for move_forward vanishes.  The agent oscillates
     between turns toward food (epistemic) but never approaches (pragmatic=0).

Run the suite:
    pytest tests/functional/test_c_suite.py -v
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest
from scipy.stats import mannwhitneyu, wilcoxon

# ── path setup ───────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from core.adapters.headless.env_adapter import FoodItem, HeadlessSimAdapter
from core.coordination.workspace import GlobalWorkspace
from core.layers.action_selection.FreeEnergyMinimizer import FreeEnergyMinimizer
from core.layers.interoceptive.AllostaticController import AllostaticController

# Re-use shared infrastructure from B suite.
from tests.functional.test_b_suite import (
    _BRAIN_BASE,
    _SIM_BASE,
    _make_cfg,
    _run_episode,
    _run_n_episodes,
)

# ── Food position constants ──────────────────────────────────────────────────
# Agent starts at arena_size/2 = 15.0, heading=0 (facing +z).
#
# _FOOD_AHEAD_FAR: 10 units ahead.  Used for C3a (persistence test).
#   - Angular half-width = arcsin(1.0/10) ≈ 5.7° → visible when nearly head-on.
#   - Takes ~9 move_forward steps to consume (interaction_radius=1.0).
#   - Hungry agent approaches directly; food_removal_step=5 fires before eating.
#
# _FOOD_AHEAD_NEAR: 2.5 units ahead.  Used for C3b (satiation suppression).
#   - Angular half-width = arcsin(1.0/2.5) ≈ 23.6° → visible for 50+ steps as
#     the agent oscillates turns (food stays in ray fan).
#   - At satiation, epistemic turns dominate (sat_urgency=0 → no food bonus for
#     move_forward) so agent never approaches despite persistent food visibility.
#   - Verified: 0 approaches in 50-step window across all seeds at saturation=1.0.
_FOOD_AHEAD_FAR:  Tuple[float, float] = (15.0, 25.0)
_FOOD_AHEAD_NEAR: Tuple[float, float] = (15.0, 17.5)


# ============================================================================
# Shared helpers
# ============================================================================

def _arena_coverage(
    records: List[Dict[str, Any]],
    grid_size: float = 2.0,
    arena_size: float = 30.0,
) -> float:
    """
    Fraction of (grid_size × grid_size) cells visited.
    max_cells = (arena_size / grid_size)² = 225 for default params.
    """
    cells: set = set()
    for r in records:
        gx = int(float(r.get("x", 0.0)) // grid_size)
        gz = int(float(r.get("z", 0.0)) // grid_size)
        cells.add((gx, gz))
    max_cells = (arena_size / grid_size) ** 2
    return len(cells) / max_cells


def _persistence_steps(
    records: List[Dict[str, Any]],
    removal_step: int,
    max_steps: int = 60,
) -> int:
    """
    Count move_forward steps in [removal_step, removal_step + max_steps)
    where food is NOT visible in any ray.

    These are 'ghost-directed' steps: the agent is moving toward a remembered
    food location that no longer exists.  Non-consecutive counting: any
    move_forward-without-food in the window counts, whether adjacent or not.
    """
    count = 0
    for r in records:
        s = r["step"]
        if s < removal_step:
            continue
        if s >= removal_step + max_steps:
            break
        if r["policy_id"] == "move_forward" and not r["food_in_ray"]:
            count += 1
    return count


# ============================================================================
# C1 — EFE Epistemic Decomposition
# ============================================================================

_N_C1 = 20
_STEPS_C1 = 400


@pytest.mark.behavioral
def test_c1_epistemic_decomposition():
    """
    C1: The epistemic component of EFE drives arena exploration when pragmatic
    value is near zero (agent fully satiated, no urgency).

    Full architecture: epistemic turns keep the agent active → higher coverage
    than the NO_EPISTEMIC ablation where idle dominates.

    NO_EPISTEMIC ablation: w_epistemic_eff=0.  At satiation, idle score =
    (1−urgency)×(1−novelty) > 0 while active action scores clamp to 0 →
    agent stays still → near-zero coverage.

    Pass criteria (TEST_PROCEDURES.md §C1):
      Arena coverage at step 200:  full > 0.25,  ablated < 0.12
      Mann-Whitney U: full > ablated, p < 0.05
      Idle proportion (full, satiated): < 10%

    Known failure mode (architectural):
      Without pe_batch (pe_mean=0), move_forward's epistemic value equals the
      constant foraging baseline (0.60) while turns get (area_novelty + 0.60).
      Turns always beat move_forward; the agent scans in place rather than
      navigating.  Coverage may stay below 0.25 in the full condition.
      The qualitative direction (full > ablated) should hold, but the 0.25
      absolute threshold will fail.
    """
    sim_cfg_full, brain_cfg_full = _make_cfg(
        sim_overrides={
            "simulation": {"n_food": 4},
            "homeostatic": {"saturation_depletion_rate": 0.0005},
        },
        ablation_mode="full",
    )
    sim_cfg_abl, brain_cfg_abl = _make_cfg(
        sim_overrides={
            "simulation": {"n_food": 4},
            "homeostatic": {"saturation_depletion_rate": 0.0005},
        },
        ablation_mode="no_epistemic",
    )

    eps_full = _run_n_episodes(
        sim_cfg_full, brain_cfg_full, _N_C1, _STEPS_C1,
        base_seed=3000,
        starting_saturation=1.0,
        starting_health=0.90,
    )
    eps_abl = _run_n_episodes(
        sim_cfg_abl, brain_cfg_abl, _N_C1, _STEPS_C1,
        base_seed=3000,
        starting_saturation=1.0,
        starting_health=0.90,
    )

    # Arena coverage over the first 200 steps only (satiation window).
    def _cov_200(ep: List[Dict]) -> float:
        return _arena_coverage([r for r in ep if r["step"] < 200])

    cov_full = [_cov_200(ep) for ep in eps_full]
    cov_abl  = [_cov_200(ep) for ep in eps_abl]

    mean_full = float(np.mean(cov_full))
    mean_abl  = float(np.mean(cov_abl))

    # ── Primary: coverage targets ────────────────────────────────────────────
    assert mean_full > 0.25, (
        f"C1 FAIL — Full architecture mean arena coverage {mean_full:.3f} < 0.25.\n"
        f"  Architectural cause: pe_batch=None → pe_mean=0 → move_forward epistemic\n"
        f"  = 0.60 × w_epistemic, while turns = (area_novelty + 0.60) × w_epistemic.\n"
        f"  At area_novelty > 0, turns always outscore move_forward.  Agent scans\n"
        f"  (turns) in place rather than navigating (move_forward).  Coverage stays\n"
        f"  near 0 because scanning does not change grid cell."
    )
    assert mean_abl < 0.12, (
        f"C1 FAIL — NO_EPISTEMIC ablation mean coverage {mean_abl:.3f} ≥ 0.12.\n"
        f"  With w_epistemic_eff=0 and satiation, idle should dominate: score =\n"
        f"  (1−urgency)×(1−novelty) > 0, all active actions = 0.  Agent stays still."
    )

    # ── Mann-Whitney U: full > ablated ───────────────────────────────────────
    _, p_mwu = mannwhitneyu(cov_full, cov_abl, alternative="greater")
    assert p_mwu < 0.05, (
        f"C1 FAIL — Mann-Whitney U p={p_mwu:.4f} ≥ 0.05.\n"
        f"  Coverage difference not significant: full={mean_full:.3f}, "
        f"ablated={mean_abl:.3f}"
    )

    # ── Idle proportion in full condition ────────────────────────────────────
    all_full_recs = [r for ep in eps_full for r in ep]
    idle_frac = (
        sum(1 for r in all_full_recs if r["policy_id"] == "idle")
        / max(len(all_full_recs), 1)
    )
    assert idle_frac < 0.10, (
        f"C1 FAIL — Idle proportion in full condition {idle_frac:.1%} ≥ 10%.\n"
        f"  Epistemic foraging baseline should keep agent active (turning/moving)\n"
        f"  even at satiation.  Idle should not dominate in novel environments."
    )


# ============================================================================
# C2 — Precision Weighting Adaptation (within-episode prior)
# ============================================================================

_N_C2 = 20
_STEPS_C2 = 200

# HIGH_PRIOR: food visible 15 steps before removal → buffer refreshed 15 times
# LOW_PRIOR:  food visible  2 steps before removal → buffer refreshed  2 times
_TELEPORT_HIGH = 15
_TELEPORT_LOW  = 2


@pytest.mark.behavioral
def test_c2_precision_weighting_adaptation():
    """
    C2: A stronger in-episode prior on food location causes longer directed
    movement after food disappears — the prior-confidence signature.

    Protocol:
      HIGH_PRIOR : food visible for 15 steps before removal (food_removal_step=15)
      LOW_PRIOR  : food visible for  2 steps before removal (food_removal_step=2)
      Measure   : move_forward steps (without food visible) in 30-step window
                  after removal — 'persistence' driven by last-known-food buffer.

    Food is placed at _FOOD_AHEAD (10 units ahead) so the agent approaches from
    step 0.  The hungry agent (sat=0.35) has active food memory (sat_urgency ≈ 0.45
    > threshold 0.30), so the buffer drives forward movement after removal.

    Pass criteria:
      Median persistence: high_prior > low_prior + 5 steps
      Wilcoxon signed-rank test on paired differences: p < 0.05

    Known failure mode (architectural):
      The FEM last_food_angle buffer has no Bayesian confidence update.  Whether
      food was seen once or 15 times, the buffer decays after exactly
      food_memory_decay_steps (30) without a new sighting.  Both conditions
      produce ~30 steps of persistence (the buffer persists regardless of prior
      strength).  The Wilcoxon test will show no significant difference.
    """
    sim_cfg, brain_cfg = _make_cfg(
        sim_overrides={
            "simulation": {"n_food": 0},
            "homeostatic": {"saturation_depletion_rate": 0.001},
        },
        ablation_mode="full",
    )

    def _run_prior_condition(
        removal_step: int,
        base_seed: int,
    ) -> List[int]:
        """Run N episodes, return persistence counts per episode."""
        persistence_counts: List[int] = []
        for ep in range(_N_C2):
            ep_sim = copy.deepcopy(sim_cfg)
            ep_sim["simulation"]["seed"] = base_seed + ep
            recs = _run_episode(
                ep_sim,
                brain_cfg,
                n_steps=_STEPS_C2,
                starting_saturation=0.35,
                starting_health=0.90,
                food_positions_override=[_FOOD_AHEAD_FAR],
                food_removal_step=removal_step,
            )
            p = _persistence_steps(recs, removal_step=removal_step, max_steps=35)
            persistence_counts.append(p)
        return persistence_counts

    # Same base_seed so episode i sees the same arena layout in both conditions.
    high_latencies = _run_prior_condition(_TELEPORT_HIGH, base_seed=4000)
    low_latencies  = _run_prior_condition(_TELEPORT_LOW,  base_seed=4000)

    med_high = float(np.median(high_latencies))
    med_low  = float(np.median(low_latencies))

    # ── Persistence difference ───────────────────────────────────────────────
    assert med_high > med_low + 5, (
        f"C2 FAIL — High-prior persistence ({med_high:.1f} steps) not meaningfully "
        f"greater than low-prior ({med_low:.1f} steps; expected > +5 steps).\n"
        f"  Architectural cause: FEM last_food_angle buffer has no confidence\n"
        f"  parameter — both conditions decay over exactly food_memory_decay_steps\n"
        f"  (30) regardless of how many times food was observed before removal."
    )

    # ── Wilcoxon signed-rank test (paired) ───────────────────────────────────
    diffs = [h - l for h, l in zip(high_latencies, low_latencies)]
    if any(d != 0 for d in diffs):
        _, p_wilcox = wilcoxon(high_latencies, low_latencies, alternative="greater")
    else:
        p_wilcox = 1.0

    assert p_wilcox < 0.05, (
        f"C2 FAIL — Wilcoxon p={p_wilcox:.4f} ≥ 0.05.\n"
        f"  High-prior and low-prior persistence distributions are statistically\n"
        f"  indistinguishable.  Both conditions show ~{min(med_high, med_low):.0f} "
        f"steps of persistence because the recency buffer has no confidence weight."
    )


# ============================================================================
# C3a — Drive-Conditioned Goal Persistence (hungry agent, food removed)
# ============================================================================

_N_C3A = 25
_STEPS_C3A = 120
_FOOD_REMOVAL_STEP_C3A = 5   # Remove food before agent can eat it (dist 10, takes ~10 steps)


@pytest.mark.behavioral
def test_c3a_goal_persistence():
    """
    C3a: When hungry and food disappears mid-approach, the full architecture
    continues directed movement (move_forward without food) toward the last-known
    food heading for substantially more steps than a no-memory baseline.

    Setup: food placed at _FOOD_AHEAD (10 units ahead) so it is visible from
    step 0.  Agent is hungry (sat=0.35, sat_urgency ≈ 0.45 > threshold 0.30),
    so the food memory is active.  Food is removed at step 5 (before the agent
    can eat it — the approach takes ~10 steps at step_size=1.0).

    Full architecture: buffer decay=30 steps → continues forward for ~25 steps.
    No-memory baseline: decay=1 step → buffer expires after step 6 → stops.

    Persistence metric: move_forward steps without food visible in the 35-step
    window after food removal.

    Pass criteria (TEST_PROCEDURES.md §C3a):
      Mean persistence: full > 15 steps,  no-memory < 3 steps
      % trials persistence > 10 steps: full > 55%,  no-memory < 10%
      Mann-Whitney U: full > no-memory, p < 0.05
    """
    sim_cfg, brain_cfg_full = _make_cfg(
        sim_overrides={
            "simulation": {"n_food": 0},
            "homeostatic": {"saturation_depletion_rate": 0.001},
        },
        ablation_mode="full",
    )

    # No-memory baseline: decay=1 means buffer expires after 1 step without sighting.
    brain_cfg_nomem = copy.deepcopy(brain_cfg_full)
    brain_cfg_nomem.setdefault("policy_generator", {})
    brain_cfg_nomem["policy_generator"]["food_memory_decay_steps"] = 1

    def _run_c3a_episodes(brain_cfg: Dict, base_seed: int) -> List[int]:
        counts: List[int] = []
        for ep in range(_N_C3A):
            ep_sim = copy.deepcopy(sim_cfg)
            ep_sim["simulation"]["seed"] = base_seed + ep
            recs = _run_episode(
                ep_sim,
                brain_cfg,
                n_steps=_STEPS_C3A,
                starting_saturation=0.35,
                starting_health=0.90,
                food_positions_override=[_FOOD_AHEAD_FAR],
                food_removal_step=_FOOD_REMOVAL_STEP_C3A,
            )
            p = _persistence_steps(
                recs,
                removal_step=_FOOD_REMOVAL_STEP_C3A,
                max_steps=35,
            )
            counts.append(p)
        return counts

    pers_full  = _run_c3a_episodes(brain_cfg_full,  base_seed=5000)
    pers_nomem = _run_c3a_episodes(brain_cfg_nomem, base_seed=5000)

    mean_full  = float(np.mean(pers_full))
    mean_nomem = float(np.mean(pers_nomem))
    pct_full   = sum(1 for p in pers_full  if p > 10) / _N_C3A
    pct_nomem  = sum(1 for p in pers_nomem if p > 10) / _N_C3A

    # ── Mean persistence targets ─────────────────────────────────────────────
    assert mean_full > 15, (
        f"C3a FAIL — Full architecture mean persistence {mean_full:.1f} steps < 15.\n"
        f"  At sat=0.35, sat_urgency ≈ 0.45 > 0.30 threshold → memory bonus fires:\n"
        f"  move_forward gets mem_bonus = (1−last_dist) × 0.60 × 0.4 × sat_urgency.\n"
        f"  Buffer should sustain forward movement for ~food_memory_decay_steps (30).\n"
        f"  Check that food at {_FOOD_AHEAD_FAR} is visible at step 0 (state rebuild\n"
        f"  required after food_positions_override) and that sat_urgency > 0.30."
    )
    assert mean_nomem < 3, (
        f"C3a FAIL — No-memory baseline mean persistence {mean_nomem:.1f} steps ≥ 3.\n"
        f"  With food_memory_decay_steps=1, buffer expires after step {_FOOD_REMOVAL_STEP_C3A + 1}.\n"
        f"  move_forward should lose to turns immediately (no urgency-scaled memory bonus)."
    )

    # ── % trials with > 10 steps persistence ────────────────────────────────
    assert pct_full > 0.55, (
        f"C3a FAIL — Full: {pct_full:.0%} trials with persistence > 10 steps; "
        f"expected > 55%."
    )
    assert pct_nomem < 0.10, (
        f"C3a FAIL — No-memory: {pct_nomem:.0%} trials with persistence > 10 steps; "
        f"expected < 10%."
    )

    # ── Mann-Whitney U ────────────────────────────────────────────────────────
    _, p_mwu = mannwhitneyu(pers_full, pers_nomem, alternative="greater")
    assert p_mwu < 0.05, (
        f"C3a FAIL — Mann-Whitney U p={p_mwu:.4f} ≥ 0.05.\n"
        f"  Full mean={mean_full:.1f}  No-memory mean={mean_nomem:.1f}"
    )


# ============================================================================
# C3b — Satiation Suppression (satiated agent ignores visible food)
# ============================================================================

_N_C3B = 20
_STEPS_C3B = 80
_SAT_WINDOW = 50   # steps to observe


@pytest.mark.behavioral
def test_c3b_satiation_suppression():
    """
    C3b: When satiated (saturation ≥ setpoint 0.70), the agent does not approach
    food visible in the forward ray.

    Mechanism: sat_urgency = 0 when sat ≥ setpoint.  This zeros the food-proximity
    bonus for move_forward (proximity × food_prox_bonus × sat_urgency = 0).
    Idle (score ≈ (1−urgency)×(1−novelty) at satiation) wins OR, when food is
    visible and suppresses idle (idle -= 0.6), epistemic turns win — either way
    the agent oscillates/stays still rather than approaching.

    Food at _FOOD_AHEAD (10 units, forward ray) is visible from step 0.
    Saturation=1.0, depletion_rate=0.0005 keeps sat > setpoint for 600+ steps.

    Pass criterion: > 50% of episodes show no move_forward toward visible food
    in the first 50 steps.

    'Approaching' = policy_id == 'move_forward' AND food_in_ray at any step in
    the first 50 steps.  One such step is enough to fail suppression for that
    episode.
    """
    sim_cfg, brain_cfg = _make_cfg(
        sim_overrides={
            "simulation": {"n_food": 0},
            "homeostatic": {"saturation_depletion_rate": 0.0005},
        },
        ablation_mode="full",
    )

    suppressed_count = 0
    for ep in range(_N_C3B):
        ep_sim = copy.deepcopy(sim_cfg)
        ep_sim["simulation"]["seed"] = 6000 + ep
        recs = _run_episode(
            ep_sim,
            brain_cfg,
            n_steps=_STEPS_C3B,
            starting_saturation=1.0,
            starting_health=0.90,
            food_positions_override=[_FOOD_AHEAD_NEAR],   # 2.5 units: visible 50+ steps
        )
        # Suppression: no move_forward toward visible food in first 50 steps.
        approached = any(
            r["policy_id"] == "move_forward" and r["food_in_ray"]
            for r in recs
            if r["step"] < _SAT_WINDOW
        )
        if not approached:
            suppressed_count += 1

    suppression_rate = suppressed_count / _N_C3B

    assert suppression_rate > 0.50, (
        f"C3b FAIL — Satiation suppression rate {suppression_rate:.0%} < 50%.\n"
        f"  At sat=1.0, sat_urgency=0 → food-proximity bonus for move_forward = 0.\n"
        f"  Idle or epistemic turns should dominate, preventing food approach.\n"
        f"  If food at {_FOOD_AHEAD_NEAR} is NOT visible from step 0, check that\n"
        f"  food_positions_override triggers state rebuild (adapter._build_state).\n"
        f"  Current suppression counts: {suppressed_count}/{_N_C3B} episodes."
    )
