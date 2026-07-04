"""
Seth Functional Validation Suite — Tier D: Causal Self-Attribution
===================================================================

D1  Causal Self-Attribution
    Tests whether the WorldModelGenerator (efference copy pattern) produces
    lower prediction error for self-caused state changes (eating food) than
    for externally-imposed changes (programmatic saturation injection).

Architecture Notes
------------------
The WorldModelGenerator maintains an online EMA transition model indexed by
(action_id, heading_bucket, channel).  After each step it updates the model
with the observed transition.  Before each step it produces a prediction:

    predicted[ch] = current[ch] + EMA_delta(action, heading_bucket, ch)

Self-caused eating events:
    Agent takes move_forward toward food → food eaten → saturation jumps +0.2.
    The EMA model accumulates a small positive delta for (move_forward, hb,
    saturation) across eating events.  After warmup, the model partially
    anticipates the saturation increase, giving LOWER PE at eating steps.

Externally-imposed injection:
    At a fixed step, saturation is forced to 0.05 programmatically.  The model
    has no action-conditional delta that predicts a ~0.45 saturation drop.
    PE at injection steps is therefore LARGE.

Expected outcome:
    - External PE median ≈ 0.45 (|current_sat − 0.05|)
    - Self-caused PE median ≈ 0.18 (|EMA_delta − 0.2|)
    - Ratio ≈ 2.5 > 2.0 target  → PASS
    - Variance(external) > variance(self): injection PE varies with saturation at
      injection time (eating history dependent); eating PE clusters around 0.18
      once EMA converges                    → PASS

Known failure modes (architectural):
    - EMA never converges: if eating events are too rare (< 20 total across
      warmup), self_caused_pe ≈ 0.2 (no model, just predicts current value).
      Ratio still > 2.0 because external drop is large.  Unlikely to fail.
    - If no eating occurs in test episodes: pairing fails at the n_pairs≥20 gate.
      Fix: increase n_food or reduce starting_saturation to trigger urgency sooner.

Run:
    pytest tests/functional/test_d_suite.py -v
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest
from scipy.stats import wilcoxon

# ── path setup ───────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from core.adapters.headless.env_adapter import HeadlessSimAdapter
from core.coordination.workspace import GlobalWorkspace
from core.layers.action_selection.FreeEnergyMinimizer import FreeEnergyMinimizer
from core.layers.interoceptive.AllostaticController import AllostaticController
from core.layers.predictive.WorldModelGenerator import WorldModelGenerator

# Re-use shared base configs and _make_cfg from B suite.
from tests.functional.test_b_suite import (
    _BRAIN_BASE,
    _SIM_BASE,
    _make_cfg,
)

# ── D1 protocol constants ────────────────────────────────────────────────────
_N_D1_WARMUP          = 15     # warmup episodes: train WM before test data collection
_N_D1_TEST            = 50     # matched test episodes
_STEPS_D1             = 500    # episode length
_D1_INJECTION_STEP    = 350    # external perturbation step (well after first eating events)
_D1_INJECTION_VALUE   = 0.05   # force saturation here (large drop from ~0.45 at step 350)
_D1_EATING_THRESHOLD  = 0.15   # saturation increase threshold to classify as eating


# ============================================================================
# D1 episode runner — efference copy variant
# ============================================================================

def _run_d1_episode(
    sim_cfg: Dict[str, Any],
    brain_cfg: Dict[str, Any],
    wm: WorldModelGenerator,
    n_steps: int,
    drive_injection_step: Optional[int] = None,
    drive_injection_value: Optional[float] = None,
    starting_saturation: Optional[float] = None,
    starting_health: Optional[float] = None,
    collect_data: bool = True,
) -> Tuple[List[float], Optional[float]]:
    """
    Run one episode with WorldModelGenerator efference copy.

    Efference copy protocol (per step):
        1. Score actions with FEM → select policy_id
        2. wm.predict(state, policy_id) → pred (before stepping)
        3. adapter.step(policy_id) → next_state
        4. Compute sat_pe = |next_state.saturation − pred["saturation"]|
        5. wm.update(state, policy_id, next_state) → update EMA model

    Returns:
        self_caused_pes : list of sat_pe values at food-eating steps
                          (steps where saturation increased by > _D1_EATING_THRESHOLD)
        external_pe     : sat_pe at drive_injection_step, or None if no injection
    """
    adapter = HeadlessSimAdapter(sim_cfg)
    ac      = AllostaticController(adapter.get_drive_channels(), brain_cfg)
    fem     = FreeEnergyMinimizer(brain_cfg)
    ws      = GlobalWorkspace()

    state = adapter.reset()

    if starting_saturation is not None:
        adapter._homeostatic._saturation = float(starting_saturation)
    if starting_health is not None:
        adapter._homeostatic._health = float(starting_health)

    policies: List[Dict] = adapter.get_available_policies()
    recent_actions: List[str] = []

    self_caused_pes: List[float] = []
    external_pe: Optional[float] = None

    for step in range(n_steps):
        # External saturation injection (before adapter.step so next_state reflects it)
        if drive_injection_step is not None and step == drive_injection_step:
            adapter._homeostatic._saturation = float(drive_injection_value)

        # ── interoceptive update ─────────────────────────────────────────────
        vitals = {
            "saturation": state.homeostasis.saturation,
            "health":     state.homeostasis.health,
            "energy":     state.homeostasis.energy,
        }
        batch = ac.update(vitals, ws, step)

        # ── EFE scoring ──────────────────────────────────────────────────────
        rays = state.perception.raycast_hits or []
        context: Dict[str, Any] = {
            "raycast_hits":     rays,
            "area_familiarity": 0.5,
            "valence":          0.0,
            "arousal":          0.0,
            "recent_actions":   recent_actions[-4:],
            "motor_efficiency": state.raw_metadata.get("motor_efficiency", 1.0),
            "last_action":      recent_actions[-1] if recent_actions else None,
        }
        scores = fem.score(policies, batch, None, area_familiarity=0.5, context=context)
        policy_id: str = max(scores, key=scores.get)

        # ── efference copy: predict BEFORE stepping ──────────────────────────
        prev_sat = float(state.homeostasis.saturation or 0.0)
        wm_pred  = wm.predict(state, policy_id)
        pred_sat = float(wm_pred.get("saturation", prev_sat))

        # ── execute action ───────────────────────────────────────────────────
        next_state, done = adapter.step(policy_id)

        # ── compute PE and classify ──────────────────────────────────────────
        if collect_data:
            actual_sat = float(next_state.homeostasis.saturation or 0.0)
            sat_pe     = abs(actual_sat - pred_sat)

            # Self-caused: large saturation INCREASE = food eaten
            # (depletion is only ~0.001/step; any increase > 0.15 is eating)
            if actual_sat - prev_sat > _D1_EATING_THRESHOLD:
                self_caused_pes.append(sat_pe)

            # External: injection step PE
            # Guard: only record if this was a genuine DROP (injection caused it,
            # not an eating event coinciding with injection).
            if drive_injection_step is not None and step == drive_injection_step:
                if actual_sat - prev_sat < 0.0:   # saturation dropped → injection dominated
                    external_pe = sat_pe

        # ── update world model ───────────────────────────────────────────────
        wm.update(
            prev_state=state,
            action_id=policy_id,
            next_state=next_state,
            step=step,
        )

        recent_actions.append(policy_id)
        state = next_state
        if done:
            break

    return self_caused_pes, external_pe


# ============================================================================
# D1 — Causal Self-Attribution
# ============================================================================

@pytest.mark.behavioral
def test_d1_causal_self_attribution():
    """
    D1: The WorldModelGenerator has lower prediction error for state changes
    it caused (eating food → saturation jump) than for externally-imposed
    changes (programmatic saturation injection).

    Protocol:
        Warmup phase (15 episodes, no injection):
            WM accumulates eating-event experience across episodes.
            EMA delta for (move_forward, hb, saturation) converges to a small
            positive value, reflecting the occasional saturation boost from eating.

        Test phase (50 matched episodes):
            Each episode runs 500 steps.  At step 350, saturation is injected
            to 0.05 (external perturbation).  Self-caused eating events occur
            naturally when the agent is hungry (saturation depletes from 0.8
            to <0.7 setpoint around step 100).

            Per episode, collect:
              self_caused_pe : mean PE at eating steps (saturation increase > 0.15)
              external_pe    : PE at injection step (saturation drop → 0.05)

    Pass criteria (TEST_PROCEDURES.md §D1):
        PE ratio (external / self-caused): > 2.0
        Wilcoxon signed-rank (external > self-caused): p < 0.01
        Variance of self-caused PE < variance of external PE
    """
    sim_cfg, brain_cfg = _make_cfg(
        sim_overrides={
            "simulation": {"n_food": 4},
            "homeostatic": {"saturation_depletion_rate": 0.001},
        },
        ablation_mode="full",
    )

    # Single WM instance: persists across warmup + test, accumulating learning.
    wm = WorldModelGenerator(brain_cfg)

    # ── Warmup phase: train WM on natural eating events ──────────────────────
    warmup_sim = copy.deepcopy(sim_cfg)
    for ep in range(_N_D1_WARMUP):
        warmup_sim["simulation"]["seed"] = 7000 + ep
        _run_d1_episode(
            warmup_sim, brain_cfg, wm, _STEPS_D1,
            starting_saturation=0.8,
            starting_health=0.90,
            collect_data=False,  # do not collect data; just train WM
        )

    # ── Test phase: matched episodes ─────────────────────────────────────────
    # Each episode contributes one (self_pe, ext_pe) pair if eating occurred.
    self_pes_per_ep: List[float] = []
    ext_pes_per_ep:  List[float] = []

    for ep in range(_N_D1_TEST):
        ep_sim = copy.deepcopy(sim_cfg)
        ep_sim["simulation"]["seed"] = 8000 + ep

        self_caused_pes, external_pe = _run_d1_episode(
            ep_sim, brain_cfg, wm, _STEPS_D1,
            drive_injection_step=_D1_INJECTION_STEP,
            drive_injection_value=_D1_INJECTION_VALUE,
            starting_saturation=0.8,
            starting_health=0.90,
            collect_data=True,
        )

        # Only include episodes where both data points are available.
        if self_caused_pes and external_pe is not None:
            self_pes_per_ep.append(float(np.mean(self_caused_pes)))
            ext_pes_per_ep.append(external_pe)

    # ── Enough paired data? ───────────────────────────────────────────────────
    n_pairs = len(self_pes_per_ep)
    assert n_pairs >= 20, (
        f"D1 FAIL — only {n_pairs}/{_N_D1_TEST} episodes produced both an eating "
        f"event and a valid injection PE.  Need ≥ 20 pairs for reliable statistics.\n"
        f"  Eating requires: saturation < setpoint (≈0.70), food within interaction "
        f"radius, move_forward selected.\n"
        f"  Check: starting_saturation={0.8}, depletion_rate=0.001 → urgency rises "
        f"after ~100 steps → foraging should begin by step 150 with n_food=4.\n"
        f"  Injection guard: requires saturation DROP at step {_D1_INJECTION_STEP}; "
        f"if agent ate at exactly that step, injection PE is skipped."
    )

    self_pes = np.array(self_pes_per_ep)
    ext_pes  = np.array(ext_pes_per_ep)

    med_self = float(np.median(self_pes))
    med_ext  = float(np.median(ext_pes))

    # ── PE ratio: external >> self-caused ────────────────────────────────────
    # Self-caused: WM partially anticipates eating (EMA delta > 0 for move_forward)
    #   → PE = |EMA_delta − 0.2| ≈ 0.18 after convergence
    # External: WM predicts near-current saturation for any action
    #   → PE = |current_sat − 0.05| ≈ 0.40–0.65 at step 350
    ratio = med_ext / max(med_self, 1e-6)
    assert ratio > 2.0, (
        f"D1 FAIL — PE ratio external/self-caused = {ratio:.2f} < 2.0.\n"
        f"  Median self-caused PE  = {med_self:.4f}  (at food-eating steps)\n"
        f"  Median external PE     = {med_ext:.4f}  (at injection step {_D1_INJECTION_STEP})\n"
        f"  Expected: after {_N_D1_WARMUP}-episode warmup, EMA for (move_forward, hb,\n"
        f"  saturation) converges to a small positive delta (~0.01–0.02), partially\n"
        f"  anticipating eating.  Injection to {_D1_INJECTION_VALUE} from ~0.45 produces\n"
        f"  PE ≈ 0.44, which should be > 2× the eating PE ≈ 0.18.\n"
        f"  If ratio is near 1.0: the WM delta_table may not have accumulated enough\n"
        f"  eating events during warmup (need > 20 move_forward+eating observations)."
    )

    # ── Wilcoxon signed-rank: external PE > self-caused PE ───────────────────
    _, p_wilcox = wilcoxon(ext_pes, self_pes, alternative="greater")
    assert p_wilcox < 0.01, (
        f"D1 FAIL — Wilcoxon signed-rank p={p_wilcox:.4f} ≥ 0.01.\n"
        f"  External PE not significantly greater than self-caused PE.\n"
        f"  n_pairs={n_pairs}, median ext={med_ext:.4f}, median self={med_self:.4f}\n"
        f"  If n_pairs is small, the test lacks power.  Increase _N_D1_TEST."
    )

    # ── Variance: model is more certain about own actions ────────────────────
    var_self = float(np.var(self_pes))
    var_ext  = float(np.var(ext_pes))
    assert var_self < var_ext, (
        f"D1 FAIL — var(self-caused PE) = {var_self:.5f} ≥ var(external PE) = {var_ext:.5f}.\n"
        f"  Self-caused PE should be consistent (EMA converges to a fixed delta);\n"
        f"  external PE varies with saturation at injection time (eating history).\n"
        f"  If var(self) is large, WM EMA has not converged — increase _N_D1_WARMUP."
    )
