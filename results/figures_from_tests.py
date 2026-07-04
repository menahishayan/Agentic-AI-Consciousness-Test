#!/usr/bin/env python3
"""
Generate §7.5-7.6 figures from the FUNCTIONAL TEST pipeline — the real source
of those results — not from src/logs/runs (the tests never persist there).

Each figure function imports the suite's own helpers and reproduces the exact
protocol (same constants, same seeds), so the plotted data is identical to what
pytest computes. It also prints the statistics so you can confirm they match
both the test's pass criteria AND the specific numbers quoted in the paper.

Run from the repo root (same cwd you run pytest from):
    python results/figures_from_tests.py d1 --outdir results/figures

If the script isn't under <repo>/results/, pass --root explicitly:
    python figures_from_tests.py d1 --root /Users/shayan/Desktop/Projects/Capstone
"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np


def _setup_paths(root: str):
    root_p = Path(root).resolve()
    for p in (str(root_p), str(root_p / "src")):
        if p not in sys.path:
            sys.path.insert(0, p)


def d1(outdir: str):
    """D1 boxplot + histogram of self-caused vs external PE (Fig 7.x)."""
    from scipy.stats import wilcoxon
    from core.layers.predictive.WorldModelGenerator import WorldModelGenerator
    from tests.functional.test_d_suite import (
        _run_d1_episode,
        _N_D1_WARMUP, _N_D1_TEST, _STEPS_D1,
        _D1_INJECTION_STEP, _D1_INJECTION_VALUE,
    )
    from tests.functional.test_b_suite import _make_cfg

    # Mirror test_d1_causal_self_attribution exactly.
    sim_cfg, brain_cfg = _make_cfg(
        sim_overrides={
            "simulation": {"n_food": 4},
            "homeostatic": {"saturation_depletion_rate": 0.001},
        },
        ablation_mode="full",
    )

    wm = WorldModelGenerator(brain_cfg)

    # Warmup (train WM; no data collection) — seeds 7000+ep, matching the test.
    warmup = copy.deepcopy(sim_cfg)
    for ep in range(_N_D1_WARMUP):
        warmup["simulation"]["seed"] = 7000 + ep
        _run_d1_episode(warmup, brain_cfg, wm, _STEPS_D1,
                        starting_saturation=0.8, starting_health=0.90,
                        collect_data=False)

    # Test phase — seeds 8000+ep, matching the test.
    self_pes, ext_pes = [], []
    for ep in range(_N_D1_TEST):
        s = copy.deepcopy(sim_cfg)
        s["simulation"]["seed"] = 8000 + ep
        scp, ext = _run_d1_episode(
            s, brain_cfg, wm, _STEPS_D1,
            drive_injection_step=_D1_INJECTION_STEP,
            drive_injection_value=_D1_INJECTION_VALUE,
            starting_saturation=0.8, starting_health=0.90,
            collect_data=True,
        )
        if scp and ext is not None:
            self_pes.append(float(np.mean(scp)))
            ext_pes.append(float(ext))

    self_pes = np.array(self_pes)
    ext_pes = np.array(ext_pes)
    n_pairs = len(self_pes)
    if n_pairs < 2:
        sys.exit(f"D1: only {n_pairs} paired episodes — cannot plot.")

    med_self, med_ext = float(np.median(self_pes)), float(np.median(ext_pes))
    ratio = med_ext / max(med_self, 1e-6)
    var_self, var_ext = float(np.var(self_pes)), float(np.var(ext_pes))
    _, p_w = wilcoxon(ext_pes, self_pes, alternative="greater")

    stats = {
        "n_pairs": n_pairs,
        "median_self": med_self, "median_external": med_ext,
        "ratio_external_over_self": ratio,
        "var_self": var_self, "var_external": var_ext,
        "wilcoxon_p": float(p_w),
    }

    os.makedirs(outdir, exist_ok=True)
    # tidy table for the figure
    with open(os.path.join(outdir, "d1_pe_paired.csv"), "w") as f:
        f.write("episode,self_caused_pe,external_pe\n")
        for i in range(n_pairs):
            f.write(f"{i},{self_pes[i]:.6f},{ext_pes[i]:.6f}\n")

    # figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
        bp = ax1.boxplot([self_pes, ext_pes], tick_labels=["Self-caused", "External"],
                         widths=0.55, showmeans=True, patch_artist=True)
        for patch, c in zip(bp["boxes"], ["#8FBFD4", "#D48F8F"]):
            patch.set_facecolor(c); patch.set_alpha(0.75)
        for i, arr in enumerate([self_pes, ext_pes], start=1):
            ax1.scatter(np.random.normal(i, 0.05, len(arr)), arr, s=14,
                        color="#222", alpha=0.5, zorder=3)
        ax1.set_ylabel("Prediction error  |actual − predicted saturation|")
        ax1.set_title("D1: PE by cause")

        hi = max(ext_pes.max(), self_pes.max()) * 1.05
        bins = np.linspace(0, hi, 20)
        ax2.hist(self_pes, bins=bins, alpha=0.6, color="#8FBFD4", label="Self-caused")
        ax2.hist(ext_pes, bins=bins, alpha=0.6, color="#D48F8F", label="External")
        ax2.set_xlabel("Prediction error"); ax2.set_ylabel("Episodes")
        ax2.set_title("Distribution"); ax2.legend()

        fig.suptitle(f"ratio={ratio:.2f}  var(ext)={var_ext:.3f}>var(self)={var_self:.3f}  "
                     f"Wilcoxon p={p_w:.2e}  (n={n_pairs} paired episodes)", fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(os.path.join(outdir, "fig_7x_d1_pe_distribution.png"), dpi=200)
    except ImportError:
        print("matplotlib not installed — wrote CSV + stats only. "
              "pip install matplotlib to render the figure.")

    print(json.dumps(stats, indent=2))
    print(f"\nCompare to paper §7.6.1: ratio ~2.2, Wilcoxon p<0.01, var(ext)>var(self).")
    print(f"Wrote: {outdir}/d1_pe_paired.csv  and  fig_7x_d1_pe_distribution.png")


def _grouped_figure(groups: dict, ylabel: str, title: str, outpath: str,
                    hline: float = None, hline_label: str = ""):
    """Bar (mean ± SEM) + per-episode dots for 2-3 conditions."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping figure, CSV written.")
        return
    names = list(groups)
    palette = ["#8FBFD4", "#D48F8F", "#A8C6A1", "#C6A1C6"]
    fig, ax = plt.subplots(figsize=(1.6 * len(names) + 2, 4.5))
    for i, name in enumerate(names):
        arr = np.asarray(groups[name], dtype=float)
        m = arr.mean()
        sem = arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
        ax.bar(i, m, yerr=sem, width=0.55, capsize=5, color=palette[i % len(palette)],
               alpha=0.75, edgecolor="#333", zorder=2)
        ax.scatter(np.random.normal(i, 0.05, len(arr)), arr, s=26, color="#222",
                   alpha=0.6, zorder=3)
    if hline is not None:
        ax.axhline(hline, ls="--", lw=1, color="#666")
        ax.text(len(names) - 0.5, hline, f" {hline_label}", fontsize=8,
                color="#666", va="bottom", ha="right")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f"{n}\n(n={len(groups[n])})" for n in names])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)


def b1(outdir: str):
    """B1 anticipatory foraging rate, full vs reactive (Fig 7.x)."""
    from scipy.stats import mannwhitneyu
    from tests.functional.test_b_suite import (
        _run_n_episodes, _make_cfg, _b1_anticipatory_rate, _N_B1, _STEPS_B1,
    )
    ov = {"simulation": {"n_food": 4}, "homeostatic": {"saturation_depletion_rate": 0.001}}
    sf, bf = _make_cfg(sim_overrides=ov, ablation_mode="full")
    sr, br = _make_cfg(sim_overrides=ov, ablation_mode="reactive")
    eps_full = _run_n_episodes(sf, bf, _N_B1, _STEPS_B1, base_seed=1000, starting_health=0.90)
    eps_react = _run_n_episodes(sr, br, _N_B1, _STEPS_B1, base_seed=1000, starting_health=0.90)
    rf = np.array(_b1_anticipatory_rate(eps_full))
    rr = np.array(_b1_anticipatory_rate(eps_react))
    _, p = mannwhitneyu(rf, rr, alternative="greater")

    # Per-episode diagnostic: distinguish "foraged reactively" (rate 0, but
    # food-seeking steps > 0) from "never foraged" (0 food-seeking steps, rate
    # undefined -> scored 0 by the helper).
    print("B1 full-condition per-episode diagnostic:")
    print(f"  {'seed':>5} {'steps':>6} {'food_seek':>9} {'antic_rate':>10}  note")
    for i, ep in enumerate(eps_full):
        fs = [r for r in ep if r["policy_id"] == "move_forward" and r["food_in_ray"]]
        note = "no foraging (rate undefined)" if not fs else ""
        print(f"  {1000 + i:>5} {len(ep):>6} {len(fs):>9} {rf[i]:>10.3f}  {note}")

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "b1_anticipatory_rates.csv"), "w") as fh:
        fh.write("condition,episode,anticipatory_rate\n")
        for i, v in enumerate(rf):
            fh.write(f"full,{i},{v:.6f}\n")
        for i, v in enumerate(rr):
            fh.write(f"reactive,{i},{v:.6f}\n")
    _grouped_figure(
        {"Full": rf, "Reactive": rr},
        "Anticipatory rate (food-seeking steps at sat > 0.35)",
        f"B1: Allostatic vs. Reactive   (MWU p = {p:.2e}, n={_N_B1} matched)",
        os.path.join(outdir, "fig_7x_b1_anticipatory.png"),
        hline=0.50, hline_label="0.50 target",
    )
    print(json.dumps({
        "mean_full": float(rf.mean()), "mean_reactive": float(rr.mean()),
        "median_full": float(np.median(rf)), "median_reactive": float(np.median(rr)),
        "mannwhitney_p": float(p), "n_episodes": _N_B1,
        "full_rates": rf.round(4).tolist(), "reactive_rates": rr.round(4).tolist(),
    }, indent=2))
    print("\nUse mean_full / mean_reactive to replace the '100% / 0%' figures in §7.5.1.")


def coverage(outdir: str):
    """C1 + E2 arena coverage at step 200: full vs no_epistemic vs efe_only."""
    from scipy.stats import mannwhitneyu
    from tests.functional.test_b_suite import _run_n_episodes, _make_cfg
    from tests.functional.test_c_suite import _arena_coverage, _N_C1, _STEPS_C1

    def cov200(ep):
        return _arena_coverage([r for r in ep if r["step"] < 200])

    ov = {"simulation": {"n_food": 4}, "homeostatic": {"saturation_depletion_rate": 0.0005}}
    out = {}
    for mode in ("full", "no_epistemic", "efe_only"):
        s, b = _make_cfg(sim_overrides=ov, ablation_mode=mode)
        eps = _run_n_episodes(s, b, _N_C1, _STEPS_C1, base_seed=3000,
                              starting_saturation=1.0, starting_health=0.90)
        out[mode] = np.array([cov200(e) for e in eps])

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "c1_e2_coverage.csv"), "w") as fh:
        fh.write("condition,episode,coverage_at_200\n")
        for mode, arr in out.items():
            for i, v in enumerate(arr):
                fh.write(f"{mode},{i},{v:.6f}\n")
    _grouped_figure(
        {"Full": out["full"], "No-epistemic": out["no_epistemic"], "EFE-only": out["efe_only"]},
        "Arena coverage at step 200 (fraction of cells)",
        "C1 / E2: Epistemic exploration coverage",
        os.path.join(outdir, "fig_7x_c1_e2_coverage.png"),
        hline=0.25, hline_label="0.25 C1 target",
    )
    _, p_c1 = mannwhitneyu(out["full"], out["no_epistemic"], alternative="greater")
    print(json.dumps({
        "mean_full": float(out["full"].mean()),
        "mean_no_epistemic": float(out["no_epistemic"].mean()),
        "mean_efe_only": float(out["efe_only"].mean()),
        "c1_full_vs_no_epistemic_mwu_p": float(p_c1),
        "e2_full_minus_efe_only": float(out["full"].mean() - out["efe_only"].mean()),
    }, indent=2))
    print("\nC1: note full coverage is expected < 0.25 (pe_batch=None infra fail) but > no_epistemic.")
    print("E2: full ≈ efe_only (diff should be < 0.03) confirms EFE-structural exploration.")


def d1_control(outdir: str):
    """
    D1 mechanism control: rerun the exact D1 protocol with the efference copy
    DISABLED (world model predicts 'no change' = current saturation), and
    compare self-caused PE with vs without the mechanism.

    Logic:
      - If self-PE is meaningfully LOWER with the real WM than with the null
        model, the efference copy is anticipating self-caused change -> the
        mechanism claim in section 7.6.1 is supported.
      - If self-PE is the SAME or HIGHER with the real WM, the ratio is a
        perturbation-magnitude artifact (eating = +0.20 bounded vs injection =
        large drop), NOT self-attribution -> the mechanism claim must be dropped.
    """
    import copy
    from scipy.stats import wilcoxon
    from core.layers.predictive.WorldModelGenerator import WorldModelGenerator
    from tests.functional.test_d_suite import (
        _run_d1_episode,
        _N_D1_WARMUP, _N_D1_TEST, _STEPS_D1,
        _D1_INJECTION_STEP, _D1_INJECTION_VALUE,
    )
    from tests.functional.test_b_suite import _make_cfg

    class NullWorldModel:
        """Predicts no change (current saturation); learns nothing."""
        def predict(self, state, policy_id):
            return {"saturation": float(state.homeostasis.saturation or 0.0)}
        def update(self, prev_state, action_id, next_state, step):
            pass

    sim_cfg, brain_cfg = _make_cfg(
        sim_overrides={"simulation": {"n_food": 4},
                       "homeostatic": {"saturation_depletion_rate": 0.001}},
        ablation_mode="full",
    )

    def run_test_phase(wm):
        self_pes, ext_pes = [], []
        for ep in range(_N_D1_TEST):
            s = copy.deepcopy(sim_cfg)
            s["simulation"]["seed"] = 8000 + ep
            scp, ext = _run_d1_episode(
                s, brain_cfg, wm, _STEPS_D1,
                drive_injection_step=_D1_INJECTION_STEP,
                drive_injection_value=_D1_INJECTION_VALUE,
                starting_saturation=0.8, starting_health=0.90, collect_data=True)
            if scp and ext is not None:
                self_pes.append(float(np.mean(scp)))
                ext_pes.append(float(ext))
        return np.array(self_pes), np.array(ext_pes)

    # Real WM (with warmup, exactly as the D1 test does)
    wm = WorldModelGenerator(brain_cfg)
    warmup = copy.deepcopy(sim_cfg)
    for ep in range(_N_D1_WARMUP):
        warmup["simulation"]["seed"] = 7000 + ep
        _run_d1_episode(warmup, brain_cfg, wm, _STEPS_D1,
                        starting_saturation=0.8, starting_health=0.90, collect_data=False)
    self_real, ext_real = run_test_phase(wm)

    # Null WM (no efference copy, no warmup needed)
    self_null, ext_null = run_test_phase(NullWorldModel())

    def stat(self_pes, ext_pes):
        return {
            "n": len(self_pes),
            "median_self": float(np.median(self_pes)),
            "median_external": float(np.median(ext_pes)),
            "ratio": float(np.median(ext_pes) / max(np.median(self_pes), 1e-6)),
            "var_self": float(np.var(self_pes)),
        }

    result = {"with_efference_copy": stat(self_real, ext_real),
              "no_efference_copy": stat(self_null, ext_null)}
    # Paired comparison of self-PE: does the WM reduce self-caused PE?
    n = min(len(self_real), len(self_null))
    if n >= 2:
        _, p = wilcoxon(self_null[:n], self_real[:n], alternative="greater")
        result["self_pe_reduced_by_wm_p"] = float(p)   # p<0.05 => WM lowers self-PE

    os.makedirs(outdir, exist_ok=True)
    _grouped_figure(
        {"Self (with WM)": self_real, "Self (no WM)": self_null},
        "Self-caused prediction error", "D1 control: does the efference copy anticipate?",
        os.path.join(outdir, "fig_d1_control_self_pe.png"))

    print(json.dumps(result, indent=2))
    print("\nInterpretation:")
    print("  median_self WITH efference copy should be LOWER than WITHOUT it,")
    print("  and self_pe_reduced_by_wm_p < 0.05, for the mechanism claim to hold.")
    print("  If they are equal/higher, drop the efference-copy mechanism from 7.6.1.")


def b3(outdir: str):
    """B3 interoceptive-exteroceptive crosstalk: saturation vs food-attention weight."""
    from scipy.stats import pearsonr
    from tests.functional.test_b_suite import (
        _run_episode, _make_cfg, _partial_pearsonr,
        _SAT_CONDITIONS, _N_B3_PER_CONDITION, _STEPS_B3,
    )
    base_ov = {"simulation": {"n_food": 4, "seed": 42},
               "homeostatic": {"saturation_depletion_rate": 0.0}}
    sim_base, brain = _make_cfg(sim_overrides=base_ov, ablation_mode="full")

    sat_flat, w_flat, d_flat = [], [], []
    by_cond = {}
    for sat in _SAT_CONDITIONS:
        cw = []
        for ep in range(_N_B3_PER_CONDITION):
            s = copy.deepcopy(sim_base)
            s["simulation"]["seed"] = 42 + ep
            recs = _run_episode(s, brain, n_steps=_STEPS_B3,
                                starting_saturation=sat, starting_health=0.90,
                                hold_saturation_steps=_STEPS_B3)
            w = float(np.mean([r["food_attention_weight_sat"] for r in recs]))
            d = float(np.mean([r["food_distance"] for r in recs]))
            sat_flat.append(sat); w_flat.append(w); d_flat.append(d); cw.append(w)
        by_cond[f"sat={sat}"] = np.array(cw)

    r, p = pearsonr(sat_flat, w_flat)
    r_partial = _partial_pearsonr(sat_flat, w_flat, d_flat)
    m025 = float(by_cond["sat=0.25"].mean())
    m075 = float(by_cond["sat=0.75"].mean())
    ratio = (m025 + 1e-6) / (m075 + 1e-6)

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "b3_attention_weights.csv"), "w") as fh:
        fh.write("saturation,mean_food_attention_weight_sat,mean_food_distance\n")
        for a, b, c in zip(sat_flat, w_flat, d_flat):
            fh.write(f"{a},{b:.6f},{c:.6f}\n")
    _grouped_figure(by_cond, "Mean food-attention weight (sat_urgency-gated)",
                    f"B3: urgency-gated attention by saturation   (r={r:.2f}, p={p:.1e})",
                    os.path.join(outdir, "fig_7x_b3_crosstalk.png"))
    print(json.dumps({
        "pearson_r_sat_weight": float(r), "pearson_p": float(p),
        "partial_r_controlling_food_dist": float(r_partial),
        "ratio_sat025_over_sat075": float(ratio),
        "mean_weight_by_condition": {k: float(v.mean()) for k, v in by_cond.items()},
    }, indent=2))
    print("\nPaper §7.5 B3: report r (expect < -0.50, p < 0.01), partial r < -0.30, ratio >= 2.")


def c3(outdir: str):
    """C3a goal persistence (full vs no-memory) and C3b satiation suppression."""
    from scipy.stats import mannwhitneyu
    from tests.functional.test_b_suite import _run_episode, _make_cfg
    from tests.functional.test_c_suite import (
        _persistence_steps, _FOOD_AHEAD_FAR, _FOOD_AHEAD_NEAR,
        _FOOD_REMOVAL_STEP_C3A, _SAT_WINDOW,
        _N_C3A, _STEPS_C3A, _N_C3B, _STEPS_C3B,
    )

    # ---- C3a: persistence after food removal, full vs no-memory ----
    sim_a, brain_full = _make_cfg(
        sim_overrides={"simulation": {"n_food": 0},
                       "homeostatic": {"saturation_depletion_rate": 0.001}},
        ablation_mode="full")
    brain_nomem = copy.deepcopy(brain_full)
    brain_nomem.setdefault("policy_generator", {})
    brain_nomem["policy_generator"]["food_memory_decay_steps"] = 1

    def run_c3a(brain, base_seed):
        out = []
        for ep in range(_N_C3A):
            s = copy.deepcopy(sim_a)
            s["simulation"]["seed"] = base_seed + ep
            recs = _run_episode(s, brain, n_steps=_STEPS_C3A,
                                starting_saturation=0.35, starting_health=0.90,
                                food_positions_override=[_FOOD_AHEAD_FAR],
                                food_removal_step=_FOOD_REMOVAL_STEP_C3A)
            out.append(_persistence_steps(recs, removal_step=_FOOD_REMOVAL_STEP_C3A, max_steps=35))
        return np.array(out)

    pers_full = run_c3a(brain_full, 5000)
    pers_nomem = run_c3a(brain_nomem, 5000)
    _, p_c3a = mannwhitneyu(pers_full, pers_nomem, alternative="greater")

    # ---- C3b: satiation suppression rate ----
    sim_b, brain_b = _make_cfg(
        sim_overrides={"simulation": {"n_food": 0},
                       "homeostatic": {"saturation_depletion_rate": 0.0005}},
        ablation_mode="full")
    suppressed = 0
    for ep in range(_N_C3B):
        s = copy.deepcopy(sim_b)
        s["simulation"]["seed"] = 6000 + ep
        recs = _run_episode(s, brain_b, n_steps=_STEPS_C3B,
                            starting_saturation=1.0, starting_health=0.90,
                            food_positions_override=[_FOOD_AHEAD_NEAR])
        approached = any(r["policy_id"] == "move_forward" and r["food_in_ray"]
                         for r in recs if r["step"] < _SAT_WINDOW)
        if not approached:
            suppressed += 1
    supp_rate = suppressed / _N_C3B

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "c3a_persistence.csv"), "w") as fh:
        fh.write("condition,episode,persistence_steps\n")
        for i, v in enumerate(pers_full):
            fh.write(f"full,{i},{int(v)}\n")
        for i, v in enumerate(pers_nomem):
            fh.write(f"no_memory,{i},{int(v)}\n")
    _grouped_figure({"Full": pers_full, "No-memory": pers_nomem},
                    "Ghost-directed persistence (steps after food removal)",
                    f"C3a: goal persistence   (MWU p = {p_c3a:.2e})",
                    os.path.join(outdir, "fig_7x_c3a_persistence.png"),
                    hline=15, hline_label="15-step target")
    print(json.dumps({
        "c3a_mean_full": float(pers_full.mean()), "c3a_mean_no_memory": float(pers_nomem.mean()),
        "c3a_pct_full_gt10": float(np.mean(pers_full > 10)),
        "c3a_pct_nomem_gt10": float(np.mean(pers_nomem > 10)),
        "c3a_mwu_p": float(p_c3a),
        "c3b_suppression_rate": float(supp_rate),
        "c3b_suppressed_of_n": f"{suppressed}/{_N_C3B}",
    }, indent=2))
    print("\nPaper: C3a targets full mean > 15 & >55% trials >10 steps (this is the architectural FAIL — "
          "report observed mean). C3b PASS if suppression_rate > 0.50.")


def c3a_probe(outdir: str):
    """
    5-episode diagnostic: WHY is full C3a identical to no-memory (p=1.0)?
    Wraps the real FreeEnergyMinimizer.score to record, per step, whether the
    food-memory bonus (a) is gated open, (b) is populated/aligned, (c) is large
    enough to flip the argmax to move_forward. Runs the REAL _run_episode (full
    condition) so nothing is reimplemented.
    """
    from tests.functional.test_b_suite import _run_episode, _make_cfg
    from tests.functional.test_c_suite import _FOOD_AHEAD_FAR, _FOOD_REMOVAL_STEP_C3A, _STEPS_C3A
    from core.layers.action_selection.FreeEnergyMinimizer import FreeEnergyMinimizer

    N_EPISODES = 5
    WINDOW = 35  # persistence window used by the test: [removal, removal+35)

    log = []          # list of per-step dicts for the current episode
    step_counter = {"i": 0}
    orig_score = FreeEnergyMinimizer.score

    def wrapped(self, policies, drive_batch, pe_batch, area_familiarity=0.5, context=None, **kw):
        scores = orig_score(self, policies, drive_batch, pe_batch,
                            area_familiarity=area_familiarity, context=context, **kw)
        step = step_counter["i"]
        step_counter["i"] += 1

        rays = (context or {}).get("raycast_hits") or []
        food_visible = any(r.get("hit_tag") in ("GoodGoal", "GoodGoalMulti") for r in rays)
        sat_urg = 0.0
        if drive_batch:
            for sig in drive_batch.signals:
                if sig.channel_id == "saturation":
                    sat_urg = float(sig.urgency)
        lfa, lfd = self._last_food_angle, self._last_food_dist
        gate = (not food_visible) and sat_urg >= self._food_memory_urgency_threshold
        mf_bonus = 0.0
        if gate and lfa is not None and abs(lfa) < 20:
            mf_bonus = (1.0 - lfd) * self._food_proximity_bonus * 0.4 * sat_urg

        winner = max(scores, key=scores.get)
        mf = float(scores.get("move_forward", float("nan")))
        best_other = max((v for k, v in scores.items() if k != "move_forward"), default=float("nan"))
        # would move_forward still win without the memory bonus?
        decisive = (winner == "move_forward") and (mf - mf_bonus) < best_other

        log.append({"step": step, "food_visible": food_visible, "sat_urg": round(sat_urg, 3),
                    "gate": gate, "lfa": None if lfa is None else round(lfa, 1),
                    "lfd": round(lfd, 3), "mf_bonus": round(mf_bonus, 4),
                    "winner": winner, "mf": round(mf, 4), "best_other": round(best_other, 4),
                    "decisive": decisive})
        return scores

    sim, brain = _make_cfg(
        sim_overrides={"simulation": {"n_food": 0},
                       "homeostatic": {"saturation_depletion_rate": 0.001}},
        ablation_mode="full")

    gate_open = pop = decisive = 0
    total = 0
    FreeEnergyMinimizer.score = wrapped
    try:
        for ep in range(N_EPISODES):
            log.clear()
            step_counter["i"] = 0
            s = copy.deepcopy(sim)
            s["simulation"]["seed"] = 5000 + ep
            _run_episode(s, brain, n_steps=_STEPS_C3A,
                         starting_saturation=0.35, starting_health=0.90,
                         food_positions_override=[_FOOD_AHEAD_FAR],
                         food_removal_step=_FOOD_REMOVAL_STEP_C3A)
            win = [r for r in log if _FOOD_REMOVAL_STEP_C3A <= r["step"] < _FOOD_REMOVAL_STEP_C3A + WINDOW]
            print(f"\n=== episode seed {5000 + ep} (post-removal window) ===")
            print(f"  {'step':>4} {'food_vis':>8} {'sat_urg':>7} {'gate':>5} {'lfa':>6} "
                  f"{'mf_bonus':>8} {'winner':>13} {'mf':>7} {'best_oth':>8} {'decisive':>8}")
            for r in win[:20]:
                print(f"  {r['step']:>4} {str(r['food_visible']):>8} {r['sat_urg']:>7} "
                      f"{str(r['gate']):>5} {str(r['lfa']):>6} {r['mf_bonus']:>8} "
                      f"{r['winner']:>13} {r['mf']:>7} {r['best_other']:>8} {str(r['decisive']):>8}")
            for r in win:
                total += 1
                gate_open += int(r["gate"])
                pop += int(r["mf_bonus"] > 0)
                decisive += int(r["decisive"])
    finally:
        FreeEnergyMinimizer.score = orig_score

    print("\n=== summary across 5 episodes (post-removal window) ===")
    print(json.dumps({"window_steps": total, "gate_open_steps": gate_open,
                      "bonus_populated_steps": pop, "bonus_decisive_steps": decisive}, indent=2))
    if gate_open == 0:
        verdict = ("GATE NEVER OPENS: sat_urgency stayed below the "
                   "food_memory_urgency_threshold, so the buffer is never consulted. "
                   "C3a fails at the drive gate, not the bonus scaling.")
    elif pop == 0:
        verdict = ("BUFFER NEVER POPULATED/ALIGNED: gate opened but _last_food_angle "
                   "was None or misaligned (>20 deg), so no bonus was applied.")
    elif decisive == 0:
        verdict = ("BONUS FIRES BUT IS NEVER DECISIVE: move_forward received the memory "
                   "bonus but it never changed the winning action. This is 'insufficient "
                   "scaling' — the bonus is real but too small to overcome the epistemic baseline.")
    else:
        verdict = (f"BONUS DECISIVE on {decisive} steps: the memory mechanism IS active. "
                   "full == no_memory would be unexpected; re-check the comparison.")
    print("\nVERDICT:", verdict)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("figure", choices=["d1", "b1", "coverage", "d1_control", "b3", "c3", "c3a_probe"],
                    help="which figure/analysis to generate")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent),
                    help="repo root (defaults to parent of this script's dir)")
    ap.add_argument("--outdir", default="results/figures")
    args = ap.parse_args()
    _setup_paths(args.root)
    {"d1": d1, "b1": b1, "coverage": coverage, "d1_control": d1_control,
     "b3": b3, "c3": c3, "c3a_probe": c3a_probe}[args.figure](args.outdir)


if __name__ == "__main__":
    main()
