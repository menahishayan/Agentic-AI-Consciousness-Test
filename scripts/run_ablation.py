#!/usr/bin/env python3
"""
Ablation study runner — six conditions × N episodes.

Kills the null hypothesis:
  "A sufficiently prompted LLM with raw observations achieves the same
   behavioral profile, making the interoceptive/predictive layers unnecessary."

Each condition isolates one architectural contribution by removing or zeroing
it, then running the same episode protocol and computing four behavioral probes
that target the system's theoretical claims rather than raw survival score.

Usage:
    cd /path/to/Capstone
    source venv/bin/activate
    python scripts/run_ablation.py [options]

    # Quick validation (2 episodes, 100 steps):
    python scripts/run_ablation.py --episodes 2 --max-steps 100

    # Full experiment (10 episodes, 500 steps per condition):
    python scripts/run_ablation.py --episodes 10 --max-steps 500

    # Single condition for debugging:
    python scripts/run_ablation.py --conditions full llm_only --episodes 3

MLflow logging is optional — results are always saved to JSON.
Install mlflow with: pip install mlflow
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure src/ is on the Python path
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
for _p in [str(_SRC), str(_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

CONDITIONS = [
    "full",
    "llm_only",
    "no_efe",
    "no_interoceptive",
    "no_arousal",
    "efe_only",
]

log = logging.getLogger("run_ablation")


def _load_config(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _patch_config(
    config: Dict[str, Any],
    condition: str,
    injection_step: Optional[int],
    max_steps: int,
) -> Dict[str, Any]:
    """Deep-copy config and apply per-condition overrides."""
    patched = copy.deepcopy(config)

    # Ablation mode
    patched.setdefault("ablation", {})
    patched["ablation"]["mode"] = condition
    if injection_step is not None:
        patched["ablation"]["drive_injection_step"] = injection_step

    # Always use headless adapter for reproducibility — no Unity required
    patched["adapter_folder"] = "headless"

    # Suppress verbose logging during batch runs
    # patched.setdefault("observability", {})
    # patched["observability"]["log_state"] = False
    # patched["observability"]["log_prompts"] = False
    # patched["observability"]["log_memory"] = False

    return patched


def _run_episode(config: Dict[str, Any], max_steps: int) -> Dict[str, Any]:
    """
    Bootstrap and run a single episode.

    Returns episode stats dict including 'run_dir' (Path as str) for
    subsequent BehavioralMetrics analysis.
    """
    from core.adapters.loader import build_adapter
    from core.llm.factory import build_llm_client
    from core.memory.manager import MemoryManager
    from core.observability.logger import RunLogger
    from core.observability.paths import make_run_dir
    from core.runtime.loop import AgentLoop

    obs_cfg = config.get("observability", {})
    log_root = obs_cfg.get("log_root", "src/logs/runs")

    run_dir = make_run_dir(log_root)
    run_logger = RunLogger(
        run_dir=run_dir,
        config=config,
        log_state=obs_cfg.get("log_state", False),
        log_prompts=obs_cfg.get("log_prompts", False),
        log_memory=obs_cfg.get("log_memory", False),
    )

    adapter = build_adapter(config["adapter_folder"], config.get("adapter_config", {}))
    llm_client = build_llm_client(config.get("llm"), logger=run_logger)
    memory = MemoryManager(config)

    loop = AgentLoop(
        adapter=adapter,
        memory=memory,
        llm_client=llm_client,
        logger=run_logger,
        config=config,
    )

    try:
        stats = loop.run(max_steps=max_steps)
    finally:
        adapter.close()
        run_logger.close()

    stats["run_dir"] = str(run_dir)
    return stats


def _compute_probes(run_dir: str, injection_step: Optional[int]) -> Dict[str, Any]:
    from core.observability.behavioral_metrics import BehavioralMetrics
    bm = BehavioralMetrics(Path(run_dir), injection_step=injection_step)
    return bm.summary()


def _safe_mean(vals: List[float]) -> float:
    clean = [v for v in vals if not math.isnan(v) and not math.isinf(v)]
    return sum(clean) / len(clean) if clean else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ablation study across 6 conditions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--episodes", type=int, default=10,
                        help="Episodes per condition")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Max steps per episode")
    parser.add_argument("--out", type=str, default="results/ablation",
                        help="Output directory for results JSON")
    parser.add_argument("--injection-step", type=int, default=200,
                        help="Episode step at which saturation is injected (Probe 3 SRL)")
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS,
                        choices=CONDITIONS,
                        help="Which conditions to run (default: all 6)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Try to import mlflow (optional dependency)
    try:
        import mlflow
        mlflow.set_experiment("ablation_study")
        use_mlflow = True
        log.info("MLflow enabled — logging to 'ablation_study' experiment.")
    except ImportError:
        use_mlflow = False
        log.info("MLflow not installed — results saved to JSON only.")

    config_path = _ROOT / "config.json"
    if not config_path.exists():
        log.error("config.json not found at %s", config_path)
        sys.exit(1)
    base_config = _load_config(config_path)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict[str, Any]] = []

    for condition in args.conditions:
        log.info("=== Condition: %s (%d episodes × %d steps) ===",
                 condition, args.episodes, args.max_steps)
        config = _patch_config(base_config, condition, args.injection_step, args.max_steps)

        for ep in range(args.episodes):
            log.info("  Episode %d/%d ...", ep + 1, args.episodes)

            if use_mlflow:
                with mlflow.start_run(run_name=f"{condition}_ep{ep:02d}"):
                    mlflow.set_tags({"condition": condition, "episode": ep})
                    stats = _run_episode(config, args.max_steps)
                    probes = _compute_probes(stats["run_dir"], args.injection_step)
                    mlflow.log_metrics({
                        k: v for k, v in {**probes,
                            "steps_survived": stats.get("steps_run", 0),
                            "final_health": stats.get("final_health", 0.0),
                            "final_saturation": stats.get("final_saturation", 0.0),
                        }.items()
                        if not math.isnan(float(v)) and not math.isinf(float(v))
                    }, step=ep)
            else:
                stats = _run_episode(config, args.max_steps)
                probes = _compute_probes(stats["run_dir"], args.injection_step)

            row = {
                "condition": condition,
                "episode": ep,
                **stats,
                **probes,
            }
            all_results.append(row)
            log.info("    steps=%d  dbc=%.3f  adi=%.3f  srl=%d  q4/q1=%.2f",
                     probes.get("steps_survived", 0),
                     probes.get("dbc", float("nan")),
                     probes.get("adi", float("nan")),
                     probes.get("srl", -1),
                     probes.get("uqcr_q4_q1", float("nan")),
                     )

    # Save full results
    results_path = out_dir / "ablation_results.json"
    results_path.write_text(json.dumps(all_results, indent=2, default=str))
    log.info("Full results saved to %s", results_path)

    # Print summary table
    _print_summary(all_results, args.conditions)


def _print_summary(results: List[Dict[str, Any]], conditions: List[str]) -> None:
    print("\n" + "=" * 78)
    print("ABLATION STUDY SUMMARY")
    print("=" * 78)
    header = f"{'Condition':<20} {'DBC':>8} {'ADI':>8} {'SRL':>8} {'Q4/Q1':>8} {'Steps':>8} {'N':>4}"
    print(header)
    print("-" * 78)

    for cond in conditions:
        rows = [r for r in results if r["condition"] == cond]
        if not rows:
            continue

        dbc = _safe_mean([r.get("dbc", float("nan")) for r in rows])
        adi = _safe_mean([r.get("adi", float("nan")) for r in rows])
        srl_vals = [r.get("srl", -1) for r in rows if r.get("srl", -1) >= 0]
        srl = _safe_mean([float(v) for v in srl_vals]) if srl_vals else float("nan")
        q4q1 = _safe_mean([r.get("uqcr_q4_q1", float("nan")) for r in rows])
        steps = _safe_mean([float(r.get("steps_survived", 0)) for r in rows])
        n = len(rows)

        def _fmt(v: float, width: int = 8, decimals: int = 3) -> str:
            if math.isnan(v):
                return f"{'nan':>{width}}"
            return f"{v:>{width}.{decimals}f}"

        print(
            f"{cond:<20}"
            f"{_fmt(dbc)}"
            f"{_fmt(adi)}"
            f"{_fmt(srl, decimals=1)}"
            f"{_fmt(q4q1, decimals=2)}"
            f"{_fmt(steps, decimals=1)}"
            f"{n:>4}"
        )

    print("=" * 78)
    print("\nInterpretation guide:")
    print("  DBC  — Drive-Urgency Behavioral Coupling (Pearson r). Full > 0.4 expected.")
    print("  ADI  — Arousal-Diversity Index (Pearson r). Full positive; no_arousal ≈ 0.")
    print("  SRL  — Satiation Reorientation Latency (steps). Full < llm_only expected.")
    print("  Q4/Q1 — Urgency quartile food rate ratio. Full >> 1 expected.")


def _safe_mean(vals: List[float]) -> float:
    clean = [v for v in vals if not math.isnan(v) and not math.isinf(v)]
    return sum(clean) / len(clean) if clean else float("nan")


if __name__ == "__main__":
    main()
