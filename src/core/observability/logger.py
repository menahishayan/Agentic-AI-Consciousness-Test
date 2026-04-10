"""
RunLogger — structured JSONL telemetry per run.

Writes to src/logs/runs/<timestamp>_<run_id>/:
  run.json       — run metadata (config, start time, Python version)
  events.jsonl   — per-step events
  state.jsonl    — AgentState snapshots (gated by LOG_STATE)
  llm.jsonl      — LLM calls (gated by LOG_PROMPTS)
  memory.jsonl   — FAISS operations (gated by LOG_MEMORY)
  metrics.jsonl  — numeric per-step metrics (always on, fixed schema)
  tracebacks.jsonl — structured exceptions
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

from core.observability.serializer import SafeEncoder, safe_dumps

log = logging.getLogger(__name__)


class RunLogger:
    """
    Structured telemetry for a single agent run.
    All writes are append-only JSONL for easy streaming analysis.
    """

    def __init__(
        self,
        run_dir: Path,
        config: Dict[str, Any],
        log_state: bool = True,
        log_prompts: bool = False,
        log_memory: bool = True,
    ) -> None:
        self._dir = run_dir
        self._log_state = log_state
        self._log_prompts = log_prompts
        self._log_memory = log_memory
        self._start_time = time.time()

        self._events_f = open(run_dir / "events.jsonl", "a")
        self._metrics_f = open(run_dir / "metrics.jsonl", "a")
        self._tracebacks_f = open(run_dir / "tracebacks.jsonl", "a")
        self._state_f = open(run_dir / "state.jsonl", "a") if log_state else None
        self._llm_f = open(run_dir / "llm.jsonl", "a") if log_prompts else None
        self._memory_f = open(run_dir / "memory.jsonl", "a") if log_memory else None

        self._write_run_metadata(config)

    @property
    def run_dir(self) -> Path:
        return self._dir

    # ------------------------------------------------------------------
    # Public logging methods
    # ------------------------------------------------------------------

    def event(self, name: str, payload: Any, step: int = 0) -> None:
        self._append(self._events_f, {
            "t": time.time(),
            "step": step,
            "event": name,
            "payload": payload,
        })

    def metrics(
        self,
        step: int,
        health: float,
        saturation: float,
        arousal: float,
        valence: float,
        pe_mean: float,
        policy_id: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        row: Dict[str, Any] = {
            "step": step,
            "health": round(health, 4),
            "saturation": round(saturation, 4),
            "arousal": round(arousal, 4),
            "valence": round(valence, 4),
            "pe_mean": round(pe_mean, 4),
            "policy_id": policy_id,
        }
        if extra:
            row.update(extra)
        self._append(self._metrics_f, row)

    def state(self, state_obj: Any, step: int = 0) -> None:
        if self._state_f:
            self._append(self._state_f, {"step": step, "state": state_obj})

    def llm(
        self,
        prompt: str,
        response: str,
        model: str,
        latency_ms: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        trigger_reason: str = "",
        selected: Optional[str] = None,
        reason: Optional[str] = None,
        step: int = 0,
        efe_scores: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append one LLM call record to llm.jsonl.

        trigger_reason is the canonical field for inferring inference depth in
        analysis code — do NOT add a separate "depth" field. The mapping is:
          "normal"        → fast prompt
          "drive_conflict"
          "pe_streak"
          "skill_gap"     → full CoT prompt
          "high_arousal"  → EFE argmax (LLM bypassed; this entry is not written)
          "no_llm"        → EFE argmax (LLM absent; this entry is not written)
          "<reason>:efe_argmax"  → fast/full fell back to EFE argmax on timeout
        A "depth" field would duplicate this information and silently go stale
        whenever the trigger logic changes.
        """
        if self._llm_f:
            entry: Dict[str, Any] = {
                "t": time.time(),
                "step": step,
                "model": model,
                "trigger_reason": trigger_reason,
                "selected": selected,
                "latency_ms": round(latency_ms, 1),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "prompt": prompt,
                "response": response,
            }
            if reason is not None:
                entry["reason"] = reason
            if efe_scores:
                entry["efe_scores"] = efe_scores
            self._append(self._llm_f, entry)

    def memory_op(self, op: str, details: Any, step: int = 0) -> None:
        if self._memory_f:
            self._append(self._memory_f, {
                "t": time.time(),
                "step": step,
                "op": op,
                "details": details,
            })

    def traceback(self, exc: BaseException, context: str = "", step: int = 0) -> None:
        tb = traceback.format_exc()
        self._append(self._tracebacks_f, {
            "t": time.time(),
            "step": step,
            "context": context,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": tb,
        })
        log.error("[step %d] %s: %s", step, context, exc)

    def close(self) -> None:
        for f in [
            self._events_f,
            self._metrics_f,
            self._tracebacks_f,
            self._state_f,
            self._llm_f,
            self._memory_f,
        ]:
            if f:
                try:
                    f.close()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write_run_metadata(self, config: Dict[str, Any]) -> None:
        meta = {
            "start_time": self._start_time,
            "start_time_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self._start_time)),
            "python_version": sys.version,
            "platform": platform.platform(),
            "log_dir": str(self._dir),
            "adapter": config.get("adapter_folder", "unknown"),
            "llm_provider": config.get("llm", {}).get("provider", "unknown"),
            "llm_model": config.get("llm", {}).get("model", "unknown"),
            "ablation_mode": config.get("ablation", {}).get("mode", "full"),
            "config": config,
        }
        (self._dir / "run.json").write_text(safe_dumps(meta))

    def _append(self, f: Any, obj: Any) -> None:
        if f is None:
            return
        try:
            f.write(safe_dumps(obj) + "\n")
            f.flush()
        except Exception as exc:
            log.warning("RunLogger write error: %s", exc)
