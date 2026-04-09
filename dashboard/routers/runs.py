"""
REST endpoints for run listing, metadata, and bulk data retrieval.
"""

from fastapi import APIRouter, HTTPException, Request

from dashboard.services.log_tailer import JsonlTailer

router = APIRouter(tags=["runs"])


def _registry(request: Request):
    return request.app.state.registry


@router.get("/runs")
async def list_runs(request: Request):
    registry = _registry(request)
    runs = registry.all_runs()
    return [
        {
            "run_id": r.run_id,
            "adapter": r.metadata.get("adapter", "unknown"),
            "llm_provider": r.metadata.get("llm_provider", "unknown"),
            "llm_model": r.metadata.get("llm_model", "unknown"),
            "start_time": r.metadata.get("start_time"),
            "start_time_str": r.metadata.get("start_time_str", r.run_id[:15]),
            "is_complete": r.is_complete,
            "step_count": r.step_count,
        }
        for r in runs
    ]


@router.get("/runs/{run_id}")
async def get_run(run_id: str, request: Request):
    registry = _registry(request)
    info = registry.get(run_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return {
        **info.metadata,
        "run_id": run_id,
        "is_complete": info.is_complete,
        "step_count": info.step_count,
    }


@router.get("/runs/{run_id}/bulk")
async def get_bulk(run_id: str, request: Request):
    """
    Return all historical data for a run.
    Used on initial page load before SSE stream opens.
    Position data is extracted from state.jsonl (only step, x, z, heading).
    """
    registry = _registry(request)
    info = registry.get(run_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Run not found")

    metrics = JsonlTailer(info.run_dir / "metrics.jsonl").read_all()
    events = JsonlTailer(info.run_dir / "events.jsonl").read_all()
    raw_states = JsonlTailer(info.run_dir / "state.jsonl").read_all()
    raw_llm = JsonlTailer(info.run_dir / "llm.jsonl").read_all()

    positions = []
    for row in raw_states:
        step = row.get("step")
        state = row.get("state", {})
        pos = state.get("position", {})
        if pos:
            positions.append({
                "step": step,
                "x": pos.get("x"),
                "z": pos.get("z"),
                "heading": pos.get("heading"),
            })

    llm_calls = [
        {
            "step": row.get("step"),
            "t": row.get("t"),
            "model": row.get("model"),
            "trigger_reason": row.get("trigger_reason", ""),
            "latency_ms": row.get("latency_ms"),
            "input_tokens": row.get("input_tokens"),
            "output_tokens": row.get("output_tokens"),
            "response": row.get("response", ""),
            "prompt": row.get("prompt", ""),
        }
        for row in raw_llm
    ]

    # Extract run_complete payload from the episode_complete event if present
    run_complete = None
    for row in events:
        if row.get("event") == "episode_complete":
            run_complete = row.get("payload", {})

    return {
        "metrics": metrics,
        "events": events,
        "positions": positions,
        "llm_calls": llm_calls,
        "run_complete": run_complete,
    }
