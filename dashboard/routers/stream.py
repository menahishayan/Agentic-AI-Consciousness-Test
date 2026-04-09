"""
SSE (Server-Sent Events) streaming endpoints.

/runs/{run_id}/stream  — streams new metrics, events, and positions for a run
/runs/stream/new       — notifies when new run directories are discovered
"""

import asyncio
import json
import time
from typing import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from dashboard.config import POLL_INTERVAL_SECONDS
from dashboard.services.log_tailer import JsonlTailer

router = APIRouter(tags=["stream"])

HEARTBEAT_INTERVAL = 15  # seconds between heartbeat events when idle


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _run_stream(
    run_id: str,
    registry,
    after_step: int,
) -> AsyncIterator[str]:
    info = registry.get(run_id)
    if info is None:
        yield _sse("error", {"message": f"Run {run_id} not found"})
        return

    metrics_tailer = JsonlTailer(info.run_dir / "metrics.jsonl")
    events_tailer = JsonlTailer(info.run_dir / "events.jsonl")
    state_tailer = JsonlTailer(info.run_dir / "state.jsonl")
    llm_tailer = JsonlTailer(info.run_dir / "llm.jsonl")

    # Fast-forward tailers past already-seen steps by advancing offsets
    # via read_all (we skip emission since bulk endpoint covered history)
    if after_step > 0:
        for row in metrics_tailer.read_all():
            pass  # offsets advanced by read_all
        for row in events_tailer.read_all():
            pass
        for row in state_tailer.read_all():
            pass
        for row in llm_tailer.read_all():
            pass
        # Now reset to tail from current position
        metrics_tailer._offset = 0
        events_tailer._offset = 0
        state_tailer._offset = 0
        llm_tailer._offset = 0
        # Re-advance by seeking to end
        for tailer in (metrics_tailer, events_tailer, state_tailer, llm_tailer):
            if tailer._path.exists():
                size = tailer._path.stat().st_size
                tailer._offset = size

    last_activity = time.monotonic()
    empty_cycles = 0
    complete_seen = info.is_complete

    try:
        while True:
            await asyncio.sleep(POLL_INTERVAL_SECONDS)

            # Refresh run state
            registry.scan()
            info = registry.get(run_id)

            had_data = False

            # --- metrics ---
            for row in metrics_tailer.poll():
                if row.get("step", -1) <= after_step:
                    continue
                yield _sse("metric", row)
                had_data = True

            # --- events ---
            for row in events_tailer.poll():
                step = row.get("step", -1)
                if step <= after_step:
                    continue
                event_type = row.get("event", "")
                if event_type == "episode_complete":
                    yield _sse("run_complete", row.get("payload", {}))
                    complete_seen = True
                elif event_type == "episode_cancelled":
                    yield _sse("run_cancelled", row.get("payload", {}))
                    complete_seen = True
                else:
                    yield _sse("agent_event", row)
                had_data = True

            # --- state (positions only) ---
            for row in state_tailer.poll():
                step = row.get("step", -1)
                if step <= after_step:
                    continue
                state = row.get("state", {})
                pos = state.get("position", {})
                if pos:
                    yield _sse("position", {
                        "step": step,
                        "x": pos.get("x"),
                        "z": pos.get("z"),
                        "heading": pos.get("heading"),
                    })
                    had_data = True

            # --- llm calls ---
            for row in llm_tailer.poll():
                step = row.get("step", -1)
                if step <= after_step:
                    continue
                yield _sse("llm_call", {
                    "step": step,
                    "t": row.get("t"),
                    "model": row.get("model"),
                    "trigger_reason": row.get("trigger_reason", ""),
                    "selected": row.get("selected"),
                    "reason": row.get("reason"),
                    "latency_ms": row.get("latency_ms"),
                    "input_tokens": row.get("input_tokens"),
                    "output_tokens": row.get("output_tokens"),
                    "response": row.get("response", ""),
                    "prompt": row.get("prompt", ""),
                })
                had_data = True

            if had_data:
                last_activity = time.monotonic()
                empty_cycles = 0
            else:
                empty_cycles += 1

            # Heartbeat to keep connection alive
            if time.monotonic() - last_activity >= HEARTBEAT_INTERVAL:
                yield _sse("heartbeat", {})
                last_activity = time.monotonic()

            # Close stream after run is complete and no new data for 2 cycles
            if (info and info.is_complete or complete_seen) and empty_cycles >= 2:
                break

    except asyncio.CancelledError:
        pass
    finally:
        metrics_tailer.close()
        events_tailer.close()
        state_tailer.close()
        llm_tailer.close()


async def _new_runs_stream(registry) -> AsyncIterator[str]:
    """Emits 'new_run' SSE events when new run directories are discovered."""
    # Track which runs we've already announced
    announced = {r.run_id for r in registry.all_runs()}
    # Announce all existing runs immediately on connect
    for info in registry.all_runs():
        yield _sse("existing_run", {
            "run_id": info.run_id,
            "adapter": info.metadata.get("adapter", "unknown"),
            "llm_provider": info.metadata.get("llm_provider", "unknown"),
            "llm_model": info.metadata.get("llm_model", "unknown"),
            "ablation_mode": info.metadata.get("ablation_mode", "full"),
            "start_time": info.metadata.get("start_time"),
            "start_time_str": info.metadata.get("start_time_str", info.run_id[:15]),
            "is_complete": info.is_complete,
            "is_cancelled": info.is_cancelled,
            "step_count": info.step_count,
        })

    try:
        while True:
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
            new_ids = registry.scan()
            for run_id in new_ids:
                if run_id in announced:
                    continue
                announced.add(run_id)
                info = registry.get(run_id)
                if info:
                    yield _sse("new_run", {
                        "run_id": run_id,
                        "adapter": info.metadata.get("adapter", "unknown"),
                        "llm_provider": info.metadata.get("llm_provider", "unknown"),
                        "llm_model": info.metadata.get("llm_model", "unknown"),
                        "ablation_mode": info.metadata.get("ablation_mode", "full"),
                        "start_time": info.metadata.get("start_time"),
                        "start_time_str": info.metadata.get("start_time_str", run_id[:15]),
                        "is_complete": info.is_complete,
                        "is_cancelled": info.is_cancelled,
                        "step_count": info.step_count,
                    })
    except asyncio.CancelledError:
        pass


@router.get("/runs/{run_id}/stream")
async def stream_run(run_id: str, request: Request, after_step: int = 0):
    registry = request.app.state.registry

    async def generator():
        async for chunk in _run_stream(run_id, registry, after_step):
            if await request.is_disconnected():
                break
            yield chunk

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/runs/stream/new")
async def stream_new_runs(request: Request):
    registry = request.app.state.registry

    async def generator():
        async for chunk in _new_runs_stream(registry):
            if await request.is_disconnected():
                break
            yield chunk

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
