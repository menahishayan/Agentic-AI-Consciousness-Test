"""
FastAPI application factory for the agent run dashboard.
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from dashboard.config import LOGS_ROOT, POLL_INTERVAL_SECONDS, STATIC_DIR
from dashboard.routers import runs, stream
from dashboard.services.run_registry import RunRegistry


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry = RunRegistry(LOGS_ROOT)
    registry.scan()
    app.state.registry = registry
    task = asyncio.create_task(_discovery_loop(registry))
    yield
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def _discovery_loop(registry: RunRegistry) -> None:
    while True:
        await asyncio.sleep(POLL_INTERVAL_SECONDS)
        registry.scan()


def create_app() -> FastAPI:
    app = FastAPI(title="Agent Run Dashboard", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(runs.router, prefix="/api/v1")
    app.include_router(stream.router, prefix="/api/v1")
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    async def index():
        return FileResponse(str(STATIC_DIR / "index.html"))

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "dashboard.server:create_app",
        factory=True,
        host="0.0.0.0",
        port=8765,
        reload=False,
    )
