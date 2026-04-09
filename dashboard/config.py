from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_ROOT = PROJECT_ROOT / "src" / "logs" / "runs"
STATIC_DIR = Path(__file__).resolve().parent / "static"

POLL_INTERVAL_SECONDS = 1.0
TAIL_CHUNK_BYTES = 65536  # 64 KB per poll cycle
RUN_DIR_PATTERN = r"^\d{8}_\d{6}_[0-9a-f]{8}$"
