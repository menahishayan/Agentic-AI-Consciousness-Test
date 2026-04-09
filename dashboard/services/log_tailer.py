"""
Byte-offset based JSONL file tailer.
Holds an open file handle and advances a byte offset on each poll() call.
Safe against partial writes: only emits lines ending in '\\n'.
"""

import json
from pathlib import Path


class JsonlTailer:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._offset = 0
        self._fh = None
        self._buffer = b""

    def _open(self) -> bool:
        if self._fh is not None:
            return True
        if not self._path.exists():
            return False
        self._fh = open(self._path, "rb")
        return True

    def read_all(self) -> list[dict]:
        """Read all lines from byte 0. Used for initial bulk load."""
        if not self._path.exists():
            return []
        try:
            with open(self._path, "rb") as f:
                raw = f.read()
        except OSError:
            return []
        rows = []
        for line in raw.split(b"\n"):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return rows

    def poll(self) -> list[dict]:
        """Read any new complete lines since last call. Returns [] if no new data."""
        if not self._open():
            return []

        # Detect file truncation (e.g. log rotation)
        try:
            file_size = self._path.stat().st_size
        except OSError:
            return []

        if file_size < self._offset:
            self._offset = 0
            self._buffer = b""
            self._fh.seek(0)

        self._fh.seek(self._offset)
        chunk = self._fh.read(65536)
        if not chunk:
            return []

        self._offset += len(chunk)
        self._buffer += chunk

        rows = []
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return rows

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None
