"""Structured exception hooks."""
from __future__ import annotations

import logging
import sys
import traceback
from typing import Callable, Optional

log = logging.getLogger(__name__)


def install_exception_hooks(on_exception: Optional[Callable] = None) -> None:
    original = sys.excepthook

    def hook(exc_type, exc_value, exc_tb):
        tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        log.critical("Unhandled exception:\n%s", tb_str)
        if on_exception:
            try:
                on_exception(exc_type, exc_value, tb_str)
            except Exception:
                pass
        original(exc_type, exc_value, exc_tb)

    sys.excepthook = hook
