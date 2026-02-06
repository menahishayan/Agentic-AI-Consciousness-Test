from __future__ import annotations

import sys
import threading
from typing import Optional

import faulthandler

from core.observability.logger import RunLogger


def install_exception_hooks(logger: Optional[RunLogger]) -> None:
    if logger is None:
        return

    try:
        fh = logger.paths.tracebacks_log.open("a", encoding="utf-8")
        faulthandler.enable(file=fh)
        logger._faulthandler_file = fh
    except Exception:
        pass

    original_sys_hook = sys.excepthook

    def _sys_hook(exc_type, exc, tb):
        logger.exception(exc, context={"uncaught": True})
        original_sys_hook(exc_type, exc, tb)

    sys.excepthook = _sys_hook

    if hasattr(threading, "excepthook"):
        original_thread_hook = threading.excepthook

        def _thread_hook(args):
            thread_name = args.thread.name if args.thread else None
            logger.exception(
                args.exc_value,
                context={"thread": thread_name, "uncaught": True},
            )
            if original_thread_hook:
                original_thread_hook(args)

        threading.excepthook = _thread_hook
