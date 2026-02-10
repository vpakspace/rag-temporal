"""Async helpers for running coroutines from sync context (e.g. Streamlit)."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from typing import Any, Coroutine

logger = logging.getLogger(__name__)


def run_async(coro: Coroutine) -> Any:
    """Run an async coroutine from synchronous code.

    Handles the case where an event loop is already running
    (e.g. inside Streamlit) by spawning a separate thread.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        else:
            return asyncio.run(coro)
    except RuntimeError:
        return asyncio.run(coro)
