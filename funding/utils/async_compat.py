"""
Async compatibility helpers for Python 3.9+ runtimes.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager


@asynccontextmanager
async def async_timeout(seconds: float):
    """
    Async timeout context manager compatible with Python 3.9+.

    Uses asyncio.timeout when available (3.11+), falls back to task cancellation
    with TimeoutError translation on older versions.
    """
    if hasattr(asyncio, "timeout"):
        async with asyncio.timeout(seconds):
            yield
        return

    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    if task is None:
        yield
        return

    timed_out = {"flag": False}

    def _cancel() -> None:
        timed_out["flag"] = True
        task.cancel()

    handle = loop.call_later(float(seconds), _cancel)
    try:
        yield
    except asyncio.CancelledError as exc:
        if timed_out["flag"]:
            raise asyncio.TimeoutError() from exc
        raise
    finally:
        handle.cancel()
