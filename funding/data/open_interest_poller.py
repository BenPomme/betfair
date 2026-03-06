from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Callable, Deque, Dict, Optional, Set

import config

logger = logging.getLogger(__name__)


class OpenInterestPoller:
    def __init__(self, futures_client, symbols_fn: Callable[[], Set[str]], interval_seconds: Optional[int] = None) -> None:
        self._client = futures_client
        self._symbols_fn = symbols_fn
        self._interval_seconds = max(10, int(interval_seconds or config.FEAR_GREED_POLL_SECONDS))
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._latest: Dict[str, dict] = {}
        self._history: Dict[str, Deque[dict]] = defaultdict(lambda: deque(maxlen=256))

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run(), name="open_interest_poller")

    async def stop(self) -> None:
        self._running = False
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    async def _run(self) -> None:
        while self._running:
            symbols = sorted(self._symbols_fn())[:10]
            for symbol in symbols:
                try:
                    oi = await self._client.get_open_interest(symbol)
                    row = {"symbol": symbol, "open_interest": float(oi), "ts": datetime.now(timezone.utc).isoformat()}
                    self._latest[symbol] = row
                    self._history[symbol].append(row)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.debug("OpenInterestPoller failed for %s: %s", symbol, exc)
            await asyncio.sleep(self._interval_seconds)

    def get_latest(self, symbol: str) -> Optional[dict]:
        return self._latest.get(symbol.upper())

    def get_history(self, symbol: str) -> list[dict]:
        return list(self._history.get(symbol.upper(), []))

    def get_state(self) -> dict:
        return {
            "running": self._running,
            "tracked_symbols": sorted(self._latest.keys()),
            "snapshot_count": len(self._latest),
        }
