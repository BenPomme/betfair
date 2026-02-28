"""
Track open orders; cancel stale legs (>5s unmatched). In paper mode no-op (no live API).
Interface ready for live_executor to register orders.
"""
import asyncio
import logging
import time
from typing import List, Optional, Any

import config

logger = logging.getLogger(__name__)

STALE_ORDER_SECONDS = 5


class OrderMonitor:
    """
    Poll unmatched orders; cancel those older than STALE_ORDER_SECONDS.
    In paper mode: no-op (no API calls). Live mode: register orders from live_executor.
    """

    def __init__(self) -> None:
        self._order_ids: List[Any] = []  # (order_id, placed_at_ts) for live
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def register_orders(self, order_ids: List[Any]) -> None:
        """Called by live executor after placing orders. No-op in paper mode."""
        if config.PAPER_TRADING:
            return
        now = time.monotonic()
        for oid in order_ids:
            self._order_ids.append((oid, now))

    async def start(self) -> None:
        """Start background task to poll and cancel stale. No-op in paper mode."""
        if config.PAPER_TRADING:
            logger.debug("Order monitor: paper mode, not starting poll")
            return
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _poll_loop(self) -> None:
        """In live mode: poll Betfair for unmatched, cancel if older than STALE_ORDER_SECONDS."""
        if config.PAPER_TRADING:
            return
        while self._running:
            await asyncio.sleep(1)
            # Placeholder: live implementation will call Betfair list_current_orders
            # and cancel any in self._order_ids older than STALE_ORDER_SECONDS
            # No API call here until live_executor is implemented
