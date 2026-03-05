"""
Track open orders; cancel stale legs (>5s unmatched). In paper mode no-op (no live API).
Interface ready for live_executor to register orders.
"""
import asyncio
import logging
import time
from typing import List, Optional, Any

import config
from monitoring.alerting import alert_partial_fill, alert_stale_order_cancelled

logger = logging.getLogger(__name__)

STALE_ORDER_SECONDS = 5


class OrderMonitor:
    """
    Poll unmatched orders; cancel those older than STALE_ORDER_SECONDS.
    In paper mode: no-op (no API calls). Live mode: register orders from live_executor.
    """

    def __init__(self, client=None) -> None:
        self._client = client
        self._order_ids: List[Any] = []  # dict: {bet_id, market_id, placed_at}
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def register_orders(self, order_ids: List[Any]) -> None:
        """Called by live executor after placing orders. No-op in paper mode."""
        if config.PAPER_TRADING:
            return
        now = time.monotonic()
        for oid in order_ids:
            if isinstance(oid, dict):
                self._order_ids.append({
                    "bet_id": str(oid.get("bet_id", "")),
                    "market_id": str(oid.get("market_id", "")),
                    "placed_at": now,
                })
            else:
                self._order_ids.append({
                    "bet_id": str(oid),
                    "market_id": "",
                    "placed_at": now,
                })

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
            await asyncio.sleep(1.0)
            if self._client is None or not self._order_ids:
                continue
            try:
                current = self._client.betting.list_current_orders(order_by="BY_PLACE_TIME")
            except Exception as e:
                logger.exception("order monitor list_current_orders failed: %s", e)
                continue

            reports = getattr(current, "current_orders", None)
            if reports is None and isinstance(current, list):
                reports = current
            reports = reports or []
            by_bet_id = {str(getattr(r, "bet_id", "")): r for r in reports}

            next_tracked = []
            for tracked in self._order_ids:
                bet_id = tracked.get("bet_id", "")
                market_id = tracked.get("market_id", "")
                placed_at = float(tracked.get("placed_at", time.monotonic()))
                age = time.monotonic() - placed_at
                report = by_bet_id.get(bet_id)
                if report is None:
                    continue
                status = str(getattr(report, "status", "") or "")
                size_matched = float(getattr(report, "size_matched", 0.0) or 0.0)
                size_total = float(getattr(report, "size", 0.0) or 0.0)

                if status == "EXECUTION_COMPLETE":
                    continue

                if status == "EXECUTABLE" and size_matched > 0 and size_total > 0 and size_matched < size_total:
                    try:
                        alert_partial_fill(
                            {
                                "market_id": market_id or getattr(report, "market_id", ""),
                                "size_matched": size_matched,
                                "size_total": size_total,
                            }
                        )
                    except Exception:
                        pass

                if status == "EXECUTABLE" and age > STALE_ORDER_SECONDS:
                    try:
                        from betfairlightweight.filters import cancel_instruction
                        self._client.betting.cancel_orders(
                            market_id=market_id or getattr(report, "market_id", None),
                            instructions=[cancel_instruction(bet_id=bet_id)],
                        )
                        logger.warning("Cancelled stale order bet_id=%s market_id=%s age=%.2fs", bet_id, market_id, age)
                        try:
                            alert_stale_order_cancelled(
                                {"market_id": market_id, "bet_id": bet_id, "age_seconds": round(age, 2)}
                            )
                        except Exception:
                            pass
                        continue
                    except Exception as e:
                        logger.exception("Failed cancelling stale order bet_id=%s: %s", bet_id, e)

                next_tracked.append(tracked)
            self._order_ids = next_tracked
