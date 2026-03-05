"""
WebSocket client for Betfair streaming API. EX_ALL_OFFERS, EX_MARKET_DEF.
Reconnect with exponential backoff + jitter. Pushes price updates to a callback.
"""
import asyncio
import logging
import random
from datetime import datetime, timezone
from decimal import Decimal
from typing import Callable, List, Optional, Any

from core.types import PriceSnapshot, SelectionPrice

logger = logging.getLogger(__name__)

# Optional betfairlightweight
try:
    import betfairlightweight
    from betfairlightweight.resources import StreamListener
    HAS_BETFAIR = True
except ImportError:
    HAS_BETFAIR = False


def _market_book_to_snapshot(market_id: str, market_book: Any) -> Optional[PriceSnapshot]:
    """Convert a betfairlightweight market book to PriceSnapshot."""
    if not market_book or not getattr(market_book, "runners", None):
        return None
    selections = []
    for runner in market_book.runners:
        ex = getattr(runner, "ex", None)
        if not ex or not getattr(ex, "available_to_back", None):
            continue
        atb = ex.available_to_back
        if not atb:
            continue
        best_price = Decimal(str(atb[0].price)) if atb else Decimal("0")
        available = Decimal(str(atb[0].size)) if atb else Decimal("0")
        selection_id = str(getattr(runner, "selection_id", ""))
        name = getattr(runner, "runner_name", "") or selection_id
        selections.append(SelectionPrice(
            selection_id=selection_id,
            name=name,
            best_back_price=best_price,
            available_to_back=available,
            runner_status=str(getattr(runner, "status", "UNKNOWN") or "UNKNOWN"),
        ))
    if not selections:
        return None
    return PriceSnapshot(
        market_id=market_id,
        selections=tuple(selections),
        timestamp=datetime.now(timezone.utc),
        market_status=str(getattr(market_book, "status", "OPEN") or "OPEN"),
    )


class BetfairStreamClient:
    """
    Wraps betfairlightweight streaming. Runs in a thread; on each market book update,
    calls on_price_update(snapshot). Reconnects with exponential backoff + jitter.
    """

    def __init__(
        self,
        client: Any,
        on_price_update: Callable[[PriceSnapshot], None],
        market_ids: Optional[List[str]] = None,
    ):
        self._client = client
        self._on_price_update = on_price_update
        self._market_ids = market_ids or []
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def set_market_ids(self, market_ids: List[str]) -> None:
        self._market_ids = market_ids

    async def start(self) -> None:
        """Start streaming in a background task (blocking stream runs in executor)."""
        if not HAS_BETFAIR:
            logger.warning("betfairlightweight not installed; stream disabled")
            return
        self._running = True
        self._task = asyncio.create_task(self._run_stream_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _run_stream_loop(self) -> None:
        backoff = 1.0
        max_backoff = 60.0
        while self._running and self._market_ids:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: self._run_blocking_stream(),
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("Stream error: %s", e)
            if not self._running:
                break
            jitter = random.uniform(0, 1)
            delay = min(backoff + jitter, max_backoff)
            logger.info("Reconnecting in %.1fs", delay)
            await asyncio.sleep(delay)
            backoff = min(backoff * 2, max_backoff)

    def _run_blocking_stream(self) -> None:
        """Blocking: create stream, subscribe, iterate. Calls on_price_update from sync context."""
        listener = StreamListener(max_latency=None)
        stream = self._client.streaming.create_stream(listener)
        from betfairlightweight.filters import streaming_market_data_filter
        market_data_filter = streaming_market_data_filter(
            ladder_levels=3,
            fields=["EX_ALL_OFFERS", "EX_MARKET_DEF"],
        )
        stream.subscribe_to_markets(
            market_ids=self._market_ids,
            market_data_filter=market_data_filter,
        )
        for market_books in stream.get_generator():
            if not self._running:
                break
            for market_book in market_books:
                market_id = getattr(market_book, "market_id", None)
                if not market_id:
                    continue
                snapshot = _market_book_to_snapshot(str(market_id), market_book)
                if snapshot:
                    self._on_price_update(snapshot)
