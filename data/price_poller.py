"""
Poll Betfair list_market_book on an interval and push PriceSnapshot into PriceCache.
Allows the main loop to run without the streaming API (no stream activation required).
"""
import asyncio
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional

from core.types import PriceSnapshot, SelectionPrice

logger = logging.getLogger(__name__)

try:
    import betfairlightweight
    from betfairlightweight.filters import price_projection, price_data
    HAS_BETFAIR = True
except ImportError:
    HAS_BETFAIR = False


def _market_book_to_snapshot(market_id: str, market_book: Any, runner_name_map: Optional[Dict[str, str]] = None) -> Optional[PriceSnapshot]:
    """Convert betfairlightweight market_book to PriceSnapshot.

    CRITICAL: Include ALL runners, even those with no back/lay data (price=0).
    Dropping runners silently causes the scanner to see fewer selections than
    the market actually has, producing false arb signals (e.g. 2 of 3 outcomes
    in a MATCH_ODDS market → overround 0.66 instead of 1.05).
    """
    if not market_book or not getattr(market_book, "runners", None):
        return None
    # Only include active runners (status ACTIVE)
    active_runners = [
        r for r in market_book.runners
        if getattr(r, "status", "ACTIVE") == "ACTIVE"
    ]
    if not active_runners:
        return None
    selections = []
    for runner in active_runners:
        selection_id = str(getattr(runner, "selection_id", ""))
        name = getattr(runner, "runner_name", "") or ""
        if not name and runner_name_map:
            name = runner_name_map.get(selection_id, "")
        if not name:
            name = selection_id
        ex = getattr(runner, "ex", None)
        atb = getattr(ex, "available_to_back", None) or [] if ex else []
        atl = getattr(ex, "available_to_lay", None) or [] if ex else []
        best_price = Decimal(str(atb[0].price)) if atb else Decimal("0")
        available = Decimal(str(atb[0].size)) if atb else Decimal("0")
        best_lay_price = Decimal(str(atl[0].price)) if atl else Decimal("0")
        available_lay = Decimal(str(atl[0].size)) if atl else Decimal("0")
        selections.append(SelectionPrice(
            selection_id=selection_id,
            name=name,
            best_back_price=best_price,
            available_to_back=available,
            best_lay_price=best_lay_price,
            available_to_lay=available_lay,
        ))
    if not selections:
        return None
    return PriceSnapshot(
        market_id=market_id,
        selections=tuple(selections),
        timestamp=datetime.now(timezone.utc),
    )


async def run_price_poller(
    client: Any,
    market_ids: List[str],
    price_cache: Any,
    interval_seconds: float = 2.0,
    is_running: Callable[[], bool] = lambda: True,
    runner_names: Optional[Dict[str, Dict[str, str]]] = None,
) -> None:
    """
    Loop: every interval_seconds call list_market_book for market_ids, convert to
    PriceSnapshot, call price_cache.set_prices. Runs until is_running() returns False.
    """
    if not HAS_BETFAIR or not market_ids:
        return

    # Betfair API returns TOO_MUCH_DATA if too many markets in one call.
    # Batch into chunks of BATCH_SIZE to stay within limits.
    BATCH_SIZE = 5
    batches = [market_ids[i:i + BATCH_SIZE] for i in range(0, len(market_ids), BATCH_SIZE)]

    CONCURRENT_BATCHES = 5
    semaphore = asyncio.Semaphore(CONCURRENT_BATCHES)
    loop = asyncio.get_event_loop()

    async def _fetch_batch(batch: List[str]) -> None:
        """Fetch a single batch with semaphore-bounded concurrency."""
        async with semaphore:
            if not is_running():
                return
            try:
                def _fetch(ids=batch) -> Any:
                    return client.betting.list_market_book(
                        market_ids=ids,
                        price_projection=price_projection(
                            price_data=price_data(ex_best_offers=True),
                        ),
                    )
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, _fetch),
                    timeout=30.0,
                )
            except asyncio.TimeoutError:
                logger.warning("list_market_book timed out for batch %s", batch)
                return
            except Exception as e:
                logger.exception("list_market_book failed: %s", e)
                return

            books_list = result if isinstance(result, list) else []
            for book in books_list:
                mid = str(getattr(book, "market_id", ""))
                if not mid:
                    continue
                rn_map = runner_names.get(mid) if runner_names else None
                snapshot = _market_book_to_snapshot(mid, book, rn_map)
                if snapshot:
                    price_cache.set_prices(snapshot)

    while is_running():
        await asyncio.gather(*[_fetch_batch(batch) for batch in batches])
        await asyncio.sleep(interval_seconds)
