"""
Poll Betfair list_market_book on an interval and push PriceSnapshot into PriceCache.
Allows the main loop to run without the streaming API (no stream activation required).
"""
import asyncio
import logging
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional

import config
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
    market_status = str(getattr(market_book, "status", "OPEN") or "OPEN")
    # For settlement-aware consumers we must keep runners on CLOSED markets
    # (statuses WINNER/LOSER are required for final outcome labels).
    if market_status == "CLOSED":
        selected_runners = list(market_book.runners)
    else:
        selected_runners = [
            r for r in market_book.runners
            if getattr(r, "status", "ACTIVE") == "ACTIVE"
        ]
    if not selected_runners:
        return None
    selections = []
    for runner in selected_runners:
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
            runner_status=str(getattr(runner, "status", "UNKNOWN") or "UNKNOWN"),
        ))
    if not selections:
        return None
    return PriceSnapshot(
        market_id=market_id,
        selections=tuple(selections),
        timestamp=datetime.now(timezone.utc),
        market_status=market_status,
    )


async def run_price_poller(
    client: Any,
    market_ids: List[str],
    price_cache: Any,
    interval_seconds: float = 2.0,
    is_running: Callable[[], bool] = lambda: True,
    runner_names: Optional[Dict[str, Dict[str, str]]] = None,
    on_metrics: Optional[Callable[[Dict[str, Any]], None]] = None,
    extra_market_ids_provider: Optional[Callable[[], List[str]]] = None,
) -> None:
    """
    Loop: every interval_seconds call list_market_book for market_ids, convert to
    PriceSnapshot, call price_cache.set_prices. Runs until is_running() returns False.
    """
    if not HAS_BETFAIR or not market_ids:
        return

    # Throughput is heavily impacted by batch and concurrency sizes.
    # We default higher than legacy values and degrade only when API rejects a batch.
    # Betfair market-data weight limit: sum(weight) * marketIds <= 200.
    # For EX_BEST_OFFERS weight=5, safe marketIds/request <= 40.
    BATCH_SIZE = min(40, max(1, int(getattr(config, "POLLER_BATCH_SIZE", 20))))
    CONCURRENT_BATCHES = max(1, int(getattr(config, "POLLER_CONCURRENT_BATCHES", 8)))
    REQUEST_TIMEOUT_SECONDS = float(getattr(config, "POLLER_REQUEST_TIMEOUT_SECONDS", 12.0))
    semaphore = asyncio.Semaphore(CONCURRENT_BATCHES)
    loop = asyncio.get_event_loop()
    zero_snapshot_cycles = 0
    zero_book_cycles = 0
    timeout_cycles = 0

    async def _fetch_batch(batch: List[str], metrics: Dict[str, Any]) -> None:
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
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                logger.warning("list_market_book timed out for batch %s", batch)
                metrics["timeouts"] += 1
                return
            except Exception as e:
                msg = str(e)
                # Adaptive split: if a batch is too large, split and retry recursively.
                if (
                    len(batch) > 1
                    and ("TOO_MUCH_DATA" in msg or "INVALID_INPUT_DATA" in msg)
                ):
                    mid = len(batch) // 2
                    await _fetch_batch(batch[:mid], metrics)
                    await _fetch_batch(batch[mid:], metrics)
                    return
                logger.exception("list_market_book failed: %s", e)
                metrics["errors"] += 1
                return

            books_list = result if isinstance(result, list) else []
            metrics["books_received"] += len(books_list)
            for book in books_list:
                mid = str(getattr(book, "market_id", ""))
                if not mid:
                    continue
                rn_map = runner_names.get(mid) if runner_names else None
                snapshot = _market_book_to_snapshot(mid, book, rn_map)
                if snapshot:
                    price_cache.set_prices(snapshot)
                    metrics["snapshots_set"] += 1

    while is_running():
        cycle_start = time.time()
        base_ids = list(market_ids)
        extra_ids: List[str] = []
        if extra_market_ids_provider is not None:
            try:
                extra_ids = [m for m in (extra_market_ids_provider() or []) if m]
            except Exception:
                extra_ids = []
        all_market_ids = list(dict.fromkeys(base_ids + extra_ids))
        batches = [all_market_ids[i:i + BATCH_SIZE] for i in range(0, len(all_market_ids), BATCH_SIZE)]
        metrics: Dict[str, Any] = {
            "requested_markets_base": len(base_ids),
            "requested_markets_extra": len(extra_ids),
            "requested_markets_total": len(all_market_ids),
            "batch_size": BATCH_SIZE,
            "concurrent_batches": CONCURRENT_BATCHES,
            "batches_total": len(batches),
            "books_received": 0,
            "snapshots_set": 0,
            "timeouts": 0,
            "errors": 0,
        }
        await asyncio.gather(*[_fetch_batch(batch, metrics) for batch in batches])
        zero_snapshot_cycles = zero_snapshot_cycles + 1 if metrics["snapshots_set"] == 0 else 0
        zero_book_cycles = zero_book_cycles + 1 if metrics["books_received"] == 0 else 0
        timeout_cycles = timeout_cycles + 1 if metrics["timeouts"] > 0 else 0
        metrics["zero_snapshot_cycles"] = zero_snapshot_cycles
        metrics["zero_book_cycles"] = zero_book_cycles
        metrics["timeout_cycles"] = timeout_cycles
        metrics["cycle_duration_sec"] = round(time.time() - cycle_start, 4)
        if on_metrics is not None:
            try:
                on_metrics(metrics)
            except Exception:
                pass
        await asyncio.sleep(interval_seconds)
