"""
Periodic order book depth collector for Binance USDT-Margined Futures.

Polls the top N symbols (by watchlist) every DEPTH_POLL_SECONDS and stores
gzipped JSONL snapshots under data/funding_history/depth/{symbol}/YYYY-MM-DD.jsonl.gz.

Each line in the JSONL file records:
  {
    "timestamp": "<ISO 8601 UTC>",
    "bids": [[price, qty], ...],
    "asks": [[price, qty], ...],
    "bid_depth_usd": <float>,
    "ask_depth_usd": <float>,
    "bid_ask_imbalance": <float>   # (bid - ask) / (bid + ask), or 0.0 if denom == 0
  }

Usage:
    collector = DepthCollector(futures_client=client, watchlist_fn=lambda: {"BTCUSDT", ...})
    await collector.start()   # runs until stop() is called
    snapshot = collector.get_latest("BTCUSDT")
    state = collector.get_state()
"""
import asyncio
import gzip
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

import config

logger = logging.getLogger(__name__)

# Base directory for all depth snapshots (relative to cwd at startup)
_DEPTH_BASE = Path("data/funding_history/depth")


def _compute_depth_metrics(
    bids: List[List[str]], asks: List[List[str]]
) -> tuple[float, float, float]:
    """Return (bid_depth_usd, ask_depth_usd, bid_ask_imbalance).

    Args:
        bids: List of [price_str, qty_str] pairs (best bid first).
        asks: List of [price_str, qty_str] pairs (best ask first).

    Returns:
        Tuple of (bid_depth_usd, ask_depth_usd, bid_ask_imbalance).
    """
    bid_usd: float = sum(float(p) * float(q) for p, q in bids)
    ask_usd: float = sum(float(p) * float(q) for p, q in asks)
    denom = bid_usd + ask_usd
    imbalance = (bid_usd - ask_usd) / denom if denom > 0.0 else 0.0
    return bid_usd, ask_usd, imbalance


def _depth_path(symbol: str, dt: datetime) -> Path:
    """Return the gzipped JSONL path for a symbol and date.

    Example: data/funding_history/depth/BTCUSDT/2026-03-04.jsonl.gz
    """
    date_str = dt.strftime("%Y-%m-%d")
    return _DEPTH_BASE / symbol / f"{date_str}.jsonl.gz"


class DepthCollector:
    """Periodically collect and persist order book depth snapshots.

    Args:
        futures_client: An instance of BinanceFuturesClient (or compatible duck-type).
                        Must expose ``get_order_book(symbol, limit)`` as a coroutine.
        watchlist_fn: Callable that returns the current set of symbols to monitor.
                      Called once per collection cycle. Defaults to returning an empty set.
    """

    def __init__(
        self,
        futures_client=None,
        watchlist_fn: Optional[Callable[[], Set[str]]] = None,
    ) -> None:
        self._client = futures_client
        self._watchlist_fn: Callable[[], Set[str]] = watchlist_fn or (lambda: set())
        self._latest: Dict[str, dict] = {}
        self._running: bool = False
        self._task: Optional[asyncio.Task] = None

        # Ensure the base depth directory exists at construction time.
        _DEPTH_BASE.mkdir(parents=True, exist_ok=True)
        logger.debug("DepthCollector initialised; base dir: %s", _DEPTH_BASE.resolve())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background collection loop.

        Returns immediately — the loop runs as an asyncio Task.  Call
        ``await stop()`` to terminate it cleanly.
        """
        if self._running:
            logger.warning("DepthCollector.start() called but already running; ignoring.")
            return

        self._running = True
        self._task = asyncio.create_task(self._run(), name="depth_collector")
        logger.info(
            "DepthCollector started (poll_seconds=%s, top_n=%s).",
            config.DEPTH_POLL_SECONDS,
            config.DEPTH_TOP_N_SYMBOLS,
        )

    async def stop(self) -> None:
        """Signal the collection loop to stop and wait for it to finish."""
        self._running = False
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("DepthCollector stopped.")

    def get_latest(self, symbol: str) -> Optional[dict]:
        """Return the most recent depth snapshot for *symbol* from the in-memory cache.

        Returns ``None`` if no snapshot has been collected yet for that symbol.
        """
        return self._latest.get(symbol)

    def get_state(self) -> dict:
        """Return a dashboard-friendly summary of collector state.

        Returns:
            Dict containing:
              - ``running``: bool
              - ``symbols_tracked``: list of symbols with cached snapshots
              - ``snapshot_count``: number of symbols currently in cache
              - ``latest_timestamps``: mapping of symbol → ISO timestamp of last snapshot
        """
        return {
            "running": self._running,
            "symbols_tracked": list(self._latest.keys()),
            "snapshot_count": len(self._latest),
            "latest_timestamps": {
                sym: snap.get("timestamp")
                for sym, snap in self._latest.items()
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        """Main polling loop."""
        while self._running:
            try:
                await self._collect_cycle()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception("DepthCollector: unhandled error in collection cycle: %s", exc)

            # Wait for next cycle (respect stop signal)
            try:
                await asyncio.sleep(config.DEPTH_POLL_SECONDS)
            except asyncio.CancelledError:
                raise

    async def _collect_cycle(self) -> None:
        """Execute one collection cycle over the top N watchlist symbols."""
        watchlist: Set[str] = self._watchlist_fn()
        if not watchlist:
            logger.debug("DepthCollector: watchlist empty, skipping cycle.")
            return

        # Take up to DEPTH_TOP_N_SYMBOLS from the watchlist (deterministic order)
        symbols: List[str] = sorted(watchlist)[: config.DEPTH_TOP_N_SYMBOLS]
        logger.debug("DepthCollector: collecting depth for %d symbols.", len(symbols))

        for symbol in symbols:
            if not self._running:
                break
            try:
                await self._collect_symbol(symbol)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("DepthCollector: failed to collect %s: %s", symbol, exc)

    async def _collect_symbol(self, symbol: str) -> None:
        """Fetch order book for *symbol*, compute metrics, persist, and cache."""
        if self._client is None:
            logger.debug("DepthCollector: no futures_client configured; cannot collect %s.", symbol)
            return

        raw: dict = await self._client.get_order_book(symbol, limit=20)

        bids: List[List[str]] = raw.get("bids", [])
        asks: List[List[str]] = raw.get("asks", [])

        bid_depth_usd, ask_depth_usd, imbalance = _compute_depth_metrics(bids, asks)

        now: datetime = datetime.now(timezone.utc)
        record: dict = {
            "timestamp": now.isoformat(),
            "bids": bids,
            "asks": asks,
            "bid_depth_usd": bid_depth_usd,
            "ask_depth_usd": ask_depth_usd,
            "bid_ask_imbalance": imbalance,
        }

        # Persist to gzipped JSONL
        self._persist(symbol, record, now)

        # Update in-memory cache
        self._latest[symbol] = record

        logger.debug(
            "DepthCollector: %s bid_usd=%.2f ask_usd=%.2f imbalance=%.4f",
            symbol,
            bid_depth_usd,
            ask_depth_usd,
            imbalance,
        )

    def _persist(self, symbol: str, record: dict, dt: datetime) -> None:
        """Append *record* as a JSONL line to the day's gzipped file for *symbol*.

        Creates the per-symbol subdirectory on first write if necessary.
        """
        path: Path = _depth_path(symbol, dt)
        path.parent.mkdir(parents=True, exist_ok=True)

        line: str = json.dumps(record, separators=(",", ":")) + "\n"
        with gzip.open(path, "at", encoding="utf-8") as fh:
            fh.write(line)
