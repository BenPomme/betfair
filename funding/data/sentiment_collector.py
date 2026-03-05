"""
Periodic collector for market sentiment data:
  - Long/Short ratios (global accounts + top trader positions) from Binance Futures API
  - Fear & Greed Index from alternative.me

Stores data as append-only CSV files under data/funding_history/.
Follows project async patterns (asyncio throughout, asyncio.timeout on all I/O).
"""
import asyncio
import csv
import logging
import httpx
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Set

import config
from funding.utils.async_compat import async_timeout

logger = logging.getLogger(__name__)

# Data directories relative to project root
_DATA_DIR = Path("data/funding_history")
_LONG_SHORT_DIR = _DATA_DIR / "long_short_ratio"
_FEAR_GREED_DIR = _DATA_DIR / "fear_greed"

_LONG_SHORT_COLUMNS = [
    "timestamp",
    "long_short_ratio",
    "long_account",
    "short_account",
    "top_long_short_ratio",
    "top_long_account",
    "top_short_account",
]
_FEAR_GREED_COLUMNS = ["timestamp", "value", "value_classification"]

_FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1"
_HTTP_TIMEOUT_SECONDS = 10.0


class SentimentCollector:
    """Periodic collector for Binance long/short ratios and Fear & Greed index.

    Two independent polling loops are started by `start()`:
      1. Long/Short ratio loop — polls every config.LONG_SHORT_POLL_SECONDS (default 300s)
         for every symbol returned by watchlist_fn().
      2. Fear & Greed loop — polls every config.FEAR_GREED_POLL_SECONDS (default 900s)
         from the alternative.me public API.

    All data is written in append mode to CSV files under data/funding_history/.
    """

    def __init__(
        self,
        futures_client=None,
        watchlist_fn: Optional[Callable[[], Set[str]]] = None,
    ) -> None:
        """
        Args:
            futures_client: BinanceFuturesClient instance exposing
                            get_long_short_ratio() and get_top_long_short_position_ratio().
            watchlist_fn:   Zero-argument callable that returns a set of symbol strings
                            (e.g. {"BTCUSDT", "ETHUSDT"}). Called each polling cycle so
                            the watchlist can change at runtime.
        """
        self._futures_client = futures_client
        self._watchlist_fn: Callable[[], Set[str]] = watchlist_fn or (lambda: set())

        # In-memory state: latest values per symbol + fear/greed
        self._long_short_state: Dict[str, Dict[str, Any]] = {}
        self._fear_greed_state: Dict[str, Any] = {}

        # Asyncio task handles
        self._long_short_task: Optional[asyncio.Task] = None
        self._fear_greed_task: Optional[asyncio.Task]  = None
        self._running: bool = False

        # Create storage directories on init
        _LONG_SHORT_DIR.mkdir(parents=True, exist_ok=True)
        _FEAR_GREED_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start both collection loops as concurrent asyncio tasks."""
        if self._running:
            logger.warning("SentimentCollector already running — ignoring start()")
            return

        self._running = True
        self._long_short_task = asyncio.create_task(
            self._long_short_loop(), name="sentiment_long_short"
        )
        self._fear_greed_task = asyncio.create_task(
            self._fear_greed_loop(), name="sentiment_fear_greed"
        )
        logger.info(
            "SentimentCollector started (long/short every %ds, fear/greed every %ds)",
            getattr(config, "LONG_SHORT_POLL_SECONDS", 300),
            getattr(config, "FEAR_GREED_POLL_SECONDS", 900),
        )

    async def stop(self) -> None:
        """Cancel both polling loops and wait for them to finish."""
        self._running = False

        tasks = [t for t in (self._long_short_task, self._fear_greed_task) if t is not None]
        for task in tasks:
            task.cancel()

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        self._long_short_task = None
        self._fear_greed_task = None
        logger.info("SentimentCollector stopped")

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        """Return a snapshot of the current in-memory state.

        Returns:
            {
                "long_short": {symbol: {...}},
                "fear_greed": {...},
            }
        """
        return {
            "long_short": dict(self._long_short_state),
            "fear_greed": dict(self._fear_greed_state),
        }

    def get_latest_sentiment(self, symbol: str) -> dict:
        """Return the latest long/short data for a symbol merged with the fear/greed reading.

        Args:
            symbol: e.g. "BTCUSDT"

        Returns:
            Dict with long/short fields + fear/greed fields, or empty dict if no data yet.
        """
        result: Dict[str, Any] = {}
        ls = self._long_short_state.get(symbol)
        if ls:
            result.update(ls)
        fg = self._fear_greed_state
        if fg:
            result.update({f"fear_greed_{k}": v for k, v in fg.items()})
        return result

    def get_fear_greed(self) -> dict:
        """Return the latest Fear & Greed reading.

        Returns:
            {"timestamp": ..., "value": ..., "value_classification": ...}
            or empty dict if no data has been collected yet.
        """
        return dict(self._fear_greed_state)

    # ------------------------------------------------------------------
    # Polling loops
    # ------------------------------------------------------------------

    async def _long_short_loop(self) -> None:
        """Poll long/short ratios for every symbol in the watchlist."""
        poll_seconds: int = getattr(config, "LONG_SHORT_POLL_SECONDS", 300)

        while self._running:
            try:
                symbols = self._watchlist_fn()
                if not symbols:
                    logger.debug("Long/short loop: watchlist is empty, skipping cycle")
                else:
                    for symbol in symbols:
                        if not self._running:
                            break
                        await self._collect_long_short(symbol)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Unexpected error in long/short loop: %s", exc)

            try:
                await asyncio.sleep(poll_seconds)
            except asyncio.CancelledError:
                raise

    async def _fear_greed_loop(self) -> None:
        """Poll the Fear & Greed index from alternative.me."""
        poll_seconds: int = getattr(config, "FEAR_GREED_POLL_SECONDS", 900)

        while self._running:
            try:
                await self._collect_fear_greed()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Unexpected error in fear/greed loop: %s", exc)

            try:
                await asyncio.sleep(poll_seconds)
            except asyncio.CancelledError:
                raise

    # ------------------------------------------------------------------
    # Data collection helpers
    # ------------------------------------------------------------------

    async def _collect_long_short(self, symbol: str) -> None:
        """Fetch and store one long/short snapshot for a single symbol."""
        if self._futures_client is None:
            logger.debug("No futures_client configured — skipping long/short for %s", symbol)
            return

        timestamp = datetime.now(timezone.utc).isoformat()

        # --- Global long/short account ratio ---
        ls_ratio: Optional[str] = None
        ls_long_account: Optional[str] = None
        ls_short_account: Optional[str] = None
        try:
            async with async_timeout(_HTTP_TIMEOUT_SECONDS):
                ls_data = await self._futures_client.get_long_short_ratio(
                    symbol, period="5m", limit=1,
                )
            if ls_data:
                row = ls_data[0] if isinstance(ls_data, list) else ls_data
                ls_ratio = str(row.get("longShortRatio", ""))
                ls_long_account = str(row.get("longAccount", ""))
                ls_short_account = str(row.get("shortAccount", ""))
        except asyncio.TimeoutError:
            logger.warning("Timeout fetching long/short ratio for %s", symbol)
        except Exception as exc:
            logger.warning("Error fetching long/short ratio for %s: %s", symbol, exc)

        # --- Top trader long/short position ratio ---
        top_ratio: Optional[str] = None
        top_long: Optional[str] = None
        top_short: Optional[str] = None
        try:
            async with async_timeout(_HTTP_TIMEOUT_SECONDS):
                top_data = await self._futures_client.get_top_long_short_position_ratio(
                    symbol, period="5m", limit=1,
                )
            if top_data:
                row = top_data[0] if isinstance(top_data, list) else top_data
                top_ratio = str(row.get("longShortRatio", ""))
                top_long = str(row.get("longAccount", ""))
                top_short = str(row.get("shortAccount", ""))
        except asyncio.TimeoutError:
            logger.warning("Timeout fetching top long/short position ratio for %s", symbol)
        except Exception as exc:
            logger.warning("Error fetching top long/short position ratio for %s: %s", symbol, exc)

        # Only persist if we got at least the main ratio
        if ls_ratio is None and top_ratio is None:
            logger.debug("No long/short data received for %s — skipping write", symbol)
            return

        csv_row = {
            "timestamp": timestamp,
            "long_short_ratio": ls_ratio or "",
            "long_account": ls_long_account or "",
            "short_account": ls_short_account or "",
            "top_long_short_ratio": top_ratio or "",
            "top_long_account": top_long or "",
            "top_short_account": top_short or "",
        }

        # Update in-memory state
        self._long_short_state[symbol] = csv_row

        # Append to CSV
        path = _LONG_SHORT_DIR / f"{symbol}.csv"
        _append_csv(path, _LONG_SHORT_COLUMNS, csv_row)
        logger.debug("Long/short collected for %s: ratio=%s top_ratio=%s", symbol, ls_ratio, top_ratio)

    async def _collect_fear_greed(self) -> None:
        """Fetch and store one Fear & Greed snapshot from alternative.me."""
        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_SECONDS) as client:
                async with async_timeout(_HTTP_TIMEOUT_SECONDS + 2.0):
                    response = await client.get(_FEAR_GREED_URL)
                    response.raise_for_status()
                    payload = response.json()
        except asyncio.TimeoutError:
            logger.warning("Timeout fetching Fear & Greed index")
            return
        except httpx.HTTPStatusError as exc:
            logger.warning("HTTP error fetching Fear & Greed index: %s", exc)
            return
        except Exception as exc:
            logger.warning("Error fetching Fear & Greed index: %s", exc)
            return

        try:
            data_list = payload.get("data", [])
            if not data_list:
                logger.warning("Fear & Greed API returned empty data list")
                return

            entry = data_list[0]
            timestamp = datetime.now(timezone.utc).isoformat()
            value = str(entry.get("value", ""))
            classification = str(entry.get("value_classification", ""))
        except Exception as exc:
            logger.warning("Error parsing Fear & Greed response: %s", exc)
            return

        csv_row = {
            "timestamp": timestamp,
            "value": value,
            "value_classification": classification,
        }

        # Update in-memory state
        self._fear_greed_state = csv_row

        # Append to CSV
        path = _FEAR_GREED_DIR / "index.csv"
        _append_csv(path, _FEAR_GREED_COLUMNS, csv_row)
        logger.debug("Fear & Greed collected: value=%s (%s)", value, classification)


# ------------------------------------------------------------------
# CSV utility
# ------------------------------------------------------------------

def _append_csv(path: Path, columns: list, row: dict) -> None:
    """Append a single row to a CSV file, writing the header if the file is new/empty."""
    file_exists = path.exists() and path.stat().st_size > 0
    try:
        with open(path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            if not file_exists:
                writer.writeheader()
            writer.writerow({col: row.get(col, "") for col in columns})
    except Exception as exc:
        logger.error("Failed to write CSV %s: %s", path, exc)
