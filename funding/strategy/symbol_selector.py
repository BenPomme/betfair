"""
Symbol selector: filter and rank perpetual contracts by volume.
Maintains a watchlist of top N symbols worth monitoring.
"""
import logging
import time
from decimal import Decimal
from typing import Dict, List, Optional, Set

import config

logger = logging.getLogger(__name__)

# Cache refresh interval
_CACHE_TTL_SECONDS = 6 * 3600  # 6 hours


class SymbolSelector:
    """Select and maintain a watchlist of high-volume perpetual contracts."""

    def __init__(self, futures_client):
        self._client = futures_client
        self._watchlist: Set[str] = set()
        self._exchange_info: Dict[str, dict] = {}
        self._volume_data: Dict[str, Decimal] = {}
        self._last_refresh: float = 0

    async def refresh(self) -> Set[str]:
        """Refresh the watchlist from exchange info and 24h volume data."""
        now = time.monotonic()
        if self._watchlist and (now - self._last_refresh) < _CACHE_TTL_SECONDS:
            return self._watchlist

        logger.info("Refreshing symbol watchlist...")

        # Fetch exchange info for perpetual symbols
        symbols_info = await self._client.get_exchange_info()
        self._exchange_info = {s["symbol"]: s for s in symbols_info}

        # Fetch 24h volume data
        tickers = await self._client.get_ticker_24h()
        self._volume_data = {
            t["symbol"]: t["quote_volume"]
            for t in tickers
        }

        # Filter: PERPETUAL + TRADING + volume above minimum
        candidates: List[tuple] = []
        min_volume = config.FUNDING_MIN_24H_VOLUME_USD
        for symbol, info in self._exchange_info.items():
            volume = self._volume_data.get(symbol, Decimal("0"))
            if volume >= min_volume:
                candidates.append((symbol, volume))

        # Sort by volume descending, take top N
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_n = config.FUNDING_SYMBOLS_WATCHLIST_SIZE
        self._watchlist = {sym for sym, _ in candidates[:top_n]}

        self._last_refresh = now
        logger.info(
            "Watchlist updated: %d symbols (from %d perpetuals, min vol $%s)",
            len(self._watchlist),
            len(self._exchange_info),
            min_volume,
        )

        return self._watchlist

    @property
    def watchlist(self) -> Set[str]:
        return self._watchlist

    @property
    def volume_data(self) -> Dict[str, Decimal]:
        return self._volume_data

    def get_exchange_filters(self, symbol: str) -> Optional[Dict]:
        """Get exchange filters for a symbol (lot size, etc.)."""
        info = self._exchange_info.get(symbol)
        if info:
            return info.get("filters", {})
        return None
