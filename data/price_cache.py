"""
Latest price store per market. Updated from stream; consumed by scanner.
In-memory by default; interface supports Redis later. Rejects data older than max_age_seconds.
"""
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional, Dict, Any

from core.types import PriceSnapshot, SelectionPrice


class PriceCache:
    """In-memory cache of latest PriceSnapshot per market_id. Thread-safe for single-thread async."""

    def __init__(self, max_age_seconds: int = 2):
        self._cache: Dict[str, PriceSnapshot] = {}
        self._max_age_seconds = max_age_seconds

    def set_prices(self, snapshot: PriceSnapshot) -> None:
        """Store the latest snapshot for the market."""
        self._cache[snapshot.market_id] = snapshot

    def get_prices(self, market_id: str) -> Optional[PriceSnapshot]:
        """
        Return latest snapshot for market_id if present and not stale.
        Stale = timestamp older than max_age_seconds from now.
        """
        snapshot = self._cache.get(market_id)
        if snapshot is None:
            return None
        age = (datetime.now(timezone.utc) - snapshot.timestamp).total_seconds()
        if age > self._max_age_seconds:
            return None
        return snapshot

    def get_prices_by_regime(
        self,
        market_id: str,
        market_start: Optional[datetime],
        stale_prematch_seconds: int,
        stale_inplay_seconds: int,
    ) -> Optional[PriceSnapshot]:
        """
        Regime-aware staleness:
        - in-play markets use a tighter stale threshold
        - pre-match markets can tolerate slightly older snapshots
        """
        snapshot = self._cache.get(market_id)
        if snapshot is None:
            return None
        now = datetime.now(timezone.utc)
        in_play = False
        if market_start is not None:
            ms = market_start
            if ms.tzinfo is None:
                ms = ms.replace(tzinfo=timezone.utc)
            in_play = ms <= now
        threshold = stale_inplay_seconds if in_play else stale_prematch_seconds
        age = (now - snapshot.timestamp).total_seconds()
        if age > threshold:
            return None
        return snapshot

    def get_prices_ignore_stale(self, market_id: str) -> Optional[PriceSnapshot]:
        """Return latest snapshot without staleness check (e.g. for backtester)."""
        return self._cache.get(market_id)
