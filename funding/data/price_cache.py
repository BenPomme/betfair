"""
In-memory price cache for funding rate data.
Keyed by symbol → FundingSnapshot. TTL-based staleness check.
Follows data/price_cache.py pattern.
"""
from datetime import datetime, timezone
from typing import Dict, Optional

from funding.core.schemas import FundingSnapshot


class FundingPriceCache:
    """In-memory cache of latest FundingSnapshot per symbol."""

    def __init__(self, max_age_seconds: int = 10):
        self._cache: Dict[str, FundingSnapshot] = {}
        self._max_age_seconds = max_age_seconds

    def update(self, symbol: str, snapshot: FundingSnapshot) -> None:
        """Store the latest snapshot for a symbol."""
        self._cache[symbol] = snapshot

    def get_snapshot(self, symbol: str) -> Optional[FundingSnapshot]:
        """Return latest snapshot if not stale."""
        snapshot = self._cache.get(symbol)
        if snapshot is None:
            return None
        age = (datetime.now(timezone.utc) - snapshot.timestamp).total_seconds()
        if age > self._max_age_seconds:
            return None
        return snapshot

    def get_all_snapshots(self) -> Dict[str, FundingSnapshot]:
        """Return all non-stale snapshots."""
        now = datetime.now(timezone.utc)
        return {
            symbol: snap
            for symbol, snap in self._cache.items()
            if (now - snap.timestamp).total_seconds() <= self._max_age_seconds
        }

    def get_all_snapshots_ignore_stale(self) -> Dict[str, FundingSnapshot]:
        """Return all snapshots regardless of staleness."""
        return dict(self._cache)

    def remove(self, symbol: str) -> None:
        """Remove a symbol from cache."""
        self._cache.pop(symbol, None)

    def clear(self) -> None:
        """Clear all cached data."""
        self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)
