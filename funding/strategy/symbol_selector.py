"""
Symbol selector: filter and rank perpetual contracts by volume.
Maintains a watchlist of top N symbols worth monitoring.
"""
import logging
import time
from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple

import config

logger = logging.getLogger(__name__)

# Cache refresh interval
_CACHE_TTL_SECONDS = 6 * 3600  # 6 hours


def _safe_decimal(value) -> Decimal:
    try:
        return Decimal(str(value))
    except Exception:
        return Decimal("0")


def compute_book_metrics(order_book: Optional[dict]) -> Optional[Dict[str, Decimal]]:
    """Compute basic spread and depth metrics from an order book snapshot."""
    if not order_book:
        return None
    bids = order_book.get("bids") or []
    asks = order_book.get("asks") or []
    if not bids or not asks:
        return None
    bid_price = _safe_decimal(bids[0][0])
    bid_qty = _safe_decimal(bids[0][1])
    ask_price = _safe_decimal(asks[0][0])
    ask_qty = _safe_decimal(asks[0][1])
    if bid_price <= 0 or ask_price <= 0 or ask_price < bid_price:
        return None
    mid = (bid_price + ask_price) / Decimal("2")
    spread_bps = ((ask_price - bid_price) / mid * Decimal("10000")) if mid > 0 else Decimal("0")
    bid_depth_usd = sum(_safe_decimal(px) * _safe_decimal(qty) for px, qty in bids)
    ask_depth_usd = sum(_safe_decimal(px) * _safe_decimal(qty) for px, qty in asks)
    return {
        "bid": bid_price,
        "ask": ask_price,
        "mid": mid,
        "spread_bps": spread_bps,
        "top_bid_notional_usd": bid_price * bid_qty,
        "top_ask_notional_usd": ask_price * ask_qty,
        "bid_depth_usd": bid_depth_usd,
        "ask_depth_usd": ask_depth_usd,
    }


def estimate_round_trip_cost_bps(
    spot_metrics: Dict[str, Decimal],
    perp_metrics: Dict[str, Decimal],
    fee_bps: Decimal,
) -> Decimal:
    return (
        spot_metrics["spread_bps"]
        + perp_metrics["spread_bps"]
        + Decimal(str(config.FUNDING_PAPER_SPOT_SLIPPAGE_BPS))
        + Decimal(str(config.FUNDING_PAPER_PERP_SLIPPAGE_BPS))
        + Decimal(str(config.FUNDING_PAPER_MIN_SPREAD_BPS_BUFFER))
        + abs(((perp_metrics["mid"] - spot_metrics["mid"]) / spot_metrics["mid"]) * Decimal("10000"))
        + fee_bps
    )


def qualify_symbol_for_trading(
    symbol: str,
    position_size: Decimal,
    expected_funding_payment: Decimal,
    spot_metrics: Optional[Dict[str, Decimal]],
    perp_metrics: Optional[Dict[str, Decimal]],
) -> Tuple[bool, str, Dict[str, Decimal]]:
    """Return whether a symbol passes hard execution-quality rails."""
    if spot_metrics is None or perp_metrics is None:
        return False, "missing_spot_market", {}

    basis_bps = ((perp_metrics["mid"] - spot_metrics["mid"]) / spot_metrics["mid"]) * Decimal("10000")
    fee_proxy_usd = position_size * Decimal("0.001")
    fee_bps = (fee_proxy_usd / position_size * Decimal("10000")) if position_size > 0 else Decimal("0")
    estimated_cost_bps = estimate_round_trip_cost_bps(spot_metrics, perp_metrics, fee_bps)
    net_expected_edge_usd = expected_funding_payment - (position_size * estimated_cost_bps / Decimal("10000"))

    min_top_book = position_size * Decimal(str(config.FUNDING_MIN_TOP_BOOK_NOTIONAL_MULTIPLE))
    min_depth = Decimal(str(config.FUNDING_MIN_DEPTH_USD))
    max_spread = Decimal(str(config.FUNDING_MAX_SPREAD_BPS))
    max_basis = Decimal(str(config.FUNDING_MAX_BASIS_BPS))
    max_cost = Decimal(str(config.FUNDING_MAX_ESTIMATED_ROUND_TRIP_COST_BPS))

    if spot_metrics["spread_bps"] > max_spread or perp_metrics["spread_bps"] > max_spread:
        return False, "spread_too_wide", {"basis_bps": basis_bps, "estimated_cost_bps": estimated_cost_bps}
    if abs(basis_bps) > max_basis:
        return False, "basis_too_wide", {"basis_bps": basis_bps, "estimated_cost_bps": estimated_cost_bps}
    if min(
        spot_metrics["top_bid_notional_usd"],
        spot_metrics["top_ask_notional_usd"],
        perp_metrics["top_bid_notional_usd"],
        perp_metrics["top_ask_notional_usd"],
    ) < min_top_book:
        return False, "depth_insufficient", {"basis_bps": basis_bps, "estimated_cost_bps": estimated_cost_bps}
    if min(
        spot_metrics["bid_depth_usd"],
        spot_metrics["ask_depth_usd"],
        perp_metrics["bid_depth_usd"],
        perp_metrics["ask_depth_usd"],
    ) < min_depth:
        return False, "depth_insufficient", {"basis_bps": basis_bps, "estimated_cost_bps": estimated_cost_bps}
    if estimated_cost_bps > max_cost:
        return False, "estimated_cost_too_high", {"basis_bps": basis_bps, "estimated_cost_bps": estimated_cost_bps}

    return True, "approved", {
        "basis_bps": basis_bps,
        "estimated_cost_bps": estimated_cost_bps,
        "net_expected_edge_usd": net_expected_edge_usd,
        "spot_spread_bps": spot_metrics["spread_bps"],
        "perp_spread_bps": perp_metrics["spread_bps"],
        "spot_bid": spot_metrics["bid"],
        "spot_ask": spot_metrics["ask"],
        "perp_bid": perp_metrics["bid"],
        "perp_ask": perp_metrics["ask"],
        "spot_bid_depth_usd": spot_metrics["bid_depth_usd"],
        "spot_ask_depth_usd": spot_metrics["ask_depth_usd"],
        "perp_bid_depth_usd": perp_metrics["bid_depth_usd"],
        "perp_ask_depth_usd": perp_metrics["ask_depth_usd"],
    }


def rank_qualified_opportunities(opportunities: List[dict]) -> List[dict]:
    """Rank by net expected edge first, then liquidity, then tighter market quality."""
    return sorted(
        opportunities,
        key=lambda item: (
            item.get("net_expected_edge_usd", Decimal("0")),
            item.get("liquidity_score", Decimal("0")),
            -abs(item.get("basis_bps", Decimal("0"))),
            -item.get("combined_spread_bps", Decimal("0")),
        ),
        reverse=True,
    )


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
