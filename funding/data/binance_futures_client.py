"""
Binance USDT-Margined Futures REST client.
Wraps binance-futures-connector UMFutures with Decimal conversion and async support.
"""
import asyncio
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from binance.um_futures import UMFutures

from funding.core.schemas import FundingSnapshot

logger = logging.getLogger(__name__)


def _to_decimal(val: Any) -> Decimal:
    """Safely convert any value to Decimal via string."""
    return Decimal(str(val))


def _ms_to_datetime(ms: int) -> datetime:
    """Convert millisecond timestamp to UTC datetime."""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


class BinanceFuturesClient:
    """Wrapper around UMFutures SDK for funding rate arbitrage."""

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        base_url: str = "https://testnet.binancefuture.com",
    ):
        kwargs: Dict[str, Any] = {"base_url": base_url}
        if api_key:
            kwargs["key"] = api_key
        if api_secret:
            kwargs["secret"] = api_secret
        self._client = UMFutures(**kwargs)

    async def get_premium_index(self, symbol: Optional[str] = None) -> List[FundingSnapshot]:
        """Fetch mark price + funding rate for one or all symbols."""
        kwargs = {}
        if symbol:
            kwargs["symbol"] = symbol

        def _call():
            return self._client.mark_price(**kwargs)

        result = await asyncio.to_thread(_call)
        if isinstance(result, dict):
            result = [result]
        snapshots = []
        for item in result:
            snapshots.append(FundingSnapshot(
                symbol=item["symbol"],
                funding_rate=_to_decimal(item.get("lastFundingRate", "0")),
                next_funding_time=_ms_to_datetime(int(item.get("nextFundingTime", 0))),
                mark_price=_to_decimal(item["markPrice"]),
                index_price=_to_decimal(item.get("indexPrice", item["markPrice"])),
                open_interest=Decimal("0"),  # Not available in premium index
                timestamp=datetime.now(timezone.utc),
            ))
        return snapshots

    async def get_funding_rate_history(
        self,
        symbol: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 100,
    ) -> List[dict]:
        """Fetch historical funding rates."""
        kwargs: Dict[str, Any] = {"symbol": symbol, "limit": limit}
        if start_time:
            kwargs["startTime"] = start_time
        if end_time:
            kwargs["endTime"] = end_time

        def _call():
            return self._client.funding_rate(**kwargs)

        result = await asyncio.to_thread(_call)
        return [
            {
                "symbol": item["symbol"],
                "funding_rate": _to_decimal(item["fundingRate"]),
                "funding_time": _ms_to_datetime(int(item["fundingTime"])),
                "mark_price": _to_decimal(item.get("markPrice", "0")),
            }
            for item in result
        ]

    async def get_funding_info(self) -> Dict[str, dict]:
        """Fetch funding rate caps/floors and interval hours per symbol."""
        def _call():
            return self._client.funding_info()

        result = await asyncio.to_thread(_call)
        return {
            item["symbol"]: {
                "adjusted_funding_rate_cap": _to_decimal(item.get("adjustedFundingRateCap", "0.0075")),
                "adjusted_funding_rate_floor": _to_decimal(item.get("adjustedFundingRateFloor", "-0.0075")),
                "funding_interval_hours": int(item.get("fundingIntervalHours", 8)),
            }
            for item in result
        }

    async def get_open_interest(self, symbol: str) -> Decimal:
        """Fetch current open interest for a symbol."""
        def _call():
            return self._client.open_interest(symbol=symbol)

        result = await asyncio.to_thread(_call)
        return _to_decimal(result.get("openInterest", "0"))

    async def get_ticker_24h(self, symbol: Optional[str] = None) -> List[dict]:
        """Fetch 24h ticker stats."""
        kwargs = {}
        if symbol:
            kwargs["symbol"] = symbol

        def _call():
            return self._client.ticker_24hr_price_change(**kwargs)

        result = await asyncio.to_thread(_call)
        if isinstance(result, dict):
            result = [result]
        return [
            {
                "symbol": item["symbol"],
                "volume": _to_decimal(item.get("volume", "0")),
                "quote_volume": _to_decimal(item.get("quoteVolume", "0")),
                "price_change_pct": _to_decimal(item.get("priceChangePercent", "0")),
                "last_price": _to_decimal(item.get("lastPrice", "0")),
            }
            for item in result
        ]

    async def get_account(self) -> dict:
        """Fetch futures account information, including wallet balance."""
        def _call():
            return self._client.account()

        return await asyncio.to_thread(_call)

    async def get_exchange_info(self) -> List[dict]:
        """Fetch exchange info: tradeable perpetual symbols with filters."""
        def _call():
            return self._client.exchange_info()

        result = await asyncio.to_thread(_call)
        symbols = []
        for s in result.get("symbols", []):
            if s.get("contractType") == "PERPETUAL" and s.get("status") == "TRADING":
                filters = {}
                for f in s.get("filters", []):
                    filters[f["filterType"]] = f
                symbols.append({
                    "symbol": s["symbol"],
                    "pair": s.get("pair", ""),
                    "base_asset": s.get("baseAsset", ""),
                    "quote_asset": s.get("quoteAsset", ""),
                    "price_precision": int(s.get("pricePrecision", 8)),
                    "quantity_precision": int(s.get("quantityPrecision", 8)),
                    "filters": filters,
                })
        return symbols

    async def get_klines(
        self, symbol: str, interval: str = "1h", limit: int = 100
    ) -> List[dict]:
        """Fetch candlestick data."""
        def _call():
            return self._client.klines(symbol=symbol, interval=interval, limit=limit)

        result = await asyncio.to_thread(_call)
        return [
            {
                "open_time": _ms_to_datetime(int(k[0])),
                "open": _to_decimal(k[1]),
                "high": _to_decimal(k[2]),
                "low": _to_decimal(k[3]),
                "close": _to_decimal(k[4]),
                "volume": _to_decimal(k[5]),
                "close_time": _ms_to_datetime(int(k[6])),
            }
            for k in result
        ]

    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Optional[Decimal] = None,
        **kwargs: Any,
    ) -> dict:
        """Place a futures order."""
        params: Dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
        }
        if quantity is not None:
            params["quantity"] = str(quantity)
        params.update(kwargs)

        def _call():
            return self._client.new_order(**params)

        return await asyncio.to_thread(_call)

    async def cancel_order(self, symbol: str, order_id: int) -> dict:
        """Cancel a futures order."""
        def _call():
            return self._client.cancel_order(symbol=symbol, orderId=order_id)

        return await asyncio.to_thread(_call)

    async def get_position_risk(self, symbol: Optional[str] = None) -> List[dict]:
        """Fetch current positions with PnL and liquidation price."""
        kwargs = {}
        if symbol:
            kwargs["symbol"] = symbol

        def _call():
            return self._client.get_position_risk(**kwargs)

        result = await asyncio.to_thread(_call)
        return [
            {
                "symbol": pos["symbol"],
                "position_amt": _to_decimal(pos.get("positionAmt", "0")),
                "entry_price": _to_decimal(pos.get("entryPrice", "0")),
                "mark_price": _to_decimal(pos.get("markPrice", "0")),
                "unrealized_pnl": _to_decimal(pos.get("unRealizedProfit", "0")),
                "liquidation_price": _to_decimal(pos.get("liquidationPrice", "0")),
                "leverage": int(pos.get("leverage", 1)),
                "margin_type": pos.get("marginType", "cross"),
            }
            for pos in result
        ]

    async def set_leverage(self, symbol: str, leverage: int) -> dict:
        """Set leverage for a symbol."""
        def _call():
            return self._client.change_leverage(symbol=symbol, leverage=leverage)

        return await asyncio.to_thread(_call)

    async def get_order_book(self, symbol: str, limit: int = 20) -> dict:
        """Fetch order book (bids and asks) for a symbol.

        Args:
            symbol: Perpetual contract symbol (e.g. "BTCUSDT").
            limit: Depth levels to return. Valid values: 5, 10, 20, 50, 100, 500, 1000.

        Returns:
            Dict with keys "bids" and "asks", each a list of [price, qty] string pairs,
            plus "lastUpdateId" and "timestamp" (UTC datetime).
        """
        def _call():
            return self._client.depth(symbol=symbol, limit=limit)

        result = await asyncio.to_thread(_call)
        return {
            "lastUpdateId": result.get("lastUpdateId", 0),
            "timestamp": datetime.now(timezone.utc),
            "bids": result.get("bids", []),
            "asks": result.get("asks", []),
        }

    async def get_long_short_ratio(
        self, symbol: str, period: str = "5m", limit: int = 30
    ) -> list:
        """Fetch global long/short account ratio."""
        def _call():
            return self._client.long_short_account_ratio(
                symbol=symbol, period=period, limit=limit
            )
        return await asyncio.to_thread(_call)

    async def get_top_long_short_position_ratio(
        self, symbol: str, period: str = "5m", limit: int = 30
    ) -> list:
        """Fetch top trader long/short position ratio."""
        def _call():
            return self._client.top_long_short_position_ratio(
                symbol=symbol, period=period, limit=limit
            )
        return await asyncio.to_thread(_call)
