"""
Binance Spot REST client for the long leg of funding rate arbitrage.
Wraps binance-connector Spot class with Decimal conversion and async support.
"""
import asyncio
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, Optional

from binance.spot import Spot

logger = logging.getLogger(__name__)


def _to_decimal(val: Any) -> Decimal:
    return Decimal(str(val))


class BinanceSpotClient:
    """Wrapper around Spot SDK for funding rate arbitrage."""

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        base_url: str = "https://testnet.binance.vision",
    ):
        kwargs: Dict[str, Any] = {"base_url": base_url}
        if api_key:
            kwargs["api_key"] = api_key
        if api_secret:
            kwargs["api_secret"] = api_secret
        self._client = Spot(**kwargs)

    async def get_price(self, symbol: str) -> Decimal:
        """Fetch current spot price for a symbol."""
        def _call():
            return self._client.ticker_price(symbol=symbol)

        result = await asyncio.to_thread(_call)
        return _to_decimal(result["price"])

    async def get_order_book(self, symbol: str, limit: int = 20) -> dict:
        """Fetch spot order book for a symbol."""
        def _call():
            return self._client.depth(symbol=symbol, limit=limit)

        result = await asyncio.to_thread(_call)
        return {
            "lastUpdateId": result.get("lastUpdateId", 0),
            "timestamp": datetime.now(timezone.utc),
            "bids": result.get("bids", []),
            "asks": result.get("asks", []),
        }

    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Optional[Decimal] = None,
        **kwargs: Any,
    ) -> dict:
        """Place a spot order."""
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

    async def get_account_balance(self, asset: str = "USDT") -> Decimal:
        """Fetch spot balance for a specific asset."""
        def _call():
            return self._client.account()

        result = await asyncio.to_thread(_call)
        for balance in result.get("balances", []):
            if balance["asset"] == asset:
                return _to_decimal(balance["free"])
        return Decimal("0")

    async def transfer_to_futures(self, asset: str, amount: Decimal) -> dict:
        """Transfer from spot wallet to futures wallet (type=1)."""
        def _call():
            return self._client.futures_transfer(
                asset=asset, amount=str(amount), type=1
            )

        return await asyncio.to_thread(_call)

    async def transfer_from_futures(self, asset: str, amount: Decimal) -> dict:
        """Transfer from futures wallet to spot wallet (type=2)."""
        def _call():
            return self._client.futures_transfer(
                asset=asset, amount=str(amount), type=2
            )

        return await asyncio.to_thread(_call)
