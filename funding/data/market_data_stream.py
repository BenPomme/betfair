"""
Binance Futures WebSocket stream for mark price and funding rate data.
Subscribes to !markPrice@arr@1s for all symbols bulk updates.
Follows data/betfair_stream.py pattern with exponential backoff reconnection.
"""
import asyncio
import json
import logging
import random
from datetime import datetime, timezone
from decimal import Decimal
from typing import Callable, Optional

import websockets

from funding.core.schemas import FundingSnapshot
from funding.data.price_cache import FundingPriceCache

logger = logging.getLogger(__name__)


def _to_decimal(val) -> Decimal:
    return Decimal(str(val))


def _ms_to_datetime(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


class MarketDataStream:
    """WebSocket manager for Binance Futures mark price + funding rate stream."""

    def __init__(
        self,
        ws_url: str = "wss://fstream.binancefuture.com",
        price_cache: Optional[FundingPriceCache] = None,
        on_snapshot: Optional[Callable[[FundingSnapshot], None]] = None,
    ):
        self._ws_url = ws_url.rstrip("/") + "/ws/!markPrice@arr@1s"
        self._price_cache = price_cache
        self._on_snapshot = on_snapshot
        self._running = False
        self._ws = None
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 60.0
        self._message_count = 0

    async def start(self) -> None:
        """Start the WebSocket connection with auto-reconnect."""
        self._running = True
        while self._running:
            try:
                await self._connect()
            except asyncio.CancelledError:
                break
            except Exception as e:
                if not self._running:
                    break
                jitter = random.uniform(0, self._reconnect_delay * 0.1)
                delay = min(self._reconnect_delay + jitter, self._max_reconnect_delay)
                logger.warning(
                    "WebSocket disconnected: %s. Reconnecting in %.1fs", e, delay
                )
                await asyncio.sleep(delay)
                self._reconnect_delay = min(
                    self._reconnect_delay * 2, self._max_reconnect_delay
                )

    async def _connect(self) -> None:
        """Connect and process messages."""
        logger.info("Connecting to %s", self._ws_url)
        async with websockets.connect(
            self._ws_url,
            ping_interval=180,
            ping_timeout=30,
            close_timeout=10,
        ) as ws:
            self._ws = ws
            self._reconnect_delay = 1.0  # Reset on successful connection
            logger.info("WebSocket connected")
            async for message in ws:
                if not self._running:
                    break
                self._process_message(message)

    def _process_message(self, raw: str) -> None:
        """Parse mark price array message and update cache."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON from WebSocket")
            return

        if not isinstance(data, list):
            return

        self._message_count += 1
        now = datetime.now(timezone.utc)

        for item in data:
            try:
                symbol = item.get("s", "")
                if not symbol:
                    continue
                snapshot = FundingSnapshot(
                    symbol=symbol,
                    funding_rate=_to_decimal(item.get("r", "0")),
                    next_funding_time=_ms_to_datetime(int(item.get("T", 0))),
                    mark_price=_to_decimal(item.get("p", "0")),
                    index_price=_to_decimal(item.get("i", "0")),
                    open_interest=Decimal("0"),
                    timestamp=now,
                )
                if self._price_cache:
                    self._price_cache.update(symbol, snapshot)
                if self._on_snapshot:
                    self._on_snapshot(snapshot)
            except Exception as e:
                logger.debug("Error parsing mark price for %s: %s", item.get("s"), e)

    async def stop(self) -> None:
        """Stop the WebSocket connection."""
        self._running = False
        if self._ws:
            await self._ws.close()
            self._ws = None
        logger.info("WebSocket stopped after %d messages", self._message_count)

    @property
    def is_connected(self) -> bool:
        if self._ws is None:
            return False
        try:
            from websockets.protocol import State
            return self._ws.state == State.OPEN
        except Exception:
            return self._ws is not None

    @property
    def message_count(self) -> int:
        return self._message_count
