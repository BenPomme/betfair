from __future__ import annotations

import asyncio
import json
import logging
import random
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Callable, Deque, Dict, List, Optional, Set

import websockets

logger = logging.getLogger(__name__)


class AggTradeStream:
    """Track rolling taker-flow imbalance from Binance aggregate trades."""

    def __init__(
        self,
        ws_url: str,
        symbols_fn: Callable[[], Set[str]],
        window_seconds: int = 60,
        max_symbols: int = 10,
    ) -> None:
        self._ws_url = ws_url.rstrip("/")
        self._symbols_fn = symbols_fn
        self._window_seconds = max(10, int(window_seconds))
        self._max_symbols = max(1, int(max_symbols))
        self._running = False
        self._ws = None
        self._buffers: Dict[str, Deque[dict]] = defaultdict(lambda: deque(maxlen=5000))
        self._message_count = 0
        self._last_event_ts: Optional[str] = None
        self._reconnect_delay = 1.0

    async def start(self) -> None:
        self._running = True
        while self._running:
            symbols = sorted([s.lower() for s in self._symbols_fn()])[: self._max_symbols]
            if not symbols:
                await asyncio.sleep(2.0)
                continue
            url = self._ws_url + "/stream?streams=" + "/".join(f"{symbol}@aggTrade" for symbol in symbols)
            try:
                async with websockets.connect(url, ping_interval=180, ping_timeout=30, close_timeout=10) as ws:
                    self._ws = ws
                    self._reconnect_delay = 1.0
                    async for message in ws:
                        if not self._running:
                            break
                        self._process_message(message)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                if not self._running:
                    break
                delay = min(self._reconnect_delay + random.uniform(0, 0.2), 30.0)
                logger.warning("AggTradeStream reconnecting in %.1fs after error: %s", delay, exc)
                await asyncio.sleep(delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, 30.0)
            finally:
                self._ws = None

    async def stop(self) -> None:
        self._running = False
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    def _process_message(self, raw: str) -> None:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return
        data = payload.get("data", payload)
        symbol = str(data.get("s", "")).upper()
        if not symbol:
            return
        price = Decimal(str(data.get("p", "0")))
        qty = Decimal(str(data.get("q", "0")))
        if price <= 0 or qty <= 0:
            return
        side = "sell" if bool(data.get("m", False)) else "buy"
        notional = float(price * qty)
        ts = datetime.fromtimestamp(int(data.get("E", 0)) / 1000, tz=timezone.utc) if data.get("E") else datetime.now(timezone.utc)
        self._buffers[symbol].append({"ts": ts, "side": side, "notional": notional})
        self._message_count += 1
        self._last_event_ts = ts.isoformat()
        self._trim(symbol)

    def _trim(self, symbol: str) -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=self._window_seconds)
        buf = self._buffers.get(symbol)
        if not buf:
            return
        while buf and buf[0]["ts"] < cutoff:
            buf.popleft()

    def get_imbalance(self, symbol: str) -> float:
        symbol = symbol.upper()
        self._trim(symbol)
        buf = self._buffers.get(symbol)
        if not buf:
            return 0.0
        buy = sum(item["notional"] for item in buf if item["side"] == "buy")
        sell = sum(item["notional"] for item in buf if item["side"] == "sell")
        denom = buy + sell
        return round(((buy - sell) / denom), 6) if denom > 0 else 0.0

    def get_state(self) -> dict:
        return {
            "running": self._running,
            "message_count": self._message_count,
            "last_event_ts": self._last_event_ts,
            "tracked_symbols": sorted(self._buffers.keys()),
        }
