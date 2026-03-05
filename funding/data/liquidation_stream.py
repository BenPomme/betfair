"""
Binance Futures WebSocket stream for liquidation (forceOrder) events.
Subscribes to !forceOrder@arr@1s for all symbols bulk updates.

Note: The liquidation stream is only available on production endpoints —
not on Binance testnet. Always connects to wss://fstream.binance.com.

Follows the pattern from funding/data/market_data_stream.py with
exponential backoff reconnection and jitter.
"""
import asyncio
import json
import logging
import random
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import websockets

logger = logging.getLogger(__name__)

# Production-only URL — liquidation stream is not available on testnet
_LIQUIDATION_WS_URL = "wss://fstream.binance.com/ws/!forceOrder@arr@1s"

_JSONL_BASE = Path("data/funding_history/liquidations")


class LiquidationStream:
    """
    WebSocket manager for Binance Futures liquidation (forceOrder) events.

    Maintains a per-symbol in-memory ring buffer (maxlen=1000) and appends
    every event to a daily JSONL file at:
        data/funding_history/liquidations/YYYY-MM-DD.jsonl

    Wire format received from Binance:
        {
            "e": "forceOrder",
            "E": <event_time_ms>,
            "o": {
                "s":  symbol,
                "S":  side,           BUY | SELL
                "o":  order_type,     LIMIT
                "f":  time_in_force,  IOC
                "q":  quantity,
                "p":  price,
                "ap": avg_price,
                "X":  status,         FILLED
                "l":  last_filled_qty,
                "z":  filled_qty,
                "T":  trade_time_ms
            }
        }
    """

    def __init__(self) -> None:
        self._running = False
        self._ws = None
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 60.0
        self._event_count = 0
        self._last_event_time: Optional[datetime] = None

        # Per-symbol in-memory ring buffers
        self._buffers: Dict[str, deque] = {}

        # Ensure JSONL parent directory exists
        _JSONL_BASE.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

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

    async def stop(self) -> None:
        """Stop the WebSocket connection."""
        self._running = False
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
        logger.info("LiquidationStream stopped after %d events", self._event_count)

    def get_recent(self, symbol: str, minutes: int = 60) -> List[dict]:
        """
        Return events for *symbol* within the last *minutes* minutes.

        Events are sourced from the in-memory ring buffer; events older than
        the ring buffer are not included.
        """
        buf = self._buffers.get(symbol.upper())
        if not buf:
            return []
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        result = []
        for event in buf:
            ts = datetime.fromisoformat(event["timestamp"])
            if ts >= cutoff:
                result.append(event)
        return result

    def get_stats(self) -> dict:
        """
        Return summary statistics.

        Keys:
            total_events       -- int, cumulative count since process start
            events_per_symbol  -- Dict[str, int], buffer size per symbol
            last_event_time    -- ISO timestamp string or None
        """
        return {
            "total_events": self._event_count,
            "events_per_symbol": {sym: len(buf) for sym, buf in self._buffers.items()},
            "last_event_time": (
                self._last_event_time.isoformat() if self._last_event_time else None
            ),
        }

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
    def event_count(self) -> int:
        return self._event_count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _connect(self) -> None:
        """Open the WebSocket and process messages until disconnected."""
        logger.info("Connecting to %s", _LIQUIDATION_WS_URL)
        async with websockets.connect(
            _LIQUIDATION_WS_URL,
            ping_interval=180,
            ping_timeout=30,
            close_timeout=10,
        ) as ws:
            self._ws = ws
            self._reconnect_delay = 1.0  # Reset delay on successful connection
            logger.info("LiquidationStream connected")
            async for message in ws:
                if not self._running:
                    break
                self._process_message(message)

    def _process_message(self, raw: str) -> None:
        """Parse a forceOrder message and dispatch to buffer + JSONL."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON received from liquidation stream")
            return

        # Binance may wrap individual events or send them directly.
        # Handle both a single dict and a list of dicts.
        if isinstance(data, dict):
            events = [data]
        elif isinstance(data, list):
            events = data
        else:
            return

        for item in events:
            try:
                self._handle_event(item)
            except Exception as exc:
                logger.debug("Error processing liquidation event: %s — %s", item, exc)

    def _handle_event(self, item: dict) -> None:
        """
        Extract fields from a single forceOrder event, store in buffer,
        and append to the daily JSONL file.
        """
        if item.get("e") != "forceOrder":
            return

        order = item.get("o", {})
        symbol: str = order.get("s", "").upper()
        if not symbol:
            return

        now = datetime.now(timezone.utc)
        event = {
            "symbol": symbol,
            "side": order.get("S", ""),
            "order_type": order.get("o", ""),
            "time_in_force": order.get("f", ""),
            "quantity": float(order.get("q", 0)),
            "price": float(order.get("p", 0)),
            "avg_price": float(order.get("ap", 0)),
            "status": order.get("X", ""),
            "last_filled_qty": float(order.get("l", 0)),
            "filled_qty": float(order.get("z", 0)),
            "trade_time": int(order.get("T", 0)),
            "timestamp": now.isoformat(),
        }

        # Update in-memory ring buffer
        if symbol not in self._buffers:
            self._buffers[symbol] = deque(maxlen=1000)
        self._buffers[symbol].append(event)

        # Persist to daily JSONL
        self._append_jsonl(event, now)

        self._event_count += 1
        self._last_event_time = now

        logger.debug(
            "Liquidation: %s %s qty=%.4f price=%.2f",
            symbol,
            event["side"],
            event["filled_qty"],
            event["avg_price"],
        )

    def _append_jsonl(self, event: dict, ts: datetime) -> None:
        """Append *event* to the JSONL file for the date of *ts* (UTC)."""
        date_str = ts.strftime("%Y-%m-%d")
        path = _JSONL_BASE / f"{date_str}.jsonl"
        try:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event) + "\n")
        except OSError as exc:
            logger.error("Failed to write liquidation event to %s: %s", path, exc)
