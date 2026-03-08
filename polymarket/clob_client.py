from __future__ import annotations

import asyncio
from collections import Counter
import json
import logging
from threading import Event, Lock, Thread
from typing import Any, Dict, Iterable, List, Optional

import httpx
import websockets

import config
from polymarket.utils import as_millis_iso, clamp, parse_ts, to_float, utc_now_iso

logger = logging.getLogger(__name__)


def _normalize_levels(levels: Iterable[Dict[str, Any]], *, reverse: bool) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for level in levels or []:
        if not isinstance(level, dict):
            continue
        price = clamp(to_float(level.get("price"), 0.0))
        size = max(0.0, to_float(level.get("size"), 0.0))
        if price <= 0 or size <= 0:
            continue
        rows.append({"price": price, "size": size})
    rows.sort(key=lambda item: item["price"], reverse=reverse)
    return rows


def _book_payload(row: Dict[str, Any], observed_at: str | None = None) -> Dict[str, Any]:
    bids = _normalize_levels(row.get("bids") or row.get("buys") or [], reverse=True)
    asks = _normalize_levels(row.get("asks") or row.get("sells") or [], reverse=False)
    best_bid = bids[0]["price"] if bids else 0.0
    best_ask = asks[0]["price"] if asks else 0.0
    midpoint = 0.0
    if best_bid > 0 and best_ask > 0:
        midpoint = (best_bid + best_ask) / 2.0
    else:
        midpoint = clamp(to_float(row.get("last_trade_price") or row.get("lastTradePrice"), 0.0))
    return {
        "token_id": str(row.get("asset_id") or row.get("assetId") or ""),
        "market_id": str(row.get("market") or ""),
        "timestamp": as_millis_iso(row.get("timestamp")) or observed_at or utc_now_iso(),
        "best_bid": round(best_bid, 6),
        "best_ask": round(best_ask, 6),
        "midpoint": round(clamp(midpoint), 6),
        "spread": round(max(0.0, best_ask - best_bid), 6) if best_bid and best_ask else 0.0,
        "bids": bids,
        "asks": asks,
        "bid_depth": round(sum(level["size"] * level["price"] for level in bids[:5]), 6),
        "ask_depth": round(sum(level["size"] * level["price"] for level in asks[:5]), 6),
        "top_bid_size": round(bids[0]["size"], 6) if bids else 0.0,
        "top_ask_size": round(asks[0]["size"], 6) if asks else 0.0,
        "last_trade_price": round(clamp(to_float(row.get("last_trade_price") or row.get("lastTradePrice"), midpoint)), 6),
        "tick_size": to_float(row.get("tick_size") or row.get("tickSize"), 0.0),
        "min_order_size": to_float(row.get("min_order_size") or row.get("minOrderSize"), 0.0),
        "raw_hash": row.get("hash"),
        "source": "clob",
    }


class PolymarketClobClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        timeout_seconds: Optional[float] = None,
    ) -> None:
        self.base_url = (base_url or getattr(config, "POLYMARKET_QF_CLOB_HTTP_BASE_URL", "https://clob.polymarket.com")).rstrip("/")
        self.timeout_seconds = float(
            timeout_seconds if timeout_seconds is not None else getattr(config, "POLYMARKET_QF_HTTP_TIMEOUT_SECONDS", 10.0)
        )

    @staticmethod
    def parse_order_books(rows: Any) -> Dict[str, Dict[str, Any]]:
        if isinstance(rows, dict):
            rows = [rows]
        observed_at = utc_now_iso()
        books: Dict[str, Dict[str, Any]] = {}
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            payload = _book_payload(row, observed_at=observed_at)
            token_id = str(payload.get("token_id") or "")
            if token_id:
                books[token_id] = payload
        return books

    @staticmethod
    def parse_prices_history(payload: Any) -> List[Dict[str, Any]]:
        history = payload.get("history") if isinstance(payload, dict) else []
        rows: List[Dict[str, Any]] = []
        for item in history or []:
            if not isinstance(item, dict):
                continue
            ts = item.get("t")
            price = clamp(to_float(item.get("p"), 0.0))
            iso = as_millis_iso(ts)
            if iso is None:
                continue
            rows.append({"ts": iso, "price": price})
        rows.sort(key=lambda item: item["ts"])
        return rows

    @staticmethod
    def apply_ws_message(book_map: Dict[str, Dict[str, Any]], message: Any) -> Dict[str, Dict[str, Any]]:
        if isinstance(message, list):
            for row in message:
                PolymarketClobClient.apply_ws_message(book_map, row)
            return book_map
        if not isinstance(message, dict):
            return book_map
        event_type = str(message.get("event_type") or "").lower()
        if event_type == "book":
            payload = _book_payload(message)
            token_id = str(payload.get("token_id") or "")
            if token_id:
                payload["source"] = "clob_ws"
                book_map[token_id] = payload
            return book_map
        if event_type == "best_bid_ask":
            token_id = str(message.get("asset_id") or "")
            if not token_id:
                return book_map
            current = dict(book_map.get(token_id) or {})
            best_bid = clamp(to_float(message.get("best_bid"), current.get("best_bid", 0.0)))
            best_ask = clamp(to_float(message.get("best_ask"), current.get("best_ask", 0.0)))
            midpoint = (best_bid + best_ask) / 2.0 if best_bid and best_ask else current.get("midpoint", 0.0)
            current.update(
                {
                    "token_id": token_id,
                    "market_id": str(message.get("market") or current.get("market_id") or ""),
                    "timestamp": as_millis_iso(message.get("timestamp")) or utc_now_iso(),
                    "best_bid": round(best_bid, 6),
                    "best_ask": round(best_ask, 6),
                    "midpoint": round(clamp(midpoint), 6),
                    "spread": round(max(0.0, best_ask - best_bid), 6),
                    "source": "clob_ws",
                }
            )
            book_map[token_id] = current
            return book_map
        if event_type == "price_change":
            timestamp = as_millis_iso(message.get("timestamp")) or utc_now_iso()
            for change in message.get("price_changes") or []:
                if not isinstance(change, dict):
                    continue
                token_id = str(change.get("asset_id") or "")
                if not token_id:
                    continue
                current = dict(book_map.get(token_id) or {})
                best_bid = clamp(to_float(change.get("best_bid"), current.get("best_bid", 0.0)))
                best_ask = clamp(to_float(change.get("best_ask"), current.get("best_ask", 0.0)))
                midpoint = (best_bid + best_ask) / 2.0 if best_bid and best_ask else current.get("midpoint", 0.0)
                current.update(
                    {
                        "token_id": token_id,
                        "market_id": str(message.get("market") or current.get("market_id") or ""),
                        "timestamp": timestamp,
                        "best_bid": round(best_bid, 6),
                        "best_ask": round(best_ask, 6),
                        "midpoint": round(clamp(midpoint), 6),
                        "spread": round(max(0.0, best_ask - best_bid), 6),
                        "source": "clob_ws",
                    }
                )
                book_map[token_id] = current
            return book_map
        if event_type == "last_trade_price":
            token_id = str(message.get("asset_id") or "")
            if not token_id:
                return book_map
            current = dict(book_map.get(token_id) or {})
            current.update(
                {
                    "token_id": token_id,
                    "market_id": str(message.get("market") or current.get("market_id") or ""),
                    "timestamp": as_millis_iso(message.get("timestamp")) or utc_now_iso(),
                    "last_trade_price": round(clamp(to_float(message.get("price"), current.get("last_trade_price", 0.0))), 6),
                    "source": "clob_ws",
                }
            )
            book_map[token_id] = current
        return book_map

    def fetch_books(self, token_ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        tokens = [str(item).strip() for item in token_ids if str(item).strip()]
        if not tokens:
            return {}
        body = [{"token_id": token_id} for token_id in tokens]
        with httpx.Client(timeout=self.timeout_seconds, follow_redirects=True) as client:
            response = client.post(f"{self.base_url}/books", json=body)
            response.raise_for_status()
            payload = response.json()
        return self.parse_order_books(payload)

    def fetch_price_history(
        self,
        token_id: str,
        *,
        interval: str = "1m",
        fidelity: int = 60,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {
            "market": str(token_id),
            "interval": interval,
            "fidelity": fidelity,
        }
        if start_ts is not None:
            params["startTs"] = str(int(start_ts))
        if end_ts is not None:
            params["endTs"] = str(int(end_ts))
        with httpx.Client(timeout=self.timeout_seconds, follow_redirects=True) as client:
            response = client.get(f"{self.base_url}/prices-history", params=params)
            response.raise_for_status()
            payload = response.json()
        return self.parse_prices_history(payload)


class ClobMarketStream:
    def __init__(self, ws_url: str | None = None, *, enabled: bool = True) -> None:
        self._base_url = ws_url or getattr(config, "POLYMARKET_QF_CLOB_WS_URL", config.POLYMARKET_WS_URL)
        self._enabled = bool(enabled)
        self._lock = Lock()
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._requested_tokens: tuple[str, ...] = ()
        self._books: Dict[str, Dict[str, Any]] = {}
        self._event_counts: Counter[str] = Counter()
        self._connected = False
        self._last_message_ts: str | None = None
        self._last_error: str | None = None

    def _endpoint(self) -> str:
        if self._base_url.endswith("/market"):
            return self._base_url
        if self._base_url.endswith("/ws"):
            return f"{self._base_url}/market"
        return self._base_url.rstrip("/") + "/market"

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "enabled": self._enabled,
                "connected": self._connected,
                "token_count": len(self._requested_tokens),
                "last_message_ts": self._last_message_ts,
                "last_error": self._last_error,
                "event_counts": dict(self._event_counts),
                "books": dict(self._books),
            }

    def books(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return dict(self._books)

    def ensure_subscription(self, token_ids: Iterable[str]) -> None:
        tokens = tuple(sorted({str(item).strip() for item in token_ids if str(item).strip()}))
        if not self._enabled:
            return
        if tokens == self._requested_tokens and self._thread is not None and self._thread.is_alive():
            return
        self.stop()
        self._requested_tokens = tokens
        if not self._requested_tokens:
            return
        self._stop_event.clear()
        self._thread = Thread(target=self._run_thread, daemon=True, name="polymarket-clob-stream")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._thread = None
        with self._lock:
            self._connected = False

    async def _stream_loop(self, token_ids: tuple[str, ...]) -> None:
        url = self._endpoint()
        while not self._stop_event.is_set():
            try:
                async with websockets.connect(url, ping_interval=180, ping_timeout=30, close_timeout=10) as ws:
                    subscription = {
                        "type": "market",
                        "assets_ids": list(token_ids),
                        "custom_feature_enabled": True,
                    }
                    await ws.send(json.dumps(subscription))
                    with self._lock:
                        self._connected = True
                        self._last_error = None
                    while not self._stop_event.is_set():
                        try:
                            raw_message = await asyncio.wait_for(ws.recv(), timeout=5.0)
                        except asyncio.TimeoutError:
                            continue
                        payload = json.loads(raw_message)
                        event_type = ""
                        if isinstance(payload, dict):
                            event_type = str(payload.get("event_type") or "unknown")
                        with self._lock:
                            PolymarketClobClient.apply_ws_message(self._books, payload)
                            self._event_counts[event_type or "unknown"] += 1
                            self._last_message_ts = utc_now_iso()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Polymarket CLOB websocket reconnecting after error: %s", exc)
                with self._lock:
                    self._connected = False
                    self._last_error = str(exc)
                await asyncio.sleep(3.0)

    def _run_thread(self) -> None:
        try:
            asyncio.run(self._stream_loop(self._requested_tokens))
        except Exception:
            logger.exception("Polymarket CLOB websocket thread crashed")
