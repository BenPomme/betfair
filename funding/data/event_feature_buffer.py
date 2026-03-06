from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Deque, Dict, Optional


class EventFeatureBuffer:
    def __init__(self, max_points: int = 720) -> None:
        self._points: Dict[str, Deque[dict]] = defaultdict(lambda: deque(maxlen=max_points))

    def add(self, symbol: str, payload: dict) -> None:
        row = dict(payload)
        ts = row.get("ts")
        if isinstance(ts, str):
            row["ts"] = datetime.fromisoformat(ts)
        elif ts is None:
            row["ts"] = datetime.now(timezone.utc)
        self._points[symbol.upper()].append(row)

    def _latest_before(self, symbol: str, cutoff: datetime) -> Optional[dict]:
        buf = self._points.get(symbol.upper())
        if not buf:
            return None
        candidate = None
        for row in buf:
            if row["ts"] <= cutoff:
                candidate = row
            else:
                break
        return candidate

    def compute_metrics(self, symbol: str) -> dict:
        buf = self._points.get(symbol.upper())
        if not buf:
            return {}
        latest = buf[-1]
        now = latest["ts"]
        p1 = self._latest_before(symbol, now - timedelta(minutes=1))
        p5 = self._latest_before(symbol, now - timedelta(minutes=5))

        def _pct(cur: Optional[dict], prev: Optional[dict], key: str) -> float:
            if not cur or not prev:
                return 0.0
            prev_val = float(prev.get(key, 0.0) or 0.0)
            cur_val = float(cur.get(key, 0.0) or 0.0)
            if prev_val == 0.0:
                return 0.0
            return ((cur_val - prev_val) / prev_val) * 100.0

        spreads = [float(item.get("spread_bps", 0.0) or 0.0) for item in buf if item.get("spread_bps") is not None]
        depths = [float(item.get("depth_usd", 0.0) or 0.0) for item in buf if item.get("depth_usd") is not None]
        liqs = [float(item.get("liquidation_notional_usd", 0.0) or 0.0) for item in buf if item.get("liquidation_notional_usd") is not None]
        return {
            "price_return_1m_pct": round(_pct(latest, p1, "mark_price"), 6),
            "price_return_5m_pct": round(_pct(latest, p5, "mark_price"), 6),
            "open_interest_change_5m_pct": round(_pct(latest, p5, "open_interest"), 6),
            "avg_spread_bps": round(sum(spreads[-20:]) / len(spreads[-20:]), 6) if spreads else 0.0,
            "avg_depth_usd": round(sum(depths[-20:]) / len(depths[-20:]), 6) if depths else 0.0,
            "liquidation_burst_usd": round(sum(liqs[-20:]), 6) if liqs else 0.0,
            "latest": dict(latest),
        }

    def get_state(self) -> dict:
        return {"tracked_symbols": sorted(self._points.keys()), "buffer_sizes": {sym: len(buf) for sym, buf in self._points.items()}}
