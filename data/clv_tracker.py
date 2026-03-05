"""
Closing Line Value tracker for prediction/arb entries.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from core.types import PriceSnapshot


class CLVTracker:
    """
    Track entry odds vs closing odds.
    CLV = closing_implied_prob - entry_implied_prob
    Positive CLV means better than close.
    """

    def __init__(self, log_dir: str = "data/clv"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._entries: Dict[str, dict] = {}
        self._closing: Dict[str, Dict[str, float]] = {}
        self._clv: Dict[str, float] = {}
        self._stats = {"total": 0, "positive": 0, "sum": 0.0}

    def _file_for_today(self) -> Path:
        return self.log_dir / f"{date.today().isoformat()}.jsonl"

    def _append_log(self, payload: dict) -> None:
        payload = {**payload, "ts": datetime.now(timezone.utc).isoformat()}
        with self._file_for_today().open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, separators=(",", ":")) + "\n")

    def record_entry(
        self,
        bet_id: str,
        market_id: str,
        selection_id: str,
        entry_odds: float,
        entry_timestamp: str,
    ) -> None:
        self._entries[bet_id] = {
            "market_id": market_id,
            "selection_id": selection_id,
            "entry_odds": float(entry_odds),
            "entry_timestamp": entry_timestamp,
        }
        self._append_log({"kind": "entry", "bet_id": bet_id, **self._entries[bet_id]})

    def record_closing_prices(self, market_id: str, snapshot: PriceSnapshot) -> None:
        if str(getattr(snapshot, "market_status", "OPEN")).upper() != "CLOSED":
            return
        prices: Dict[str, float] = {}
        for s in snapshot.selections:
            if s.best_back_price > 1:
                prices[s.selection_id] = float(s.best_back_price)
        if prices:
            self._closing[market_id] = prices
            self._append_log({"kind": "close", "market_id": market_id, "prices": prices})

    def compute_clv(self, bet_id: str) -> Optional[float]:
        if bet_id in self._clv:
            return self._clv[bet_id]
        entry = self._entries.get(bet_id)
        if entry is None:
            return None
        market_id = entry["market_id"]
        selection_id = entry["selection_id"]
        close = self._closing.get(market_id, {})
        if selection_id not in close:
            return None
        entry_odds = max(1.01, float(entry["entry_odds"]))
        close_odds = max(1.01, float(close[selection_id]))
        entry_imp = 1.0 / entry_odds
        close_imp = 1.0 / close_odds
        clv = close_imp - entry_imp
        self._clv[bet_id] = clv
        self._stats["total"] += 1
        self._stats["sum"] += clv
        if clv > 0:
            self._stats["positive"] += 1
        self._append_log({"kind": "clv", "bet_id": bet_id, "clv": clv})
        return clv

    def get_summary(self) -> dict:
        total = self._stats["total"]
        avg = (self._stats["sum"] / total) if total > 0 else 0.0
        pos = (self._stats["positive"] / total * 100.0) if total > 0 else 0.0
        return {
            "avg_clv": avg,
            "positive_clv_pct": pos,
            "total_tracked": total,
        }
