"""
Persist scan candidates (executed and rejected) for supervised learning.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional


def _serialize(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(v) for v in value]
    return value


class CandidateLogger:
    """Append-only JSONL logger partitioned by UTC day."""

    def __init__(self, base_dir: str = "data/candidates"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _file_for_today(self) -> Path:
        day = date.today().isoformat()
        return self.base_dir / f"{day}.jsonl"

    def log(self, record: dict) -> None:
        target = self._file_for_today()
        payload = _serialize(record)
        with target.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, separators=(",", ":")) + "\n")


def build_scan_record(
    market_id: str,
    event_name: str,
    has_snapshot: bool,
    reason: str,
    snapshot: Optional[Any] = None,
    opportunity: Optional[Any] = None,
    scored: Optional[Any] = None,
    features: Optional[Any] = None,
    executed: bool = False,
) -> dict:
    now = datetime.now(timezone.utc)
    record = {
        "ts": now,
        "market_id": market_id,
        "event_name": event_name,
        "has_snapshot": has_snapshot,
        "reason": reason,
        "executed": executed,
    }

    if snapshot is not None:
        back_prices = [s.best_back_price for s in snapshot.selections if s.best_back_price > Decimal("0")]
        lay_prices = [s.best_lay_price for s in snapshot.selections if s.best_lay_price > Decimal("0")]
        record["selection_count"] = len(snapshot.selections)
        record["overround_back"] = sum((Decimal("1") / p for p in back_prices), Decimal("0")) if back_prices else Decimal("0")
        record["overround_lay"] = sum((Decimal("1") / p for p in lay_prices), Decimal("0")) if lay_prices else Decimal("0")

    if opportunity is not None:
        record["arb_type"] = opportunity.arb_type
        record["net_profit_eur"] = opportunity.net_profit_eur
        record["net_roi_pct"] = opportunity.net_roi_pct
        record["total_stake_eur"] = opportunity.total_stake_eur

    if features is not None:
        micro = getattr(features, "microstructure", None)
        if micro is not None:
            record["spread_mean"] = getattr(micro, "spread_mean", 0)
            record["imbalance"] = getattr(micro, "imbalance", 0)
            record["depth_total_eur"] = getattr(micro, "depth_total_eur", 0)
            record["price_velocity"] = getattr(micro, "price_velocity", 0)
            record["short_volatility"] = getattr(micro, "short_volatility", 0)
            record["time_to_start_sec"] = getattr(micro, "time_to_start_sec", 0)
            record["in_play"] = getattr(micro, "in_play", False)
            record["weighted_spread"] = getattr(micro, "weighted_spread", 0)
            record["lay_back_ratio"] = getattr(micro, "lay_back_ratio", 0)
            record["top_of_book_concentration"] = getattr(micro, "top_of_book_concentration", 0)
            record["volume_momentum"] = getattr(micro, "volume_momentum", 0)
            record["back_lay_crossover"] = getattr(micro, "back_lay_crossover", 0)
            record["overround_distance"] = getattr(micro, "overround_distance", 0)
            record["depth_ratio_top3"] = getattr(micro, "depth_ratio_top3", 0)
            record["price_range"] = getattr(micro, "price_range", 0)
        record["selection_count"] = getattr(features, "selection_count", record.get("selection_count", 0))
        record["stake_eur"] = getattr(features, "stake_eur", record.get("total_stake_eur", 0))
        record["arb_type_feature"] = getattr(features, "arb_type", record.get("arb_type", ""))

    if scored is not None:
        record["model_version"] = scored.model_version
        record["decision"] = scored.decision
        record["edge_score"] = scored.edge_score
        record["fill_prob"] = scored.fill_prob
        record["expected_net_profit_eur"] = scored.expected_net_profit_eur
        record["dynamic_threshold_eur"] = scored.dynamic_threshold_eur
        record["order_policy"] = scored.order_policy
        record["ttl_seconds"] = scored.ttl_seconds
        record["prediction_influence"] = getattr(scored, "prediction_influence", "none")

    return record
