from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from portfolio.state_store import PortfolioStateStore


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _parse_ts(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _bucket_price(price: float) -> str:
    if price < 0.15:
        return "tail_low"
    if price < 0.35:
        return "low"
    if price < 0.65:
        return "mid"
    if price < 0.85:
        return "high"
    return "tail_high"


def _bucket_liquidity(liquidity: float) -> str:
    if liquidity < 1000:
        return "thin"
    if liquidity < 5000:
        return "medium"
    return "deep"


def _bucket_spread(spread_bps: float) -> str:
    if spread_bps < 150:
        return "tight"
    if spread_bps < 350:
        return "normal"
    return "wide"


def _candidate_bucket(candidate: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
    context = dict(candidate.get("strategy_context") or {})
    return (
        str(candidate.get("reason") or "unknown"),
        str(context.get("sport") or "unknown"),
        _bucket_price(float(context.get("price", 0.0) or 0.0)),
        _bucket_liquidity(float(context.get("liquidity", 0.0) or 0.0)),
        _bucket_spread(float(context.get("spread_bps", 0.0) or 0.0)),
    )


class PolymarketBinaryRanker:
    def __init__(self, portfolio_id: str = "betfair_core") -> None:
        store = PortfolioStateStore(portfolio_id)
        self._store = store
        self._pending_path = store.runtime_dir / "polymarket_binary_pending.json"
        self._labels_path = store.runtime_dir / "polymarket_binary_labels.jsonl"
        self._model_path = store.runtime_dir / "polymarket_binary_model.json"

    def load_pending(self) -> Dict[str, Dict[str, Any]]:
        return self._store.read_json(self._pending_path, default={}) or {}

    def save_pending(self, pending: Dict[str, Dict[str, Any]]) -> None:
        self._store.write_json(self._pending_path, pending)

    def append_labels(self, rows: Iterable[Dict[str, Any]]) -> None:
        for row in rows:
            self._store.append_jsonl(self._labels_path, row)

    def load_labels(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        return self._store.read_jsonl(self._labels_path, limit=limit)

    def rebuild_model(self) -> Dict[str, Any]:
        labels = self.load_labels()
        buckets: Dict[str, Dict[str, Any]] = {}
        for row in labels:
            bucket = tuple(row.get("bucket") or [])
            if not bucket:
                continue
            key = "|".join(bucket)
            slot = buckets.setdefault(
                key,
                {
                    "bucket": list(bucket),
                    "count": 0,
                    "wins": 0,
                    "avg_realized_edge": 0.0,
                },
            )
            slot["count"] += 1
            realized_edge = float(row.get("realized_edge", 0.0) or 0.0)
            slot["wins"] += 1 if realized_edge > 0 else 0
            slot["avg_realized_edge"] += realized_edge
        ranked: List[Dict[str, Any]] = []
        for payload in buckets.values():
            count = int(payload["count"] or 0)
            if count <= 0:
                continue
            avg_edge = float(payload["avg_realized_edge"] or 0.0) / count
            win_rate = float(payload["wins"] or 0) / count
            confidence = min(1.0, count / 25.0)
            score = (avg_edge * 0.65) + ((win_rate - 0.5) * 0.35)
            ranked.append(
                {
                    "bucket": payload["bucket"],
                    "count": count,
                    "win_rate": round(win_rate, 4),
                    "avg_realized_edge": round(avg_edge, 6),
                    "confidence": round(confidence, 4),
                    "score": round(score, 6),
                }
            )
        ranked.sort(key=lambda item: (item["score"], item["count"]), reverse=True)
        model = {
            "generated_at": _utc_now_iso(),
            "labeled_examples": len(labels),
            "buckets": ranked[:100],
        }
        self._store.write_json(self._model_path, model)
        return model

    def load_model(self) -> Dict[str, Any]:
        return self._store.read_json(self._model_path, default={}) or self.rebuild_model()

    def score_candidate(self, candidate: Dict[str, Any], model: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        model_payload = model or self.load_model()
        bucket = list(_candidate_bucket(candidate))
        bucket_key = "|".join(bucket)
        bucket_map = {
            "|".join(item.get("bucket") or []): item
            for item in model_payload.get("buckets") or []
            if isinstance(item, dict)
        }
        learned = dict(bucket_map.get(bucket_key) or {})
        learned_count = int(learned.get("count", 0) or 0)
        confidence = float(learned.get("confidence", 0.0) or 0.0)
        avg_realized = float(learned.get("avg_realized_edge", 0.0) or 0.0)
        score = float(learned.get("score", 0.0) or 0.0)
        heuristic_edge = float(candidate.get("expected_edge", 0.0) or 0.0)
        empirical_edge = heuristic_edge
        if learned_count >= 5:
            empirical_edge = (heuristic_edge * (1.0 - confidence)) + (avg_realized * confidence)
        return {
            "bucket": bucket,
            "bucket_key": bucket_key,
            "learned_count": learned_count,
            "confidence": round(confidence, 4),
            "bucket_avg_realized_edge": round(avg_realized, 6),
            "bucket_score": round(score, 6),
            "empirical_expected_edge": round(empirical_edge, 6),
        }

    def update_labels(
        self,
        *,
        current_quotes: Iterable[Dict[str, Any]],
        min_elapsed_seconds: int = 120,
    ) -> Dict[str, Any]:
        pending = self.load_pending()
        quote_map = {
            str(row.get("market_slug") or row.get("selection_key") or row.get("event_slug") or ""): dict(row)
            for row in current_quotes
            if isinstance(row, dict)
        }
        kept: Dict[str, Dict[str, Any]] = {}
        completed_rows: List[Dict[str, Any]] = []
        now = _utc_now()
        for key, payload in pending.items():
            created_at = _parse_ts(payload.get("created_at"))
            if created_at is None:
                continue
            current_row = quote_map.get(key)
            if current_row is None or (now - created_at).total_seconds() < min_elapsed_seconds:
                kept[key] = payload
                continue
            candidate = dict(payload.get("candidate") or {})
            direction = 1.0 if str(candidate.get("reason") or "").lower() == "momentum" else -1.0
            entry_price = float(payload.get("entry_price", 0.0) or 0.0)
            current_price = float(
                current_row.get("last_trade_price", current_row.get("probability", 0.0)) or 0.0
            )
            spread_cost = float(payload.get("entry_spread_bps", 0.0) or 0.0) / 10000.0
            raw_move = direction * (current_price - entry_price)
            realized_edge = raw_move - spread_cost
            completed_rows.append(
                {
                    "ts": _utc_now_iso(),
                    "market_slug": key,
                    "candidate_id": candidate.get("candidate_id"),
                    "reason": candidate.get("reason"),
                    "sport": (candidate.get("strategy_context") or {}).get("sport"),
                    "entry_price": round(entry_price, 6),
                    "exit_price": round(current_price, 6),
                    "raw_move": round(raw_move, 6),
                    "spread_cost": round(spread_cost, 6),
                    "realized_edge": round(realized_edge, 6),
                    "bucket": list(_candidate_bucket(candidate)),
                }
            )
        if completed_rows:
            self.append_labels(completed_rows)
            self.rebuild_model()
        self.save_pending(kept)
        return {
            "completed": len(completed_rows),
            "remaining_pending": len(kept),
        }

    def track_candidates(self, candidates: Iterable[Dict[str, Any]]) -> Dict[str, int]:
        pending = self.load_pending()
        added = 0
        for candidate in candidates:
            key = str(candidate.get("market_id") or candidate.get("selection_key") or candidate.get("event_key") or "")
            if not key or key in pending:
                continue
            context = dict(candidate.get("strategy_context") or {})
            pending[key] = {
                "candidate": dict(candidate),
                "created_at": _utc_now_iso(),
                "entry_price": float(context.get("price", 0.0) or 0.0),
                "entry_spread_bps": float(context.get("spread_bps", 0.0) or 0.0),
            }
            added += 1
        self.save_pending(pending)
        return {"tracked": added, "pending_total": len(pending)}
