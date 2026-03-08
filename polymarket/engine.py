from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict, List

import config
from polymarket.clob_client import ClobMarketStream, PolymarketClobClient
from polymarket.features import build_feature_rows, summarize_feature_rows
from polymarket.gamma_client import PolymarketGammaClient
from polymarket.labels import QuantumFoldLabelStore
from polymarket.model_league import QuantumFoldModelLeague
from polymarket.paper_executor import PolymarketPaperExecutor
from polymarket.utils import clamp, parse_ts, to_float, utc_now, utc_now_iso


class PolymarketQuantumFoldEngine:
    def __init__(self, runtime_dir: str | Path, *, initial_balance: float) -> None:
        self.runtime_dir = Path(runtime_dir)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.initial_balance = float(initial_balance)
        self.gamma = PolymarketGammaClient()
        self.clob = PolymarketClobClient()
        self.label_horizons = [
            int(value)
            for value in str(getattr(config, "POLYMARKET_QF_LABEL_HORIZONS_SECONDS", "120,600,1800")).split(",")
            if str(value).strip()
        ]
        self.primary_horizon = int(getattr(config, "POLYMARKET_QF_PRIMARY_HORIZON_SECONDS", 600))
        self.min_edge_after_costs = float(getattr(config, "POLYMARKET_QF_MIN_EDGE_AFTER_COSTS", 0.015))
        self.max_trade_hold_seconds = int(getattr(config, "POLYMARKET_QF_MAX_HOLD_SECONDS", 1800))
        self.example_interval_seconds = int(getattr(config, "POLYMARKET_QF_EXAMPLE_INTERVAL_SECONDS", 60))
        self.max_tracked_tokens = int(getattr(config, "POLYMARKET_QF_MAX_TRACKED_TOKENS", 18))
        self.history_refresh_seconds = int(getattr(config, "POLYMARKET_QF_HISTORY_REFRESH_SECONDS", 60))
        self.history_interval = str(getattr(config, "POLYMARKET_QF_HISTORY_INTERVAL", "1m"))
        self.history_fidelity = int(getattr(config, "POLYMARKET_QF_HISTORY_FIDELITY", 60))
        self.stale_quote_seconds = int(getattr(config, "POLYMARKET_QF_STALE_QUOTE_SECONDS", 30))
        self.fee_bps = float(getattr(config, "POLYMARKET_QF_FEE_BUFFER_BPS", 20.0))
        self.queue_penalty_bps = float(getattr(config, "POLYMARKET_QF_QUEUE_PENALTY_BPS", 8.0))
        self.gamma_snapshot: Dict[str, Any] = {}
        self.quote_map: Dict[str, Dict[str, Any]] = {}
        self.history_cache: Dict[str, Dict[str, Any]] = {}
        self.events: List[Dict[str, Any]] = []
        self.balance_history: List[Dict[str, Any]] = []
        self.raw_snapshots_path = self.runtime_dir / "quantum_fold_raw_snapshots.jsonl"
        self.label_store = QuantumFoldLabelStore(self.runtime_dir, horizons=self.label_horizons)
        self.model_league = QuantumFoldModelLeague("polymarket_quantum_fold", starting_balance=initial_balance)
        self.executor = PolymarketPaperExecutor(
            starting_balance=initial_balance,
            fee_bps=self.fee_bps,
            queue_penalty_bps=self.queue_penalty_bps,
            max_open_positions=int(getattr(config, "POLYMARKET_QF_MAX_OPEN_POSITIONS", 5)),
            max_notional_per_trade=float(getattr(config, "POLYMARKET_QF_MAX_NOTIONAL_PER_TRADE_USD", 150.0)),
            max_positions_per_event=int(getattr(config, "POLYMARKET_QF_MAX_POSITIONS_PER_EVENT", 1)),
            drawdown_halt_pct=float(getattr(config, "POLYMARKET_QF_DRAWDOWN_HALT_PCT", 6.0)),
        )
        self.stream = ClobMarketStream(
            enabled=bool(getattr(config, "POLYMARKET_QF_CLOB_WS_ENABLED", True)),
        )
        self._running = False
        self._scan_count = 0
        self._tracked_examples = 0
        self._last_example_at: Dict[str, str] = {}
        self._rejections: Counter[str] = Counter()
        self._last_state: Dict[str, Any] = {}

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False
        self.stream.stop()

    def _append_snapshot(self, payload: Dict[str, Any]) -> None:
        with self.raw_snapshots_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")

    def _merge_quotes(
        self,
        gamma_tokens: List[Dict[str, Any]],
        rest_books: Dict[str, Dict[str, Any]],
        ws_books: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        for token in gamma_tokens:
            token_id = str(token.get("token_id") or "")
            rest = dict(rest_books.get(token_id) or {})
            ws = dict(ws_books.get(token_id) or {})
            book = rest
            rest_ts = parse_ts(rest.get("timestamp"))
            ws_ts = parse_ts(ws.get("timestamp"))
            if ws and (rest_ts is None or (ws_ts is not None and ws_ts >= rest_ts)):
                book = {**rest, **ws}
            midpoint = clamp(to_float(book.get("midpoint"), to_float(token.get("gamma_price"), 0.0)))
            best_bid = clamp(to_float(book.get("best_bid"), to_float(token.get("best_bid"), 0.0)))
            best_ask = clamp(to_float(book.get("best_ask"), to_float(token.get("best_ask"), 0.0)))
            if midpoint <= 0 and best_bid > 0 and best_ask > 0:
                midpoint = (best_bid + best_ask) / 2.0
            merged.append(
                {
                    **token,
                    **book,
                    "midpoint": round(midpoint, 6),
                    "best_bid": round(best_bid, 6),
                    "best_ask": round(best_ask, 6),
                    "book_timestamp": book.get("timestamp") or token.get("observed_at"),
                    "source": book.get("source", "gamma"),
                }
            )
        return merged

    def _refresh_histories(self, token_rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        now = utc_now()
        result: Dict[str, List[Dict[str, Any]]] = {}
        tracked = sorted(
            token_rows,
            key=lambda item: (to_float(item.get("liquidity"), 0.0), to_float(item.get("volume_24hr"), 0.0)),
            reverse=True,
        )[: self.max_tracked_tokens]
        for row in tracked:
            token_id = str(row.get("token_id") or "")
            if not token_id:
                continue
            cached = self.history_cache.get(token_id) or {}
            cached_at = parse_ts(cached.get("fetched_at"))
            if cached_at is None or (now - cached_at).total_seconds() >= self.history_refresh_seconds:
                try:
                    history = self.clob.fetch_price_history(
                        token_id,
                        interval=self.history_interval,
                        fidelity=self.history_fidelity,
                    )
                except Exception:
                    history = list(cached.get("history") or [])
                self.history_cache[token_id] = {
                    "fetched_at": utc_now_iso(),
                    "history": history[-120:],
                }
            result[token_id] = list((self.history_cache.get(token_id) or {}).get("history") or [])
        return result

    def _trade_features(self, feature_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ranked = sorted(
            feature_rows,
            key=lambda item: (
                to_float(item.get("folding_confidence"), 0.0),
                to_float(item.get("depth_total_usd"), 0.0),
                -to_float(item.get("spread_bps"), 0.0),
            ),
            reverse=True,
        )[: self.max_tracked_tokens]
        return ranked

    def _track_examples(self, feature_rows: List[Dict[str, Any]]) -> Dict[str, int]:
        examples: List[Dict[str, Any]] = []
        now = utc_now()
        for row in feature_rows:
            token_id = str(row.get("token_id") or "")
            last_seen = parse_ts(self._last_example_at.get(token_id))
            if last_seen is not None and (now - last_seen).total_seconds() < self.example_interval_seconds:
                continue
            model_predictions = self.model_league.track_example(row)
            example_id = f"{token_id}:{int(now.timestamp())}"
            examples.append(
                {
                    "example_id": example_id,
                    "token_id": token_id,
                    "market_slug": row.get("market_slug"),
                    "event_slug": row.get("event_slug"),
                    "tracked_at": utc_now_iso(),
                    "entry_midpoint": round(to_float(row.get("midpoint"), 0.0), 6),
                    "cost_buffer": round(to_float(row.get("cost_buffer"), 0.0), 6),
                    "features": {
                        key: row.get(key)
                        for key in [
                            "coherence_score",
                            "folding_confidence",
                            "orderbook_imbalance",
                            "momentum_bias",
                            "energy_proxy",
                            "spread_bps",
                            "gamma_delta",
                            "return_2m",
                            "basin_depth",
                            "relaxation_speed",
                        ]
                    },
                    "model_predictions": model_predictions,
                }
            )
            self._last_example_at[token_id] = utc_now_iso()
        result = self.label_store.track_examples(examples)
        self._tracked_examples += int(result.get("tracked", 0) or 0)
        return result

    def _select_execution_candidate(self, feature_rows: List[Dict[str, Any]]) -> Dict[str, Any] | None:
        best: Dict[str, Any] | None = None
        for row in feature_rows:
            predictions = self.model_league.predict_all(row)
            hybrid_probability = to_float(predictions.get("hybrid_transition"), 0.5)
            edge_after_costs = hybrid_probability - 0.5 - to_float(row.get("cost_buffer"), 0.0)
            if edge_after_costs < self.min_edge_after_costs:
                self._rejections["edge_after_costs"] += 1
                continue
            if to_float(row.get("quote_freshness_sec"), self.stale_quote_seconds) > self.stale_quote_seconds:
                self.executor.stale_halt_count += 1
                self._rejections["stale_quote"] += 1
                continue
            if best is None or edge_after_costs > to_float(best.get("edge_after_costs"), -1.0):
                best = {**row, "hybrid_probability": hybrid_probability, "edge_after_costs": round(edge_after_costs, 6)}
        return best

    def _manage_positions(self) -> None:
        for position in list(self.executor.open_positions):
            quote = dict(self.quote_map.get(str(position.get("token_id") or "")) or {})
            if not quote:
                continue
            opened_at = parse_ts(position.get("opened_at"))
            age_seconds = (utc_now() - opened_at).total_seconds() if opened_at is not None else 0.0
            close_reason = None
            if to_float(quote.get("quote_freshness_sec"), 0.0) > self.stale_quote_seconds:
                close_reason = "stale_quote"
            elif bool(quote.get("resolved") or quote.get("closed")):
                close_reason = "market_resolved"
            elif age_seconds >= self.max_trade_hold_seconds:
                close_reason = "time_stop"
            elif to_float(quote.get("best_bid"), 0.0) >= min(0.99, to_float(position.get("entry_price"), 0.0) + 0.05):
                close_reason = "take_profit"
            elif to_float(quote.get("best_bid"), 0.0) <= max(0.01, to_float(position.get("entry_price"), 0.0) - 0.04):
                close_reason = "stop_loss"
            if close_reason:
                trade = self.executor.close_trade(position, quote, reason=close_reason)
                self.events.append({"kind": "trade_closed", "data": trade})

    def _open_trade(self, feature_row: Dict[str, Any], quote_map: Dict[str, Dict[str, Any]]) -> None:
        approved, reason = self.executor.can_open(feature_row, quote_map)
        if not approved:
            self._rejections[reason] += 1
            return
        trade = self.executor.open_trade(
            feature_row,
            score_probability=to_float(feature_row.get("hybrid_probability"), 0.5),
            notional_usd=float(getattr(config, "POLYMARKET_QF_MAX_NOTIONAL_PER_TRADE_USD", 150.0)),
        )
        self.events.append({"kind": "trade_opened", "data": trade})

    def step(self) -> Dict[str, Any]:
        self._scan_count += 1
        now_iso = utc_now_iso()
        gamma_snapshot = self.gamma.fetch_snapshot()
        self.gamma_snapshot = gamma_snapshot
        gamma_tokens = list(gamma_snapshot.get("tokens") or [])
        top_tokens = [
            str(item.get("token_id") or "")
            for item in sorted(
                gamma_tokens,
                key=lambda row: (to_float(row.get("liquidity"), 0.0), to_float(row.get("volume_24hr"), 0.0)),
                reverse=True,
            )[: self.max_tracked_tokens]
            if str(item.get("token_id") or "")
        ]
        rest_books = self.clob.fetch_books(top_tokens)
        self.stream.ensure_subscription(top_tokens)
        ws_books = (self.stream.snapshot().get("books") or {}) if getattr(config, "POLYMARKET_QF_CLOB_WS_ENABLED", True) else {}
        merged_quotes = self._merge_quotes(gamma_tokens, rest_books, ws_books)
        histories = self._refresh_histories(merged_quotes)
        feature_rows = build_feature_rows(
            merged_quotes,
            histories=histories,
            stale_quote_seconds=self.stale_quote_seconds,
            fee_bps=self.fee_bps,
            queue_penalty_bps=self.queue_penalty_bps,
        )
        tracked_features = self._trade_features(feature_rows)
        self.quote_map = {str(item.get("token_id") or ""): dict(item) for item in tracked_features}
        tracked = self._track_examples(tracked_features)
        settled = self.label_store.update_labels(self.quote_map)
        self.model_league.settle_labels(settled.get("settled_labels") or [], primary_horizon=self.primary_horizon)
        self._manage_positions()
        candidate = self._select_execution_candidate(tracked_features)
        if candidate is not None:
            self._open_trade(candidate, self.quote_map)
        current_balance = self.executor.current_balance(self.quote_map)
        if not self.balance_history or abs(to_float(self.balance_history[-1].get("balance"), 0.0) - current_balance) > 1e-9:
            self.balance_history.append({"ts": now_iso, "balance": round(current_balance, 6)})
            self.balance_history = self.balance_history[-1000:]
        labels = self.label_store.load_labels(limit=5000)
        primary_labels = [row for row in labels if str(row.get("horizon_label")) == f"{self.primary_horizon}s"]
        final_labels = [row for row in labels if str(row.get("horizon_label")) == "final"]
        shadow_accounts = self.model_league.build_accounts()
        hybrid_metrics = next(
            (account.metrics for account in shadow_accounts if account.model_id == "hybrid_transition"),
            {},
        )
        source_health = {
            "gamma": {
                "healthy": bool(gamma_snapshot.get("healthy")),
                "event_count": int(gamma_snapshot.get("event_count", 0) or 0),
                "market_count": int(gamma_snapshot.get("market_count", 0) or 0),
                "token_count": int(gamma_snapshot.get("token_count", 0) or 0),
                "last_refresh_ts": gamma_snapshot.get("observed_at"),
            },
            "clob": {
                "healthy": bool(rest_books),
                "rest_book_count": len(rest_books),
                "ws_connected": bool(self.stream.snapshot().get("connected")),
                "ws_last_message_ts": self.stream.snapshot().get("last_message_ts"),
                "event_counts": dict(self.stream.snapshot().get("event_counts") or {}),
                "last_error": self.stream.snapshot().get("last_error"),
            },
        }
        training_progress = {
            "tracked_examples": self._tracked_examples,
            "labeled_examples": len(primary_labels),
            "pending_labels": int(settled.get("pending_total", 0) or 0),
            "final_resolution_labels": len(final_labels),
            "closed_trades": len(self.executor.closed_trades),
            "targets": {
                "labeled_examples": int(getattr(config, "POLYMARKET_QF_READINESS_MIN_LABELED", 250)),
                "closed_trades": int(getattr(config, "POLYMARKET_QF_READINESS_MIN_CLOSED_TRADES", 50)),
            },
            "primary_horizon_seconds": self.primary_horizon,
        }
        calibration_lift = to_float(hybrid_metrics.get("final_brier_lift"), 0.0)
        readiness_checks = [
            {
                "name": "gamma_feed_healthy",
                "ok": source_health["gamma"]["healthy"],
                "reason": "Gamma sports discovery must be healthy",
            },
            {
                "name": "clob_books_healthy",
                "ok": source_health["clob"]["healthy"],
                "reason": "CLOB books must be available for tracked tokens",
            },
            {
                "name": "labeled_examples_minimum",
                "ok": training_progress["labeled_examples"] >= training_progress["targets"]["labeled_examples"],
                "reason": "Need enough labeled examples for stable model comparison",
            },
            {
                "name": "closed_trades_minimum",
                "ok": training_progress["closed_trades"] >= training_progress["targets"]["closed_trades"],
                "reason": "Need enough closed paper trades to judge execution behavior",
            },
            {
                "name": "shadow_pnl_positive",
                "ok": self.executor.realized_pnl() > 0,
                "reason": "Paper portfolio must be positive after modeled costs",
            },
            {
                "name": "calibration_lift_positive",
                "ok": calibration_lift > 0,
                "reason": "Hybrid model should beat market baseline on final resolution calibration",
            },
        ]
        blockers = [item["name"] for item in readiness_checks if not item["ok"]]
        readiness_status = "candidate" if not blockers else "paper_validating"
        avg_freshness = summarize_feature_rows(tracked_features).get("avg_quote_freshness_sec")
        raw_state = {
            "portfolio_id": "polymarket_quantum_fold",
            "running": self._running,
            "mode": "paper",
            "status": "running" if self._running else "idle",
            "explainer": "Sports-only Polymarket paper book that combines coherence, interference, and folding-confidence features, learns online from live Gamma+CLOB observations, and executes only in a standalone paper account.",
            "scan_count": self._scan_count,
            "open_positions": list(self.executor.open_positions),
            "closed_trades": list(self.executor.closed_trades),
            "trade_count": len(self.executor.closed_trades),
            "realized_pnl_usd": self.executor.realized_pnl(),
            "current_balance_usd": current_balance,
            "gross_exposure_usd": self.executor.gross_exposure(),
            "balance_history": list(self.balance_history),
            "events": self.events[-200:],
            "execution_quality": self.executor.execution_quality(),
            "risk": {
                "drawdown_pct": self.executor.drawdown_pct(self.quote_map),
                "drawdown_halt_active": self.executor.drawdown_halt_active,
                "stale_quote_halts": self.executor.stale_halt_count,
                "open_positions": len(self.executor.open_positions),
            },
            "source_health": source_health,
            "quote_freshness_sec": avg_freshness,
            "market_count": int(gamma_snapshot.get("market_count", 0) or 0),
            "token_count": len(tracked_features),
            "training_progress": training_progress,
            "research_summary": {
                "labeled_examples": len(primary_labels),
                "pending_labels": int(settled.get("pending_total", 0) or 0),
                "avg_realized_edge": round(
                    sum(to_float(item.get("net_return"), 0.0) for item in primary_labels[-200:]) / max(1, len(primary_labels[-200:])),
                    6,
                ) if primary_labels else 0.0,
                "win_rate": round(
                    sum(1 for item in primary_labels[-200:] if to_float(item.get("net_return"), 0.0) > 0.0) / max(1, len(primary_labels[-200:])),
                    6,
                ) if primary_labels else 0.0,
            },
            "model_league": self.model_league.summary(),
            "rejections": dict(self._rejections),
            "readiness": {
                "status": readiness_status,
                "blockers": blockers if self._running else ["engine_not_running"],
                "checks": readiness_checks,
                "score_pct": round((sum(1 for item in readiness_checks if item["ok"]) / len(readiness_checks)) * 100.0, 2),
            },
        }
        self._append_snapshot(
            {
                "ts": now_iso,
                "gamma_event_count": gamma_snapshot.get("event_count"),
                "gamma_market_count": gamma_snapshot.get("market_count"),
                "tracked_token_count": len(tracked_features),
                "primary_labels": len(primary_labels),
                "pending_labels": settled.get("pending_total"),
                "open_positions": len(self.executor.open_positions),
                "realized_pnl_usd": self.executor.realized_pnl(),
            }
        )
        self.events = self.events[-200:]
        self._last_state = raw_state
        return raw_state

    def get_state(self) -> Dict[str, Any]:
        return dict(self._last_state or {})
