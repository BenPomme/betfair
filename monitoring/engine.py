"""
Trading engine runnable from the dashboard: start/stop in a background thread,
expose state (P&L, trades, events, scan stats, market breakdown, risk) for the UI.
"""
import asyncio
import importlib
import logging
import os
import sys
import threading
import time
from collections import deque
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import config

logger = logging.getLogger(__name__)

# Avoid circular import by lazy-importing main
_main_module = None

def _get_main():
    global _main_module
    if _main_module is None:
        import main as m
        _main_module = m
    return _main_module


# Modules to reload on each Start so code changes take effect without
# restarting the whole process.
_RELOAD_MODULES = [
    "core.commission",
    "core.cross_market_scanner",
    "core.scanner",
    "core.risk_manager",
    "core.stake_calculator",
    "execution.paper_executor",
    "execution.executor",
    "execution.live_executor",
    "data.event_grouper",
    "strategy.predictive_model",
    "strategy.prediction_engine",
    "strategy.features",
    "strategy.model_inference",
    "main",
]


def _reload_trading_modules() -> None:
    """Reload core/execution modules so code changes take effect on Start."""
    global _main_module
    for name in _RELOAD_MODULES:
        if name in sys.modules:
            try:
                importlib.reload(sys.modules[name])
            except Exception as e:
                logger.warning("Failed to reload %s: %s", name, e)
    # Re-acquire main after reload
    if "main" in sys.modules:
        _main_module = sys.modules["main"]


class TradingEngine:
    """Runs the trading loop in a background thread; exposes state for the dashboard."""

    def __init__(self, max_events: int = 300):
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._active_tasks: List[asyncio.Task] = []
        self._events: deque = deque(maxlen=max_events)
        self._market_ids: List[str] = []
        self._market_metadata: Dict[str, Dict[str, str]] = {}
        self._price_cache: Any = None
        self._risk_manager: Any = None
        self._paper_executor: Any = None
        self._clv_tracker: Any = None
        self._error: Optional[str] = None
        self._lock = threading.Lock()
        self._restart_requested: bool = False
        self._restart_reason: str = ""
        self._last_auto_restart_ts: Optional[float] = None
        self._manual_stop_requested: bool = False

        # Scan stats
        self._total_scans: int = 0
        self._opportunities_found: int = 0
        self._back_back_found: int = 0
        self._lay_lay_found: int = 0
        self._cross_market_found: int = 0
        self._best_overround: Optional[float] = None
        self._session_start_time: Optional[float] = None
        self._candidates_found: int = 0
        self._scored_count: int = 0
        self._deferred_count: int = 0
        self._executed_count: int = 0
        self._fill_prob_sum: float = 0.0
        self._expected_net_sum: float = 0.0
        self._scan_with_snapshot: int = 0
        self._scan_without_snapshot: int = 0
        self._decision_execute_count: int = 0
        self._decision_defer_count: int = 0
        self._decision_skip_count: int = 0
        self._cross_type_counts: Dict[str, int] = {}
        self._prediction_models: Dict[str, Any] = {}
        self._last_scan_ts: Optional[float] = None
        self._last_trade_ts: Optional[float] = None
        self._last_prediction_ts: Optional[float] = None
        self._last_architect_ts: Optional[float] = None
        self._poller_metrics: Dict[str, Any] = {}
        self._architect_state: Dict[str, Any] = {
            "enabled": config.ARCHITECT_ENABLED,
            "last_run_ts": None,
            "mode": "n/a",
            "applied": False,
            "reason": "",
            "proposals": [],
        }
        self._qa_state: Dict[str, Any] = {
            "enabled": config.QA_AGENT_ENABLED,
            "last_run_ts": None,
            "mode": "n/a",
            "applied": False,
            "reason": "",
            "actions": [],
            "results": [],
            "metrics": {},
        }
        self._qa_agent: Any = None

        # Balance history: (timestamp_iso, balance_float) pairs
        self._balance_history: deque = deque(maxlen=500)

    def record_event(self, kind: str, data: Dict[str, Any]) -> None:
        with self._lock:
            self._events.append({"kind": kind, "data": data})

    def on_scan(self, market_id: str, has_snapshot: Optional[bool] = None) -> None:
        """Called after each market scan."""
        with self._lock:
            self._total_scans += 1
            self._last_scan_ts = time.time()
            if has_snapshot is True:
                self._scan_with_snapshot += 1
            elif has_snapshot is False:
                self._scan_without_snapshot += 1
        payload = {"market_id": market_id}
        if has_snapshot is not None:
            payload["has_snapshot"] = has_snapshot
        self.record_event("scan", payload)

    def on_poller_metrics(self, payload: Dict[str, Any]) -> None:
        """Called by price poller each cycle."""
        if not payload:
            return
        with self._lock:
            self._poller_metrics = dict(payload)

    def on_opportunity(self, opp: Any, scored: Optional[Any] = None) -> None:
        """Called when an opportunity is detected."""
        with self._lock:
            self._opportunities_found += 1
            self._candidates_found += 1
            overround = float(getattr(opp, "overround_raw", 0))
            if self._best_overround is None or overround < self._best_overround:
                self._best_overround = overround
            arb_type = getattr(opp, "arb_type", "back_back")
            if str(arb_type).startswith("cross"):
                self._cross_market_found += 1
                self._cross_type_counts[arb_type] = self._cross_type_counts.get(arb_type, 0) + 1
            elif arb_type == "lay_lay":
                self._lay_lay_found += 1
            else:
                self._back_back_found += 1
            if scored is not None:
                self._scored_count += 1
                self._fill_prob_sum += float(getattr(scored, "fill_prob", 0.0))
                self._expected_net_sum += float(getattr(scored, "expected_net_profit_eur", 0.0))
                if getattr(scored, "decision", "EXECUTE") != "EXECUTE":
                    self._deferred_count += 1
                decision = str(getattr(scored, "decision", "EXECUTE"))
                if decision == "EXECUTE":
                    self._decision_execute_count += 1
                elif decision == "DEFER":
                    self._decision_defer_count += 1
                elif decision == "SKIP":
                    self._decision_skip_count += 1
        self.record_event("opportunity", {
            "market_id": opp.market_id,
            "event_name": opp.event_name,
            "net_profit_eur": float(opp.net_profit_eur),
            "overround": float(opp.overround_raw),
            "arb_type": getattr(opp, "arb_type", "back_back"),
            "decision": getattr(scored, "decision", "EXECUTE") if scored else "EXECUTE",
            "fill_prob": float(getattr(scored, "fill_prob", 0.0)) if scored else None,
            "expected_net_profit_eur": float(getattr(scored, "expected_net_profit_eur", 0.0)) if scored else None,
            "stake_multiplier": float(getattr(scored, "stake_multiplier", 1.0)) if scored else None,
            "selections": [
                {"name": s["name"], "back_price": s.get("back_price"), "lay_price": s.get("lay_price"), "stake_eur": s["stake_eur"]}
                for s in opp.selections
            ],
        })

    def on_trade(self, opp: Any, result: dict, scored: Optional[Any] = None) -> None:
        """Called when a trade is executed."""
        with self._lock:
            self._executed_count += 1
            self._last_trade_ts = time.time()
            if self._paper_executor is not None:
                self._balance_history.append((
                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    float(self._paper_executor.balance),
                ))
        self.record_event("trade", {
            "market_id": opp.market_id,
            "event_name": opp.event_name,
            "net_profit_eur": float(opp.net_profit_eur),
            "arb_type": getattr(opp, "arb_type", "back_back"),
            "selections": [
                {"name": s["name"], "back_price": s.get("back_price"), "lay_price": s.get("lay_price"), "stake_eur": s["stake_eur"]}
                for s in opp.selections
            ],
        })

    def on_prediction(self, payload: Dict[str, Any]) -> None:
        """Called from prediction paper-account engine."""
        if not payload:
            return
        models = payload.get("models")
        events = payload.get("events") or []
        with self._lock:
            if isinstance(models, dict):
                self._prediction_models = models
            self._last_prediction_ts = time.time()
        for ev in events:
            kind = ev.get("kind")
            if kind in {"prediction_open", "prediction_settle"}:
                self.record_event(kind, ev)

    def on_architect(self, payload: Dict[str, Any]) -> None:
        """Called when learning architect runs a decision cycle."""
        if not payload:
            return
        with self._lock:
            self._last_architect_ts = time.time()
            self._architect_state = {
                "enabled": config.ARCHITECT_ENABLED,
                "last_run_ts": payload.get("ts"),
                "mode": payload.get("mode", "rules"),
                "applied": bool(payload.get("applied", False)),
                "reason": payload.get("reason", ""),
                "proposals": payload.get("proposals", []),
            }
        self.record_event("architect", self._architect_state)

    def on_qa(self, payload: Dict[str, Any]) -> None:
        """Called when live QA agent runs a decision cycle."""
        if not payload:
            return
        with self._lock:
            self._qa_state = {
                "enabled": config.QA_AGENT_ENABLED,
                "last_run_ts": payload.get("ts"),
                "mode": payload.get("mode", "rules"),
                "applied": bool(payload.get("applied", False)),
                "reason": payload.get("reason", ""),
                "actions": payload.get("actions", []),
                "results": payload.get("results", []),
                "metrics": payload.get("metrics", {}),
            }
        self.record_event("qa", self._qa_state)

    def _qa_runtime_state(self) -> Dict[str, Any]:
        state = self.get_state()
        return {
            "running": bool(state.get("running", False)),
            "health": dict(state.get("health", {})),
        }

    def _request_runtime_restart(self, reason: str) -> bool:
        if not config.QA_RESTART_ON_DEGRADED_ENABLED:
            return False
        now = time.time()
        cooldown = max(30, int(config.QA_RESTART_COOLDOWN_SECONDS))
        queued = False
        with self._lock:
            if self._manual_stop_requested:
                return False
            if self._restart_requested:
                return False
            if (
                self._last_auto_restart_ts is not None
                and (now - self._last_auto_restart_ts) < cooldown
            ):
                return False
            self._restart_requested = True
            self._restart_reason = reason
            queued = True
        if queued:
            self.record_event("system_restart_requested", {"reason": reason})
            main = _get_main()
            main._running = False
        return queued

    def start(self) -> Dict[str, Any]:
        """Start the trading session in a background thread. Returns status dict."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return {"ok": False, "error": "Already running"}
            self._manual_stop_requested = False
            self._restart_requested = False
            self._restart_reason = ""
            self._error = None
            self._market_ids = []
            self._market_metadata = {}
            self._paper_executor = None
            self._clv_tracker = None
            self._risk_manager = None
            self._price_cache = None
            self._total_scans = 0
            self._opportunities_found = 0
            self._back_back_found = 0
            self._lay_lay_found = 0
            self._cross_market_found = 0
            self._best_overround = None
            self._candidates_found = 0
            self._scored_count = 0
            self._deferred_count = 0
            self._executed_count = 0
            self._fill_prob_sum = 0.0
            self._expected_net_sum = 0.0
            self._scan_with_snapshot = 0
            self._scan_without_snapshot = 0
            self._decision_execute_count = 0
            self._decision_defer_count = 0
            self._decision_skip_count = 0
            self._cross_type_counts = {}
            self._prediction_models = {}
            self._last_scan_ts = None
            self._last_trade_ts = None
            self._last_prediction_ts = None
            self._last_architect_ts = None
            self._poller_metrics = {}
            self._architect_state = {
                "enabled": config.ARCHITECT_ENABLED,
                "last_run_ts": None,
                "mode": "n/a",
                "applied": False,
                "reason": "",
                "proposals": [],
            }
            self._qa_state = {
                "enabled": config.QA_AGENT_ENABLED,
                "last_run_ts": None,
                "mode": "n/a",
                "applied": False,
                "reason": "",
                "actions": [],
                "results": [],
                "metrics": {},
            }
            self._qa_agent = None
            self._session_start_time = time.time()
            self._balance_history.clear()
            self._loop = None
            self._active_tasks = []

        self._thread = threading.Thread(target=self._run_session, daemon=True)
        self._thread.start()
        return {"ok": True}

    def stop(self) -> Dict[str, Any]:
        """Signal the session to stop (poller and loop will exit)."""
        main = _get_main()
        with self._lock:
            self._manual_stop_requested = True
            self._restart_requested = False
            self._restart_reason = ""
        main._running = False
        with self._lock:
            loop = self._loop
            tasks = list(self._active_tasks)
            thread = self._thread
        if loop is not None and loop.is_running():
            def _cancel_tasks() -> None:
                for task in tasks:
                    if not task.done():
                        task.cancel()
            try:
                loop.call_soon_threadsafe(_cancel_tasks)
            except Exception:
                pass
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        return {"ok": True}

    def get_state(self) -> Dict[str, Any]:
        """Current state for the UI: running, balance, P&L, trades, events, markets, scan stats, risk, etc."""
        with self._lock:
            running = self._thread is not None and self._thread.is_alive()
            balance = None
            daily_pnl = None
            log_entries: List[dict] = []
            market_ids = list(self._market_ids)
            market_metadata = dict(self._market_metadata)

            if self._paper_executor is not None:
                balance = float(self._paper_executor.balance)
                log_entries = self._paper_executor.log_entries[-50:]
                effective_stake = float(
                    min(
                        self._paper_executor.balance * config.STAKE_FRACTION,
                        config.MAX_STAKE_EUR,
                    )
                )
            else:
                effective_stake = None
            if self._risk_manager is not None:
                daily_pnl = float(self._risk_manager._daily_pnl_eur)

            events = list(self._events)

            # Scan stats
            total_scans = self._total_scans
            opportunities_found = self._opportunities_found
            best_overround = self._best_overround
            candidates_found = self._candidates_found
            scored_count = self._scored_count
            deferred_count = self._deferred_count
            executed_count = self._executed_count
            session_start = self._session_start_time

            # Compute scan rate
            uptime_seconds = 0.0
            scan_rate_per_min = 0.0
            if session_start is not None:
                uptime_seconds = time.time() - session_start
                if uptime_seconds > 0:
                    scan_rate_per_min = (total_scans / uptime_seconds) * 60.0

            # Trade stats
            trade_count = len(log_entries)
            hit_rate = 0.0
            avg_profit = 0.0
            if total_scans > 0:
                hit_rate = (trade_count / total_scans) * 100.0
            if trade_count > 0:
                avg_profit = sum(t.get("net_profit_eur", 0) for t in log_entries) / trade_count
            avg_fill_prob = (self._fill_prob_sum / scored_count) if scored_count > 0 else 0.0
            avg_expected_net = (self._expected_net_sum / scored_count) if scored_count > 0 else 0.0
            freshness_hit_rate = (self._scan_with_snapshot / total_scans) if total_scans > 0 else 0.0

            # Market breakdown
            by_sport: Dict[str, int] = {}  # actually by market_type now
            by_country: Dict[str, int] = {}
            for mid, meta in market_metadata.items():
                market_type = meta.get("market_type") or meta.get("sport_name", "Unknown")
                country = meta.get("country", "") or "Global"
                by_sport[market_type] = by_sport.get(market_type, 0) + 1
                by_country[country] = by_country.get(country, 0) + 1

            # Risk state
            from execution.executor import trading_halted, _consecutive_failures, CIRCUIT_BREAKER_THRESHOLD
            open_bets = 0
            if self._paper_executor is not None:
                open_bets = getattr(self._paper_executor, "_open_bets", 0)

            # Balance history
            balance_history = list(self._balance_history)
            prediction_models = dict(self._prediction_models)
            architect = dict(self._architect_state)
            qa = dict(self._qa_state)
            prediction_leader = None
            if prediction_models:
                prediction_leader = max(
                    prediction_models.values(),
                    key=lambda m: float(m.get("balance_eur", 0.0)),
                )
            clv_summary = self._clv_tracker.get_summary() if self._clv_tracker is not None else {
                "avg_clv": 0.0,
                "positive_clv_pct": 0.0,
                "total_tracked": 0,
            }
            cross_type_counts = dict(self._cross_type_counts)

            conversion_scan_to_candidate = (candidates_found / total_scans) if total_scans > 0 else 0.0
            conversion_candidate_to_scored = (scored_count / candidates_found) if candidates_found > 0 else 0.0
            conversion_scored_to_execute = (executed_count / scored_count) if scored_count > 0 else 0.0

            prediction_summary = {
                "model_count": 0,
                "active_model_count": 0,
                "settled_total": 0,
                "open_positions_total": 0,
                "resets_total": 0,
                "wins_total": 0,
                "weighted_win_rate_pct": 0.0,
                "avg_brier": 0.0,
                "avg_roi_pct": 0.0,
                "learning_settled_total": 0,
                "learning_updates_total": 0,
                "learning_open_markets_total": 0,
                "calibrated_models": 0,
                "leader_model_id": None,
                "leader_roi_pct": 0.0,
                "leader_balance_eur": 0.0,
                "clv": clv_summary,
            }
            if prediction_models:
                models = list(prediction_models.values())
                prediction_summary["model_count"] = len(models)
                prediction_summary["settled_total"] = int(sum(float(m.get("settled_bets", 0)) for m in models))
                prediction_summary["open_positions_total"] = int(sum(float(m.get("open_positions", 0)) for m in models))
                prediction_summary["resets_total"] = int(sum(float(m.get("resets", 0)) for m in models))
                prediction_summary["wins_total"] = int(sum(float(m.get("wins", 0)) for m in models))
                prediction_summary["active_model_count"] = int(
                    sum(1 for m in models if int(m.get("settled_bets", 0)) > 0)
                )
                if prediction_summary["settled_total"] > 0:
                    prediction_summary["weighted_win_rate_pct"] = round(
                        (prediction_summary["wins_total"] / prediction_summary["settled_total"]) * 100.0,
                        2,
                    )
                prediction_summary["learning_settled_total"] = int(
                    sum(float(m.get("learning_settled", 0)) for m in models)
                )
                prediction_summary["learning_updates_total"] = int(
                    sum(float(m.get("learning_updates", 0)) for m in models)
                )
                prediction_summary["learning_open_markets_total"] = int(
                    sum(float(m.get("learning_open_markets", 0)) for m in models)
                )
                prediction_summary["avg_brier"] = round(
                    sum(float(m.get("avg_brier", 0.0)) for m in models) / len(models), 6
                )
                prediction_summary["avg_roi_pct"] = round(
                    sum(float(m.get("roi_pct", 0.0)) for m in models) / len(models), 4
                )
                prediction_summary["calibrated_models"] = int(sum(
                    1
                    for m in models
                    if int(m.get("settled_bets", 0)) >= 30 and float(m.get("avg_brier", 1.0)) <= 0.28
                ))
                if prediction_leader:
                    prediction_summary["leader_model_id"] = prediction_leader.get("model_id")
                    prediction_summary["leader_roi_pct"] = float(prediction_leader.get("roi_pct", 0.0))
                    prediction_summary["leader_balance_eur"] = float(prediction_leader.get("balance_eur", 0.0))

            now = time.time()
            last_scan_age_sec = (now - self._last_scan_ts) if self._last_scan_ts else None
            last_pred_age_sec = (now - self._last_prediction_ts) if self._last_prediction_ts else None
            last_arch_age_sec = (now - self._last_architect_ts) if self._last_architect_ts else None
            restart_pending = bool(self._restart_requested)
            last_restart_ts = self._last_auto_restart_ts
            restart_cooldown = max(30, int(config.QA_RESTART_COOLDOWN_SECONDS))
            restart_cooldown_remaining_sec = 0
            if last_restart_ts is not None:
                restart_cooldown_remaining_sec = max(0, int(restart_cooldown - (now - last_restart_ts)))

            feed_ok = bool(running and last_scan_age_sec is not None and last_scan_age_sec <= 12.0)
            prediction_ok = bool(
                (not config.PREDICTION_ENABLED)
                or (last_pred_age_sec is not None and last_pred_age_sec <= 30.0)
            )
            architect_ok = bool(
                (not config.ARCHITECT_ENABLED)
                or (last_arch_age_sec is None)
                or (last_arch_age_sec <= (config.ARCHITECT_INTERVAL_SECONDS * 3))
            )
            risk_ok = bool(not trading_halted)
            system_ok = bool(feed_ok and prediction_ok and architect_ok and risk_ok and running)

            def _artifact_info(path_str: str) -> Dict[str, Any]:
                if not path_str:
                    return {"path": "", "exists": False}
                p = Path(path_str)
                if not p.exists() or not p.is_file():
                    return {"path": str(p), "exists": False}
                st = p.stat()
                return {
                    "path": str(p),
                    "exists": True,
                    "size_bytes": int(st.st_size),
                    "modified_ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime)),
                }

            artifacts = {
                "scoring_model": _artifact_info(config.ML_LINEAR_MODEL_PATH),
                "fill_model": _artifact_info(config.FILL_MODEL_PATH),
            }

            gates_path = Path("data/reports/performance_gates/latest.json")
            gates = None
            if gates_path.exists():
                try:
                    import json
                    gates = json.loads(gates_path.read_text(encoding="utf-8"))
                except Exception:
                    gates = {"status": "error", "reason": "failed_to_parse_gate_report"}
            poller_metrics = dict(self._poller_metrics)

        return {
            "running": running,
            "balance_eur": balance,
            "daily_pnl_eur": daily_pnl,
            "trades": log_entries,
            "events": events,
            "market_ids": market_ids,
            "market_count": len(market_ids),
            "error": self._error,
            "scan_stats": {
                "total_scans": total_scans,
                "opportunities_found": opportunities_found,
                "candidates_found": candidates_found,
                "scored_count": scored_count,
                "deferred_count": deferred_count,
                "executed_count": executed_count,
                "back_back_found": self._back_back_found,
                "lay_lay_found": self._lay_lay_found,
                "cross_market_found": self._cross_market_found,
                "scan_rate_per_min": round(scan_rate_per_min, 1),
                "best_overround": round(best_overround, 6) if best_overround is not None else None,
                "avg_fill_prob": round(avg_fill_prob, 4),
                "avg_expected_net": round(avg_expected_net, 4),
                "scan_with_snapshot": self._scan_with_snapshot,
                "scan_without_snapshot": self._scan_without_snapshot,
                "freshness_hit_rate": round(freshness_hit_rate, 6),
                "decision_execute_count": self._decision_execute_count,
                "decision_defer_count": self._decision_defer_count,
                "decision_skip_count": self._decision_skip_count,
                "funnel_scan_to_candidate": round(conversion_scan_to_candidate, 6),
                "funnel_candidate_to_scored": round(conversion_candidate_to_scored, 6),
                "funnel_scored_to_execute": round(conversion_scored_to_execute, 6),
                "cross_type_counts": cross_type_counts,
            },
            "market_breakdown": {
                "by_sport": by_sport,
                "by_country": by_country,
            },
            "risk": {
                "daily_loss_limit": float(config.DAILY_LOSS_LIMIT_EUR),
                "daily_loss_used": abs(daily_pnl) if daily_pnl is not None and daily_pnl < 0 else 0.0,
                "max_stake": float(config.MAX_STAKE_EUR),
                "open_bets": open_bets,
                "circuit_breaker": trading_halted,
                "consecutive_failures": _consecutive_failures,
                "circuit_breaker_threshold": CIRCUIT_BREAKER_THRESHOLD,
            },
            "health": {
                "mode": "LIVE" if not config.PAPER_TRADING else "PAPER",
                "system_ok": system_ok,
                "feed_ok": feed_ok,
                "prediction_ok": prediction_ok,
                "architect_ok": architect_ok,
                "risk_ok": risk_ok,
                "auto_restart_enabled": bool(config.QA_RESTART_ON_DEGRADED_ENABLED),
                "auto_restart_pending": restart_pending,
                "auto_restart_cooldown_remaining_sec": restart_cooldown_remaining_sec,
                "last_auto_restart_ts": (
                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(last_restart_ts))
                    if last_restart_ts is not None
                    else None
                ),
                "telegram_configured": bool(config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID),
                "last_scan_age_sec": round(last_scan_age_sec, 1) if last_scan_age_sec is not None else None,
                "last_prediction_age_sec": round(last_pred_age_sec, 1) if last_pred_age_sec is not None else None,
                "last_architect_age_sec": round(last_arch_age_sec, 1) if last_arch_age_sec is not None else None,
            },
            "poller": poller_metrics,
            "session": {
                "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(session_start)) if session_start else None,
                "uptime_seconds": round(uptime_seconds, 0),
                "hit_rate": round(hit_rate, 4),
                "avg_profit": round(avg_profit, 4),
                "trade_count": trade_count,
            },
            "config": {
                "paper_trading": config.PAPER_TRADING,
                "mbr": float(config.MBR),
                "discount_rate": float(config.DISCOUNT_RATE),
                "min_profit": float(config.MIN_NET_PROFIT_EUR),
                "min_liquidity": float(config.MIN_LIQUIDITY_EUR),
                "max_stake": float(config.MAX_STAKE_EUR),
                "daily_loss_limit": float(config.DAILY_LOSS_LIMIT_EUR),
                "sports": config.SCAN_SPORTS,
                "countries": config.SCAN_COUNTRIES,
                "max_markets": config.SCAN_MAX_MARKETS,
                "include_in_play": config.SCAN_INCLUDE_IN_PLAY,
                "scan_min_inplay_markets": config.SCAN_MIN_INPLAY_MARKETS,
                "initial_balance": float(config.INITIAL_BALANCE_EUR),
                "stake_fraction": float(config.STAKE_FRACTION),
                "scan_interval_seconds": config.SCAN_INTERVAL_SECONDS,
                "market_refresh_interval_seconds": config.MARKET_REFRESH_INTERVAL_SECONDS,
                "market_no_movement_seconds": config.MARKET_NO_MOVEMENT_SECONDS,
                "market_missing_retire_cycles": config.MARKET_MISSING_RETIRE_CYCLES,
                "poll_interval_seconds": config.PRICE_POLL_INTERVAL_SECONDS,
                "cross_market_enabled": config.CROSS_MARKET_ENABLED,
                "cross_market_mo_dnb_enabled": config.CROSS_MARKET_MO_DNB_ENABLED,
                "cross_market_mo_ou25_enabled": config.CROSS_MARKET_MO_OU25_ENABLED,
                "cross_market_mo_btts_enabled": config.CROSS_MARKET_MO_BTTS_ENABLED,
                "cross_market_cs_mo_enabled": config.CROSS_MARKET_CS_MO_ENABLED,
                "ml_scoring_enabled": config.ML_SCORING_ENABLED,
                "ml_stake_sizing_enabled": bool(getattr(config, "ML_STAKE_SIZING_ENABLED", True)),
                "ml_stake_min_multiplier": float(getattr(config, "ML_STAKE_MIN_MULTIPLIER", 0.35)),
                "ml_stake_max_multiplier": float(getattr(config, "ML_STAKE_MAX_MULTIPLIER", 1.25)),
                "ml_stake_min_eur": float(getattr(config, "ML_STAKE_MIN_EUR", 2.00)),
                "candidate_log_enabled": config.CANDIDATE_LOG_ENABLED,
                "prediction_enabled": config.PREDICTION_ENABLED,
                "prediction_initial_balance": float(config.PREDICTION_INITIAL_BALANCE_EUR),
                "prediction_stake_fraction": float(config.PREDICTION_STAKE_FRACTION),
                "prediction_min_edge": float(config.PREDICTION_MIN_EDGE),
                "prediction_model_kinds": config.PREDICTION_MODEL_KINDS,
                "clv_enabled": config.CLV_ENABLED,
                "architect_enabled": config.ARCHITECT_ENABLED,
                "architect_interval_seconds": config.ARCHITECT_INTERVAL_SECONDS,
                "qa_restart_on_degraded_enabled": config.QA_RESTART_ON_DEGRADED_ENABLED,
                "qa_degraded_min_age_seconds": config.QA_DEGRADED_MIN_AGE_SECONDS,
                "qa_restart_cooldown_seconds": config.QA_RESTART_COOLDOWN_SECONDS,
            },
            "prediction_models": prediction_models,
            "prediction_leader": prediction_leader,
            "prediction_summary": prediction_summary,
            "artifacts": artifacts,
            "gates": gates,
            "architect": architect,
            "qa": qa,
            "balance_history": balance_history,
            "effective_stake": round(effective_stake, 2) if effective_stake is not None else None,
        }

    def _run_session(self) -> None:
        _reload_trading_modules()
        main = _get_main()
        main._running = True
        self._error = None
        client = None
        try:
            from data.betfair_client import create_and_login
            from data.candidate_logger import CandidateLogger
            from data.clv_tracker import CLVTracker
            from data.price_cache import PriceCache
            from data.price_poller import run_price_poller
            from core.risk_manager import RiskManager
            from execution.order_monitor import OrderMonitor
            from execution.paper_executor import PaperExecutor
            from strategy.prediction_engine import MultiModelPredictionManager
            from strategy.learning_architect import LearningArchitect
            from qa.live_qa_agent import LiveQAAgent

            client = create_and_login()
            from data.market_catalogue import discover_markets

            market_ids_str = os.getenv("MARKET_IDS", "")
            market_ids = [m.strip() for m in market_ids_str.split(",") if m.strip()]
            market_metadata: Dict[str, Dict[str, str]] = {}
            runner_names: Dict[str, Dict[str, str]] = {}

            if not market_ids:
                try:
                    market_ids, market_metadata, runner_names = discover_markets(
                        client,
                        max_total=config.SCAN_MAX_MARKETS,
                        include_in_play=config.SCAN_INCLUDE_IN_PLAY,
                    )
                except Exception as e:
                    logger.warning("Market discovery failed: %s", e)
                    with self._lock:
                        self._error = "Market discovery failed: " + str(e)
                    return

            if not market_ids:
                with self._lock:
                    self._error = "No markets to watch. Set MARKET_IDS in .env or try again later."
                return

            with self._lock:
                self._market_ids = market_ids
                self._market_metadata = market_metadata
                # Record initial balance
                self._balance_history.append((
                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    float(config.INITIAL_BALANCE_EUR),
                ))

            price_cache = PriceCache(max_age_seconds=config.STALE_PRICE_SECONDS)
            risk_manager = RiskManager(
                max_stake_eur=config.MAX_STAKE_EUR,
                daily_loss_limit_eur=config.DAILY_LOSS_LIMIT_EUR,
            )
            paper_executor = PaperExecutor(
                initial_balance_eur=config.INITIAL_BALANCE_EUR,
                state_path=config.PAPER_STATE_PATH,
                trades_log_path=config.PAPER_TRADES_LOG_PATH,
            )
            candidate_logger = CandidateLogger(config.CANDIDATE_LOG_DIR) if config.CANDIDATE_LOG_ENABLED else None
            prediction_manager = None
            learning_architect = LearningArchitect() if config.ARCHITECT_ENABLED else None
            qa_agent = LiveQAAgent() if config.QA_AGENT_ENABLED else None
            clv_tracker = CLVTracker(config.CLV_LOG_DIR) if config.CLV_ENABLED else None
            with self._lock:
                self._clv_tracker = clv_tracker
            if config.PREDICTION_ENABLED:
                model_kinds = [x.strip() for x in config.PREDICTION_MODEL_KINDS.split(",") if x.strip()]
                prediction_manager = MultiModelPredictionManager(
                    model_kinds=model_kinds,
                    initial_balance_eur=float(config.PREDICTION_INITIAL_BALANCE_EUR),
                    stake_fraction=float(config.PREDICTION_STAKE_FRACTION),
                    min_stake_eur=float(config.PREDICTION_MIN_STAKE_EUR),
                    max_stake_eur=float(config.PREDICTION_MAX_STAKE_EUR),
                    min_edge=float(config.PREDICTION_MIN_EDGE),
                    min_liquidity_eur=float(config.PREDICTION_MIN_LIQUIDITY_EUR),
                    model_dir=config.PREDICTION_MODEL_DIR,
                    state_dir=config.PREDICTION_STATE_DIR,
                    save_every=config.PREDICTION_SAVE_EVERY,
                    clv_tracker=clv_tracker,
                )
                with self._lock:
                    self._prediction_models = prediction_manager.initial_state()
            risk_manager.set_daily_pnl(paper_executor.daily_pnl)
            risk_manager.set_open_bets(paper_executor.open_bets)
            with self._lock:
                self._price_cache = price_cache
                self._risk_manager = risk_manager
                self._paper_executor = paper_executor
                self._qa_agent = qa_agent

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            with self._lock:
                self._loop = loop

            async def _async() -> None:
                order_monitor = OrderMonitor(client=client)
                poller_task = loop.create_task(
                    run_price_poller(
                        client,
                        market_ids,
                        price_cache,
                        interval_seconds=config.PRICE_POLL_INTERVAL_SECONDS,
                        is_running=lambda: main._running,
                        runner_names=runner_names,
                        on_metrics=lambda payload: self.on_poller_metrics(payload),
                        extra_market_ids_provider=(
                            (lambda: prediction_manager.pending_market_ids()) if prediction_manager is not None else None
                        ),
                    )
                )
                async def _market_refresh_task() -> None:
                    refresh_interval = int(getattr(config, "MARKET_REFRESH_INTERVAL_SECONDS", 0))
                    if refresh_interval <= 0:
                        while main._running:
                            await asyncio.sleep(60.0)
                        return
                    refresh_interval = max(30, refresh_interval)
                    while main._running:
                        await asyncio.sleep(refresh_interval)
                        if not main._running:
                            break
                        try:
                            discovered_ids, discovered_meta, discovered_runner_names = await asyncio.to_thread(
                                discover_markets,
                                client,
                                max_total=max(int(config.SCAN_MAX_MARKETS), len(market_ids)),
                                include_in_play=config.SCAN_INCLUDE_IN_PLAY,
                            )
                            if not discovered_ids:
                                continue
                            desired = max(1, int(config.SCAN_MAX_MARKETS))
                            active_set = set(market_ids)
                            added = 0
                            for mid in discovered_ids:
                                if len(market_ids) >= desired:
                                    break
                                if mid in active_set:
                                    continue
                                market_ids.append(mid)
                                active_set.add(mid)
                                added += 1
                            for mid in list(active_set):
                                if mid in discovered_meta:
                                    market_metadata[mid] = discovered_meta[mid]
                                if mid in discovered_runner_names:
                                    runner_names[mid] = discovered_runner_names[mid]
                            for mid in list(market_metadata.keys()):
                                if mid not in active_set:
                                    market_metadata.pop(mid, None)
                            for mid in list(runner_names.keys()):
                                if mid not in active_set:
                                    runner_names.pop(mid, None)
                            if added > 0:
                                logger.info("Market refresh added %d markets (active=%d)", added, len(market_ids))
                                self.record_event(
                                    "market_refresh",
                                    {"added": added, "active_markets": len(market_ids)},
                                )
                        except Exception as e:
                            logger.exception("Market refresh task failed: %s", e)
                refresh_task = loop.create_task(_market_refresh_task())
                loop_task = loop.create_task(
                    main.run_loop(
                        market_ids,
                        price_cache,
                        risk_manager,
                        paper_executor,
                        scan_interval_seconds=config.SCAN_INTERVAL_SECONDS,
                        on_scan=lambda mid, has_snapshot=None: self.on_scan(mid, has_snapshot),
                        on_opportunity=lambda opp, scored=None: self.on_opportunity(opp, scored),
                        on_trade=lambda opp, result, scored=None: self.on_trade(opp, result, scored),
                        on_prediction=lambda payload: self.on_prediction(payload),
                        on_architect=lambda payload: self.on_architect(payload),
                        market_metadata=market_metadata,
                        candidate_logger=candidate_logger,
                        prediction_manager=prediction_manager,
                        learning_architect=learning_architect,
                        poller_metrics_provider=lambda: dict(self._poller_metrics),
                    )
                )
                async def _qa_task() -> None:
                    if qa_agent is None:
                        while main._running:
                            await asyncio.sleep(60.0)
                        return
                    await qa_agent.run(
                        is_running=lambda: main._running,
                        prediction_manager=prediction_manager,
                        on_decision=lambda payload: self.on_qa(payload),
                        runtime_state_provider=lambda: self._qa_runtime_state(),
                        request_shutdown=lambda: setattr(main, "_running", False),
                        request_restart=lambda reason: self._request_runtime_restart(reason),
                    )
                qa_task = loop.create_task(_qa_task())
                with self._lock:
                    self._active_tasks = [poller_task, loop_task, qa_task, refresh_task]
                await order_monitor.start()
                try:
                    await asyncio.gather(poller_task, loop_task, qa_task, refresh_task)
                except asyncio.CancelledError:
                    pass
                finally:
                    qa_task.cancel()
                    refresh_task.cancel()
                    with self._lock:
                        self._active_tasks = []
                    await order_monitor.stop()
                    if client:
                        try:
                            client.logout()
                        except Exception:
                            pass

            loop.run_until_complete(_async())
            loop.close()
        except Exception as e:
            logger.exception("Trading session error: %s", e)
            with self._lock:
                self._error = str(e)
            if client:
                try:
                    client.logout()
                except Exception:
                    pass
        finally:
            main._running = False
            should_restart = False
            restart_reason = ""
            with self._lock:
                should_restart = bool(self._restart_requested and not self._manual_stop_requested)
                if should_restart:
                    self._last_auto_restart_ts = time.time()
                    restart_reason = self._restart_reason or "qa_requested_restart"
                self._restart_requested = False
                self._restart_reason = ""
                self._loop = None
                self._active_tasks = []
                self._thread = None
                self._qa_agent = None
            if should_restart:
                logger.warning("Auto-restart queued by QA agent: %s", restart_reason)
                # Small delay avoids immediate reconnect spikes after transient API degradation.
                time.sleep(1.0)
                self.start()
