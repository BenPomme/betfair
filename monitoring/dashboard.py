"""
Dashboard API and UI: start/stop trading, live state, P&L, trades, events.
Run: uvicorn monitoring.dashboard:app --reload --host 0.0.0.0
"""
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Optional

import config
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from monitoring.engine import TradingEngine
from monitoring.live_readiness import evaluate_live_trading_readiness

_engine = TradingEngine()
_template_path = Path(__file__).parent / "templates" / "dashboard.html"
logger = logging.getLogger(__name__)


def get_engine() -> TradingEngine:
    return _engine


def _html() -> str:
    return _template_path.read_text(encoding="utf-8")


app = FastAPI(title="Betfair Arb Dashboard")


def _git_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


logger.info("Dashboard boot: git_sha=%s started_at=%s", _git_sha(), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


@app.get("/", response_class=HTMLResponse)
def index():
    return _html()


@app.post("/api/start")
def api_start():
    return get_engine().start()


@app.post("/api/stop")
def api_stop():
    return get_engine().stop()


@app.get("/api/state")
def api_state():
    state = get_engine().get_state()
    state["live_readiness"] = evaluate_live_trading_readiness(state, _funding_state_snapshot())
    return state


@app.get("/api/model-history")
def api_model_history(limit: int = 200):
    """Prediction model state snapshots over time."""
    results = {}
    state_dir = Path("data/prediction/state")
    if not state_dir.exists():
        return {"models": {}}
    for f in state_dir.glob("*.json"):
        try:
            raw = json.loads(f.read_text(encoding="utf-8"))
            model_id = raw.get("model_id", f.stem)
            results[model_id] = {
                "balance_eur": raw.get("balance", 0),
                "settled_bets": raw.get("settled_bets", 0),
                "avg_brier": raw.get("brier_sum", 0) / max(1, raw.get("settled_bets", 1)),
                "roi_pct": (raw.get("total_pnl", 0) / max(1, raw.get("balance", 100000))) * 100,
                "wins": raw.get("wins", 0),
                "losses": raw.get("losses", 0),
                "model_updates": raw.get("update_count", 0),
                "stake_fraction": raw.get("stake_fraction", 0.05),
                "min_edge": raw.get("min_edge", 0.03),
            }
        except Exception:
            continue
    return {"models": results}


@app.get("/api/feature-weights")
def api_feature_weights():
    """Current model weight vectors from saved prediction models."""
    results = {}
    model_dir = Path("data/prediction/models")
    if not model_dir.exists():
        return {"models": {}}
    for f in model_dir.glob("*.json"):
        try:
            raw = json.loads(f.read_text(encoding="utf-8"))
            # Models store weights as dict or nested
            weights = raw.get("weights", raw.get("w", {}))
            if isinstance(weights, dict):
                results[f.stem] = weights
        except Exception:
            continue
    return {"models": results}


def _read_jsonl_tail(path: Path, max_lines: int = 500) -> list:
    if not path.exists():
        return []
    lines = []
    with path.open("rb") as f:
        f.seek(0, 2)
        size = f.tell()
        data = b""
        block = 1024 * 1024
        while size > 0 and data.count(b"\n") < max_lines + 10:
            rd = min(block, size)
            size -= rd
            f.seek(size)
            data = f.read(rd) + data
    for ln in data.decode("utf-8", "ignore").splitlines():
        if ln.strip():
            try:
                lines.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
    return lines[-max_lines:]


@app.get("/api/clv-history")
def api_clv_history(limit: int = 500):
    """CLV (closing line value) entries over time."""
    clv_dir = Path("data/clv")
    if not clv_dir.exists():
        return {"entries": []}
    all_entries = []
    for f in sorted(clv_dir.glob("*.jsonl")):
        all_entries.extend(_read_jsonl_tail(f, max_lines=limit))
    return {"entries": all_entries[-limit:]}


@app.get("/api/pnl-history")
def api_pnl_history():
    """Balance history from engine state."""
    state = get_engine().get_state()
    return {
        "balance": state.get("balance"),
        "balance_history": state.get("balance_history", []),
        "prediction_models": state.get("prediction_models", {}),
    }


@app.get("/api/architect-history")
def api_architect_history(limit: int = 200):
    """Learning architect decision timeline."""
    path = Path("data/architect/decisions.jsonl")
    return {"decisions": _read_jsonl_tail(path, max_lines=limit)}


@app.get("/api/qa-history")
def api_qa_history(limit: int = 200):
    """QA agent decision timeline."""
    path = Path("data/qa/decisions.jsonl")
    return {"decisions": _read_jsonl_tail(path, max_lines=limit)}


@app.get("/api/audit-report")
def api_audit_report():
    """Latest audit agent report."""
    path = Path("data/audit/reports.jsonl")
    entries = _read_jsonl_tail(path, max_lines=1)
    if entries:
        return {"report": entries[-1]}
    return {"report": None}


@app.get("/api/prediction/experiments")
def api_prediction_experiments(limit: int = 200):
    """Prediction experiment snapshots and best strict-pass run."""
    path = Path("data/prediction/experiments.jsonl")
    entries = _read_jsonl_tail(path, max_lines=limit)
    strict_pass = [e for e in entries if bool(((e.get("gate") or {}).get("strict_gate_pass")))]
    best = None
    if strict_pass:
        best = max(
            strict_pass,
            key=lambda e: float(((e.get("metrics") or {}).get("rolling_200") or {}).get("roi_pct", 0.0)),
        )
    return {"entries": entries[-limit:], "best_strict_pass_run": best}


# === FUNDING MODULE ENDPOINTS ===

_funding_engine = None
_funding_pnl_history = []
_FUNDING_PNL_HISTORY_MAX = 2000


def _append_funding_pnl_snapshot(state: dict) -> None:
    if not state:
        return
    snapshot = {
        "ts": time.time(),
        "realized_net_pnl_usd": float(state.get("realized_net_pnl_usd", 0.0) or 0.0),
        "projected_next_settlement_pnl_usd": float(state.get("projected_next_settlement_pnl_usd", 0.0) or 0.0),
        "total_funding_collected": float(state.get("total_funding_collected", 0.0) or 0.0),
        "total_fees_paid": float(state.get("total_fees_paid", 0.0) or 0.0),
    }
    if _funding_pnl_history:
        prev = _funding_pnl_history[-1]
        if (
            abs(prev.get("realized_net_pnl_usd", 0.0) - snapshot["realized_net_pnl_usd"]) < 1e-12
            and abs(prev.get("projected_next_settlement_pnl_usd", 0.0) - snapshot["projected_next_settlement_pnl_usd"]) < 1e-12
            and snapshot["ts"] - float(prev.get("ts", 0.0)) < 2.0
        ):
            return
    _funding_pnl_history.append(snapshot)
    if len(_funding_pnl_history) > _FUNDING_PNL_HISTORY_MAX:
        del _funding_pnl_history[:-_FUNDING_PNL_HISTORY_MAX]


def set_funding_engine(engine) -> None:
    """Set the funding engine instance for API access."""
    global _funding_engine
    _funding_engine = engine


@app.get("/api/funding/state")
def api_funding_state():
    """Current funding engine state."""
    if _funding_engine is None:
        return {"running": False, "mode": "paper", "ws_connected": False,
                "cache_size": 0, "watchlist_size": 0, "scan_count": 0,
                "opportunity_count": 0, "trade_count": 0, "open_hedges": 0,
                "total_exposure": 0, "total_funding_collected": 0,
                "total_fees_paid": 0, "trading_halted": False, "positions": [],
                "pnl_history": _funding_pnl_history[-300:]}
    state = _funding_engine.get_state()
    _append_funding_pnl_snapshot(state)
    state["pnl_history"] = _funding_pnl_history[-300:]
    return state


@app.get("/api/funding/positions")
def api_funding_positions():
    """Open and closed funding hedge positions."""
    if _funding_engine is None:
        return {"positions": []}
    from funding.execution import executor as funding_executor
    pm = funding_executor.get_position_manager()
    return {"positions": [p.to_dict() for p in pm.all_positions()]}


@app.get("/api/funding/rates")
def api_funding_rates():
    """Current funding rates for watchlist symbols."""
    if _funding_engine is None:
        return {"rates": []}
    return {"rates": _funding_engine.get_funding_rates()}


_funding_task = None
_funding_loop = None  # The event loop running in the funding thread


def _funding_state_snapshot() -> dict:
    if _funding_engine is None:
        return {
            "running": False,
            "mode": str(getattr(config, "FUNDING_MODE", "paper")),
            "ws_connected": False,
            "trading_halted": False,
            "online_learner": {},
            "contrarian_learner": {},
            "realized_roi_pct": 0.0,
        }
    try:
        return _funding_engine.get_state()
    except Exception as exc:
        return {
            "running": False,
            "mode": str(getattr(config, "FUNDING_MODE", "paper")),
            "ws_connected": False,
            "trading_halted": True,
            "online_learner": {},
            "contrarian_learner": {},
            "realized_roi_pct": 0.0,
            "error": str(exc),
        }


@app.get("/api/live-readiness")
def api_live_readiness():
    """Readiness signal for switching both Betfair and Binance from paper to live."""
    betfair_state = get_engine().get_state()
    funding_state = _funding_state_snapshot()
    return evaluate_live_trading_readiness(betfair_state, funding_state)


def _stop_funding_engine(timeout: float = 10.0) -> bool:
    """
    Schedule engine.stop() on the funding thread's event loop, wait for the
    thread to exit (up to *timeout* seconds).  Returns True if clean stop.
    """
    import asyncio
    import logging as _log
    global _funding_engine, _funding_task, _funding_loop

    if _funding_engine is None or _funding_loop is None:
        return True

    # Schedule the coroutine on the engine's own event loop
    future = asyncio.run_coroutine_threadsafe(_funding_engine.stop(), _funding_loop)
    try:
        future.result(timeout=timeout)
    except Exception as e:
        _log.getLogger(__name__).warning("engine.stop() raised: %s", e)

    # Wait for the background thread to finish
    if _funding_task is not None and _funding_task.is_alive():
        _funding_task.join(timeout=timeout)

    _funding_engine = None
    _funding_loop = None
    _funding_task = None
    return True


@app.post("/api/funding/start")
def api_funding_start():
    """Start the funding engine."""
    import asyncio
    import logging as _log
    import threading
    import traceback
    global _funding_engine, _funding_task, _funding_loop

    # If a previous thread is still winding down, wait for it
    if _funding_task is not None and _funding_task.is_alive():
        _funding_task.join(timeout=10.0)
        if _funding_task.is_alive():
            return {"ok": False, "error": "previous_thread_still_alive"}

    if _funding_engine is not None and _funding_engine._running:
        return {"ok": False, "error": "already_running"}

    try:
        from funding.main import FundingEngine
        _funding_engine = FundingEngine()
    except Exception as e:
        return {"ok": False, "error": f"init_failed: {e}"}

    def _run():
        global _funding_loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _funding_loop = loop
        try:
            loop.run_until_complete(_funding_engine.start())
        except Exception as e:
            _log.getLogger(__name__).error("Funding engine error: %s\n%s", e, traceback.format_exc())
        finally:
            _funding_loop = None
            loop.close()

    _funding_task = threading.Thread(target=_run, daemon=True, name="funding-engine")
    _funding_task.start()
    return {"ok": True}


@app.post("/api/funding/stop")
def api_funding_stop():
    """Stop the funding engine and wait for the thread to exit."""
    global _funding_engine
    if _funding_engine is None or not _funding_engine._running:
        return {"ok": False, "error": "not_running"}

    _stop_funding_engine(timeout=10.0)
    return {"ok": True}


@app.post("/api/funding/restart")
def api_funding_restart():
    """Stop the current funding engine (if running) and start a fresh one."""
    import asyncio
    import logging as _log
    import threading
    import traceback
    global _funding_engine, _funding_task, _funding_loop

    # Stop existing engine if present
    _stop_funding_engine(timeout=10.0)

    # Start a new engine
    try:
        from funding.main import FundingEngine
        _funding_engine = FundingEngine()
    except Exception as e:
        return {"ok": False, "error": f"init_failed: {e}"}

    def _run():
        global _funding_loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _funding_loop = loop
        try:
            loop.run_until_complete(_funding_engine.start())
        except Exception as e:
            _log.getLogger(__name__).error("Funding engine error: %s\n%s", e, traceback.format_exc())
        finally:
            _funding_loop = None
            loop.close()

    _funding_task = threading.Thread(target=_run, daemon=True, name="funding-engine")
    _funding_task.start()
    return {"ok": True}


@app.get("/api/funding/ml-metrics")
def api_funding_ml_metrics():
    """ML model metrics and feature importances."""
    model_dir = Path("data/funding_models")
    meta_path = model_dir / "funding_predictor_meta.json"
    if not meta_path.exists():
        return {"loaded": False}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return {"loaded": True, **meta}


@app.get("/api/funding/ml-status")
def api_funding_ml_status():
    """Online learner status: AUC, retrain history, prediction accuracy."""
    if _funding_engine is None:
        return {
            "running": False,
            "current_auc": 0,
            "retrain_count": 0,
            "prediction_accuracy": 0,
            "retrain_history": [],
            "funding_summary": {
                "eligible_models_count": 0,
                "strict_gate_pass_rate": 0.0,
                "weighted_win_rate_pct": 0.0,
                "strict_gate_pass": False,
            },
            "online_learner": {},
            "contrarian_learner": {},
        }
    state = _funding_engine.get_state()
    return {
        **(state.get("online_learner", {}) or {}),
        "online_learner": state.get("online_learner", {}) or {},
        "contrarian_learner": state.get("contrarian_learner", {}) or {},
        "funding_summary": state.get("funding_summary", {}) or {},
    }


@app.get("/api/funding/experiments")
def api_funding_experiments(limit: int = 200):
    """Funding experiment snapshots and best strict-pass run."""
    path = Path(config.FUNDING_EXPERIMENT_LOG_PATH)
    entries = _read_jsonl_tail(path, max_lines=limit)
    strict_pass = [e for e in entries if bool(((e.get("gate") or {}).get("strict_gate_pass")))]
    best = None
    if strict_pass:
        best = max(
            strict_pass,
            key=lambda e: float(((e.get("metrics") or {}).get("rolling_200") or {}).get("roi_pct", 0.0)),
        )
    return {"entries": entries[-limit:], "best_strict_pass_run": best}


@app.get("/api/orchestrator/state")
def api_orchestrator_state():
    """Strategy orchestrator state: phase, data progress, model metrics, last decision."""
    if _funding_engine is None:
        return {"enabled": False}
    state = _funding_engine.get_state()
    orch = state.get("orchestrator")
    if not orch:
        return {"enabled": False}
    # Append last decision from JSONL log
    log_path = Path(orch.get("log_dir", "data/funding_agents")) / "orchestrator_decisions.jsonl"
    if not log_path.exists():
        log_path = Path("data/funding_agents/orchestrator_decisions.jsonl")
    decisions = _read_jsonl_tail(log_path, max_lines=10)
    orch["recent_decisions"] = decisions
    orch["enabled"] = True
    return orch


@app.get("/api/contrarian/state")
def api_contrarian_state():
    """Directional positions, P&L, win rate; enabled from config, active when Funding engine runs."""
    import config as _cfg
    if _funding_engine is None:
        return {
            "enabled": getattr(_cfg, "CONTRARIAN_ENABLED", False),
            "active": False,
            "message": "Start Funding engine to activate.",
        }
    state = _funding_engine.get_state()
    contrarian = state.get("contrarian") or {}
    learner = state.get("contrarian_learner") or {}
    positions = contrarian.get("positions") or contrarian.get("open_positions") or []
    contrarian["positions"] = positions
    contrarian["open_positions"] = positions
    if "model_name" not in contrarian:
        contrarian["model_name"] = contrarian.get("model")
    if "total_realized_pnl" not in contrarian:
        contrarian["total_realized_pnl"] = learner.get("total_realized_pnl", 0.0)
    contrarian["strict_gate_pass"] = learner.get("strict_gate_pass")
    contrarian["strict_gate_reason"] = learner.get("strict_gate_reason")
    contrarian["settled_count"] = learner.get("settled_count", 0)
    contrarian["enabled"] = getattr(_cfg, "CONTRARIAN_ENABLED", False)
    contrarian["active"] = True
    return contrarian


@app.get("/api/regime/state")
def api_regime_state():
    """Current market regime and adapter state; enabled from config, active when Funding runs."""
    import config as _cfg
    enabled = getattr(_cfg, "REGIME_ENABLED", False)
    if _funding_engine is None:
        return {
            "enabled": enabled,
            "active": False,
            "message": "Start Funding engine to activate.",
            "regime": "disabled" if not enabled else "unknown",
            "label": "disabled" if not enabled else "unknown",
            "last_updated": None,
        }
    state = _funding_engine.get_state()
    regime = state.get("regime") or {}
    if not regime:
        return {
            "enabled": enabled,
            "active": True,
            "regime": "disabled" if not enabled else "unknown",
            "label": "disabled" if not enabled else "unknown",
            "last_updated": None,
            "message": "Regime adapter disabled" if not enabled else "Regime adapter not initialized",
        }
    label = regime.get("regime_label") or regime.get("label") or "unknown"
    out = {
        "enabled": enabled,
        "active": True,
        "regime": str(label),
        "label": str(label),
        "last_updated": regime.get("last_update") or regime.get("last_updated"),
        "current_regime": regime.get("current_regime"),
        "halt_trading": bool(regime.get("halt_trading", False)),
        "adjustments": regime.get("adjustments"),
    }
    proba = regime.get("regime_proba")
    if isinstance(proba, list) and proba:
        try:
            out["confidence"] = float(max(proba))
        except Exception:
            pass
    return out


@app.get("/api/cascade/state")
def api_cascade_state():
    """Cascade monitor state; enabled from config, active when Funding engine runs."""
    import config as _cfg
    enabled = getattr(_cfg, "CASCADE_ENABLED", False)
    if _funding_engine is None:
        return {
            "enabled": enabled,
            "active": False,
            "message": "Start Funding engine to activate." if enabled else "Enable CASCADE_ENABLED and start Funding.",
        }
    return {"enabled": enabled, "active": True}


@app.get("/api/strategy-overview")
def api_strategy_overview():
    """Combined P&L across all strategies (hedge + contrarian)."""
    if _funding_engine is None:
        return {"enabled": False}
    state = _funding_engine.get_state()
    hedge_pnl = state.get("total_funding_collected", 0)
    fees = state.get("total_fees_paid", 0)
    hedge_net = hedge_pnl - fees
    assumed_capital = float(state.get("assumed_capital_usd", 0) or 0)
    hedge_roi_pct = (hedge_net / assumed_capital * 100.0) if assumed_capital > 0 else 0.0
    contrarian = state.get("contrarian") or {}
    contrarian_pnl = contrarian.get("total_pnl", 0)
    combined_net = hedge_net + contrarian_pnl
    combined_roi_pct = (combined_net / assumed_capital * 100.0) if assumed_capital > 0 else 0.0
    return {
        "enabled": True,
        "funding_summary": state.get("funding_summary", {}) or {},
        "assumed_capital_usd": assumed_capital,
        "hedge": {
            "total_funding_collected": hedge_pnl,
            "open_hedges": state.get("open_hedges", 0),
            "total_fees_paid": fees,
            "net_pnl": hedge_net,
            "roi_pct": hedge_roi_pct,
            "projected_next_settlement_pnl_usd": float(state.get("projected_next_settlement_pnl_usd", hedge_net)),
            "projected_next_settlement_roi_pct": float(state.get("projected_next_settlement_roi_pct", hedge_roi_pct)),
        },
        "contrarian": {
            "total_pnl": contrarian_pnl,
            "win_rate": contrarian.get("win_rate", 0),
            "trade_count": contrarian.get("trade_count", 0),
        },
        "combined_net_pnl": combined_net,
        "combined_roi_pct": combined_roi_pct,
    }
