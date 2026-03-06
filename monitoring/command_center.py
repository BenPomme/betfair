from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

import config
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from monitoring.live_readiness import evaluate_live_trading_readiness
from monitoring.portfolio_process_manager import PortfolioProcessManager
from monitoring.portfolio_registry import get_portfolio_spec, list_portfolios
from portfolio.accounting import build_strategy_account
from portfolio.state_store import PortfolioStateStore
from portfolio.types import PortfolioState, PortfolioSummary

logger = logging.getLogger(__name__)
app = FastAPI(title="Strategy Command Center")
_process_manager = PortfolioProcessManager()
_template_path = Path(__file__).parent / "templates" / "command_center.html"


def _git_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def _html() -> str:
    return _template_path.read_text(encoding="utf-8")


def _readiness_label(readiness: Dict[str, Any]) -> str:
    if not readiness:
        return "unknown"
    if readiness.get("research_only"):
        return "research_only"
    if bool(readiness.get("can_switch_to_live_now")):
        return "ready"
    if bool(readiness.get("validation_ready")):
        return "validated"
    if bool(readiness.get("readiness_v2")):
        return "validated"
    blockers = readiness.get("blockers") or readiness.get("blockers_v2") or []
    if blockers:
        return "blocked"
    return str(readiness.get("status", "monitoring"))


def _default_account(spec) -> dict:
    account = build_strategy_account(
        portfolio_id=spec.portfolio_id,
        currency=spec.currency,
        starting_balance=spec.initial_balance,
        current_balance=spec.initial_balance,
        realized_pnl=0.0,
    )
    return account.to_dict()


def _extract_positions(raw_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    positions = raw_state.get("open_positions")
    if isinstance(positions, list):
        return positions
    positions = raw_state.get("positions")
    if isinstance(positions, list):
        return positions
    all_positions = raw_state.get("all_positions")
    if isinstance(all_positions, list):
        return all_positions
    return []


def _build_snapshot(portfolio_id: str) -> Dict[str, Any]:
    spec = get_portfolio_spec(portfolio_id)
    store = PortfolioStateStore(portfolio_id)
    raw_state = store.read_state() or {}
    account = (store.read_account().to_dict() if store.read_account() is not None else _default_account(spec))
    readiness = store.read_readiness() or {}
    heartbeat = store.read_heartbeat() or {}
    models = store.read_models()
    trades = store.read_trades(limit=200)
    events = store.read_events(limit=200)
    balance_history = store.read_balance_history(limit=1000)
    pid = store.read_pid()
    process_status = _process_manager.status(portfolio_id)
    process_running = bool(process_status.get("running"))
    raw_running = raw_state.get("running")
    running = process_running and (bool(raw_running) if raw_running is not None else True)
    positions = _extract_positions(raw_state)
    open_count = len(positions)
    if open_count == 0:
        open_count = int(raw_state.get("open_hedges", raw_state.get("opportunity_count", 0)) or 0)
    summary_status = str(raw_state.get("status", "idle"))
    if running:
        summary_status = "running"
    elif raw_state.get("error"):
        summary_status = "error"
    elif summary_status == "running":
        summary_status = "idle"
    summary = PortfolioSummary(
        portfolio_id=spec.portfolio_id,
        label=spec.label,
        category=spec.category,
        control_mode=spec.control_mode,
        running=running,
        mode=str(raw_state.get("mode", "paper")),
        bankroll=float(account.get("starting_balance", spec.initial_balance) or spec.initial_balance),
        currency=spec.currency,
        realized_pnl=float(account.get("realized_pnl", 0.0) or 0.0),
        unrealized_pnl=float(account.get("unrealized_pnl", 0.0) or 0.0),
        roi_pct=float(account.get("roi_pct", 0.0) or 0.0),
        max_drawdown_pct=float(account.get("drawdown_pct", 0.0) or 0.0),
        open_count=open_count,
        readiness=_readiness_label(readiness),
        last_heartbeat_ts=heartbeat.get("ts"),
        status=summary_status,
        process_pid=pid,
        errors=[raw_state.get("error")] if raw_state.get("error") else [],
    )
    state = PortfolioState(
        portfolio_id=spec.portfolio_id,
        running=running,
        read_only=False,
        status=summary.status,
        account=account,
        config=store.read_config_snapshot() or {},
        metrics={
            key: raw_state.get(key)
            for key in [
                "scan_count",
                "signal_count",
                "trade_count",
                "watchlist_size",
                "realized_pnl_usd",
                "shadow_realized_pnl_usd",
                "realized_net_pnl_usd",
                "realized_roi_pct",
            ]
            if key in raw_state
        },
        positions=positions,
        recent_events=events,
        recent_trades=trades,
        execution_quality=raw_state.get("execution_quality") or {},
        risk=raw_state.get("risk") or {},
        readiness=readiness,
        models=models,
        balance_history=balance_history,
        raw_state=raw_state,
        control_mode=spec.control_mode,
        error=raw_state.get("error"),
    )
    return {"summary": summary.to_dict(), "state": state.to_dict()}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _html()


@app.get("/api/portfolios")
def api_portfolios() -> Dict[str, Any]:
    snapshots = [_build_snapshot(spec.portfolio_id) for spec in list_portfolios()]
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": _git_sha(),
        "portfolios": [item["summary"] for item in snapshots],
    }


@app.get("/api/portfolios/{portfolio_id}/summary")
def api_portfolio_summary(portfolio_id: str) -> Dict[str, Any]:
    return _build_snapshot(portfolio_id)["summary"]


@app.get("/api/portfolios/{portfolio_id}/state")
def api_portfolio_state(portfolio_id: str) -> Dict[str, Any]:
    return _build_snapshot(portfolio_id)["state"]


@app.get("/api/portfolios/{portfolio_id}/performance")
def api_portfolio_performance(portfolio_id: str) -> Dict[str, Any]:
    state = _build_snapshot(portfolio_id)["state"]
    return {
        "portfolio_id": portfolio_id,
        "account": state.get("account") or {},
        "balance_history": state.get("balance_history") or [],
    }


@app.get("/api/portfolios/{portfolio_id}/trades")
def api_portfolio_trades(portfolio_id: str) -> Dict[str, Any]:
    state = _build_snapshot(portfolio_id)["state"]
    return {"portfolio_id": portfolio_id, "trades": state.get("recent_trades") or []}


@app.get("/api/portfolios/{portfolio_id}/events")
def api_portfolio_events(portfolio_id: str) -> Dict[str, Any]:
    state = _build_snapshot(portfolio_id)["state"]
    return {"portfolio_id": portfolio_id, "events": state.get("recent_events") or []}


@app.get("/api/portfolios/{portfolio_id}/models")
def api_portfolio_models(portfolio_id: str) -> Dict[str, Any]:
    state = _build_snapshot(portfolio_id)["state"]
    return {"portfolio_id": portfolio_id, "models": state.get("models") or []}


@app.get("/api/portfolios/{portfolio_id}/readiness")
def api_portfolio_readiness(portfolio_id: str) -> Dict[str, Any]:
    state = _build_snapshot(portfolio_id)["state"]
    return {"portfolio_id": portfolio_id, "readiness": state.get("readiness") or {}}


@app.post("/api/portfolios/{portfolio_id}/start")
def api_portfolio_start(portfolio_id: str) -> Dict[str, Any]:
    result = _process_manager.start(portfolio_id)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result)
    return result


@app.post("/api/portfolios/{portfolio_id}/stop")
def api_portfolio_stop(portfolio_id: str) -> Dict[str, Any]:
    result = _process_manager.stop(portfolio_id)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result)
    return result


@app.post("/api/portfolios/{portfolio_id}/restart")
def api_portfolio_restart(portfolio_id: str) -> Dict[str, Any]:
    result = _process_manager.restart(portfolio_id)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result)
    return result


@app.get("/api/compare/portfolios")
def api_compare_portfolios() -> Dict[str, Any]:
    snapshots = [_build_snapshot(spec.portfolio_id) for spec in list_portfolios()]
    return {
        "portfolios": [item["summary"] for item in snapshots],
        "series": {
            item["summary"]["portfolio_id"]: item["state"].get("balance_history") or []
            for item in snapshots
        },
        "readiness": {
            item["summary"]["portfolio_id"]: item["state"].get("readiness") or {}
            for item in snapshots
        },
    }


@app.get("/api/state")
def api_state() -> Dict[str, Any]:
    return _build_snapshot("betfair_core")["state"].get("raw_state") or {}


@app.post("/api/start")
def api_start() -> Dict[str, Any]:
    return api_portfolio_start("betfair_core")


@app.post("/api/stop")
def api_stop() -> Dict[str, Any]:
    return api_portfolio_stop("betfair_core")


@app.get("/api/funding/state")
def api_funding_state() -> Dict[str, Any]:
    return _build_snapshot("hedge_validation")["state"].get("raw_state") or {}


@app.post("/api/funding/start")
def api_funding_start() -> Dict[str, Any]:
    return api_portfolio_start("hedge_validation")


@app.post("/api/funding/stop")
def api_funding_stop() -> Dict[str, Any]:
    return api_portfolio_stop("hedge_validation")


@app.post("/api/funding/restart")
def api_funding_restart() -> Dict[str, Any]:
    return api_portfolio_restart("hedge_validation")


@app.get("/api/funding/positions")
def api_funding_positions() -> Dict[str, Any]:
    state = _build_snapshot("hedge_validation")["state"]
    return {"positions": state.get("raw_state", {}).get("all_positions") or state.get("positions") or []}


@app.get("/api/funding/rates")
def api_funding_rates() -> Dict[str, Any]:
    state = _build_snapshot("hedge_validation")["state"]
    return {"rates": state.get("raw_state", {}).get("rates") or []}


@app.get("/api/live-readiness")
def api_live_readiness() -> Dict[str, Any]:
    betfair = _build_snapshot("betfair_core")["state"].get("raw_state") or {}
    binance = _build_snapshot("hedge_validation")["state"].get("raw_state") or {}
    return evaluate_live_trading_readiness(betfair, binance)


@app.get("/api/strategy-overview")
def api_strategy_overview() -> Dict[str, Any]:
    snapshots = [_build_snapshot(spec.portfolio_id) for spec in list_portfolios()]
    summaries = [item["summary"] for item in snapshots]
    return {
        "enabled": True,
        "portfolio_scoped": True,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "portfolios": summaries,
        "by_id": {summary["portfolio_id"]: summary for summary in summaries},
    }


logger.info("Command center boot: git_sha=%s started_at=%s", _git_sha(), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
