from __future__ import annotations

from typing import Any, Dict, Iterable, Optional


_OWNER_BY_PORTFOLIO = {
    "betfair_core": "betfair",
    "betfair_prediction_league": "betfair_prediction",
    "betfair_suspension_lag": "betfair_info_arb",
    "betfair_crossbook_consensus": "betfair_info_arb",
    "betfair_timezone_decay": "betfair_info_arb",
    "polymarket_quantum_fold": "polymarket",
    "polymarket_binary_research": "polymarket",
    "hedge_validation": "funding",
    "hedge_research": "funding",
    "cascade_alpha": "funding",
    "contrarian_legacy": "funding",
    "mev_scout_sol": "onchain",
}

_EXTERNAL_TOKENS = (
    "auth",
    "binance",
    "provider",
    "wallet",
    "rpc",
    "jito",
    "exchange",
)
_UNDERPERFORMING_TOKENS = (
    "negative_brier_lift",
    "negative_roi",
    "negative_shadow_pnl",
    "negative_calibration_lift",
    "avg_net_pnl_below_threshold",
    "win_rate_below_threshold",
    "fully_observed_underperforming",
)
_DATA_TOKENS = (
    "no_binary_microstructure_candidates",
    "not_enough_external_quote_sources",
    "awaiting_multi_source_event_confirmation",
    "market_ops_signal_below_threshold",
    "no_snapshots",
    "no_extreme_candidates",
    "insufficient_labeled_examples",
    "empty_watchlist",
)


def _iter_blockers(readiness: Dict[str, Any]) -> Iterable[str]:
    blockers = readiness.get("blockers_v2") or readiness.get("blockers") or []
    for blocker in blockers:
        if blocker is None:
            continue
        yield str(blocker)


def _closed_trade_count(state: Dict[str, Any]) -> int:
    trades = [item for item in (state.get("recent_trades") or []) if isinstance(item, dict)]
    closed = 0
    for trade in trades:
        status = str(trade.get("status") or "").upper()
        if status in {"CLOSED", "SETTLED"} or trade.get("closed_at") or trade.get("settled_at"):
            closed += 1
    account = dict(state.get("account") or {})
    return max(closed, int(account.get("trade_count", 0) or 0))


def _candidate_count(summary: Dict[str, Any], state: Dict[str, Any]) -> int:
    metrics = dict(state.get("metrics") or {})
    raw_state = dict(state.get("raw_state") or {})
    latest_candidates = raw_state.get("latest_candidates") or []
    return max(
        int(metrics.get("candidate_count", 0) or 0),
        len(latest_candidates) if isinstance(latest_candidates, list) else 0,
        int(summary.get("open_count", 0) or 0),
    )


def _learning_depth(state: Dict[str, Any]) -> int:
    raw_state = dict(state.get("raw_state") or {})
    training_progress = dict(raw_state.get("training_progress") or {})
    research_summary = dict(raw_state.get("research_summary") or {})
    learner = dict(raw_state.get("learner") or {})
    model_settled = 0
    for model in state.get("models") or []:
        if not isinstance(model, dict):
            continue
        metrics = dict(model.get("metrics") or {})
        model_settled = max(
            model_settled,
            int(metrics.get("learning_settled", 0) or 0),
            int(metrics.get("settled_count", model.get("settled_count", 0)) or 0),
        )
    return max(
        model_settled,
        int(training_progress.get("labeled_examples", 0) or 0),
        int(training_progress.get("settled_trades", 0) or 0),
        int(research_summary.get("labeled_examples", 0) or 0),
        int(learner.get("prediction_total", 0) or learner.get("settled_count", 0) or 0),
    )


def _has_token(blockers: Iterable[str], tokens: Iterable[str]) -> bool:
    joined = " ".join(str(item).lower() for item in blockers)
    return any(token in joined for token in tokens)


def classify_portfolio(
    summary: Dict[str, Any],
    state: Dict[str, Any],
    *,
    latest_research_run: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    portfolio_id = str(summary.get("portfolio_id") or "")
    readiness = dict(state.get("readiness") or {})
    blockers = list(_iter_blockers(readiness))
    running = bool(summary.get("running"))
    candidate_count = _candidate_count(summary, state)
    closed_trades = _closed_trade_count(state)
    learning_depth = _learning_depth(state)
    realized_pnl = float(summary.get("realized_pnl", 0.0) or 0.0)
    research_status = str((latest_research_run or {}).get("status") or "").lower()
    owner = _OWNER_BY_PORTFOLIO.get(portfolio_id, "ops")

    if research_status in {"running", "queued", "pending", "reseeding"}:
        audit_state = "reseeding"
        next_action = f"await_goldfish_finalize:{(latest_research_run or {}).get('run_id', 'unknown')}"
    elif not running and _has_token(blockers, _EXTERNAL_TOKENS):
        audit_state = "external_blocked"
        next_action = blockers[0] if blockers else "await_external_dependency"
    elif _has_token(blockers, _EXTERNAL_TOKENS):
        audit_state = "external_blocked"
        next_action = blockers[0]
    elif _has_token(blockers, _UNDERPERFORMING_TOKENS):
        audit_state = "underperforming"
        next_action = "reseed_from_goldfish_replay"
    elif realized_pnl < 0.0 and learning_depth >= 25 and closed_trades > 0:
        audit_state = "underperforming"
        next_action = "tighten_assumptions_then_reseed"
    elif candidate_count > 0 and closed_trades == 0:
        audit_state = "stalled_gating"
        next_action = blockers[0] if blockers else "relax_gate_and_force_paper_cycle"
    elif candidate_count == 0 and closed_trades == 0 and (
        learning_depth < 25 or _has_token(blockers, _DATA_TOKENS)
    ):
        audit_state = "stalled_data"
        next_action = blockers[0] if blockers else "repair_data_intake"
    else:
        audit_state = "learning_ok"
        next_action = "monitor_and_promote_if_stable"

    return {
        "audit_state": audit_state,
        "audit_owner": owner,
        "audit_next_action": next_action,
    }
