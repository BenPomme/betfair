import config

from monitoring.live_readiness import evaluate_live_trading_readiness


def _betfair_state_ready(paper_mode: bool = True):
    return {
        "running": True,
        "health": {"feed_ok": True, "prediction_ok": True, "risk_ok": True},
        "config": {"paper_trading": paper_mode},
        "prediction_models": {
            "implied_market_1": {
                "model_kind": "implied_market",
                "settled_bets": 500,
                "strict_gate_pass": False,
            },
            "residual_logit_2": {
                "model_kind": "residual_logit",
                "settled_bets": 350,
                "strict_gate_pass": True,
                "rolling_200": {"settled": 200, "roi_pct": 1.2, "brier_lift_abs": 0.02},
            },
            "pure_logit_3": {
                "model_kind": "pure_logit",
                "settled_bets": 360,
                "strict_gate_pass": True,
                "rolling_200": {"settled": 200, "roi_pct": 0.8, "brier_lift_abs": 0.015},
            },
        },
    }


def _funding_state_ready(mode: str = "paper"):
    return {
        "running": True,
        "mode": mode,
        "ws_connected": True,
        "trading_halted": False,
        "realized_roi_pct": 1.1,
        "realized_net_pnl_usd": 15.0,
        "closed_hedges": 10,
        "validation_mode": True,
        "execution_mode": "fail_closed",
        "validation_run_id": "run_123",
        "fresh_book_started_at": "2026-03-06T10:00:00+00:00",
        "execution_quality": {
            "avg_realized_slippage_bps": 3.0,
            "rejection_rate": 0.1,
            "simulated_fill_count": 0,
            "orphaned_single_leg_incidents": 0,
            "stale_open_positions": 0,
        },
        "settlement_audit": {
            "realized_funding_events": 15,
            "funding_cap_applied_count": 0,
        },
        "paper_rejections": {"count": 1, "rate": 0.1, "reasons": {}, "recent": []},
        "positions": [],
        "online_learner": {
            "strict_gate_pass": True,
            "rolling_200": {"settled": 200, "roi_pct": 1.5, "brier_lift_abs": 0.01},
        },
        "contrarian_learner": {
            "strict_gate_pass": True,
            "rolling_200": {"settled": 200, "roi_pct": 0.4, "brier_lift_abs": 0.005},
        },
    }


def test_live_readiness_passes_when_both_systems_validated_and_paper(monkeypatch):
    monkeypatch.setattr(config, "PREDICTION_GATE_ENFORCEMENT_MODE", "strict")
    monkeypatch.setattr(config, "FUNDING_GATE_MODE", "full")
    monkeypatch.setattr(config, "FUNDING_VALIDATION_MIN_CLOSED_HEDGES", 8)
    monkeypatch.setattr(config, "FUNDING_VALIDATION_MIN_SETTLEMENT_EVENTS", 12)
    readiness = evaluate_live_trading_readiness(
        _betfair_state_ready(paper_mode=True),
        _funding_state_ready(mode="paper"),
    )
    assert readiness["validation_ready"] is True
    assert readiness["can_activate_live_now"] is True
    assert readiness["betfair"]["can_switch_to_live_now"] is True
    assert readiness["binance"]["can_switch_to_live_now"] is True
    assert readiness["binance"]["readiness_v2"] is True


def test_live_readiness_blocks_when_rejection_rate_is_too_high(monkeypatch):
    monkeypatch.setattr(config, "PREDICTION_GATE_ENFORCEMENT_MODE", "strict")
    monkeypatch.setattr(config, "FUNDING_GATE_MODE", "full")
    funding = _funding_state_ready(mode="paper")
    funding["execution_quality"]["rejection_rate"] = 0.6
    readiness = evaluate_live_trading_readiness(_betfair_state_ready(paper_mode=True), funding)
    assert readiness["validation_ready"] is False
    assert "rejection_rate_within_limit" in readiness["binance"]["blockers_v2"]


def test_live_readiness_blocks_when_simulated_fill_present(monkeypatch):
    monkeypatch.setattr(config, "PREDICTION_GATE_ENFORCEMENT_MODE", "strict")
    monkeypatch.setattr(config, "FUNDING_GATE_MODE", "full")
    funding = _funding_state_ready(mode="paper")
    funding["execution_quality"]["simulated_fill_count"] = 1
    readiness = evaluate_live_trading_readiness(_betfair_state_ready(paper_mode=True), funding)
    assert readiness["validation_ready"] is False
    assert "no_simulated_fills" in readiness["binance"]["blockers_v2"]


def test_live_readiness_blocks_when_settlement_events_too_low(monkeypatch):
    monkeypatch.setattr(config, "PREDICTION_GATE_ENFORCEMENT_MODE", "strict")
    monkeypatch.setattr(config, "FUNDING_GATE_MODE", "full")
    funding = _funding_state_ready(mode="paper")
    funding["settlement_audit"]["realized_funding_events"] = 2
    readiness = evaluate_live_trading_readiness(_betfair_state_ready(paper_mode=True), funding)
    assert readiness["validation_ready"] is False
    assert "settlement_events_minimum" in readiness["binance"]["blockers_v2"]
