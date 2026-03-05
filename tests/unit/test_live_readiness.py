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
    readiness = evaluate_live_trading_readiness(
        _betfair_state_ready(paper_mode=True),
        _funding_state_ready(mode="paper"),
    )
    assert readiness["validation_ready"] is True
    assert readiness["can_activate_live_now"] is True
    assert readiness["betfair"]["can_switch_to_live_now"] is True
    assert readiness["binance"]["can_switch_to_live_now"] is True


def test_live_readiness_blocks_when_betfair_models_fail_strict_gate(monkeypatch):
    monkeypatch.setattr(config, "PREDICTION_GATE_ENFORCEMENT_MODE", "strict")
    monkeypatch.setattr(config, "FUNDING_GATE_MODE", "full")
    betfair = _betfair_state_ready(paper_mode=True)
    betfair["prediction_models"]["pure_logit_3"]["strict_gate_pass"] = False
    betfair["prediction_models"]["pure_logit_3"]["rolling_200"]["roi_pct"] = -0.2
    readiness = evaluate_live_trading_readiness(betfair, _funding_state_ready(mode="paper"))
    assert readiness["validation_ready"] is False
    assert readiness["can_activate_live_now"] is False
    assert readiness["betfair"]["validation_ready"] is False
    assert any(b.startswith("betfair:") for b in readiness["blockers"])


def test_live_readiness_reports_already_live_mode_even_if_validated(monkeypatch):
    monkeypatch.setattr(config, "PREDICTION_GATE_ENFORCEMENT_MODE", "strict")
    monkeypatch.setattr(config, "FUNDING_GATE_MODE", "full")
    readiness = evaluate_live_trading_readiness(
        _betfair_state_ready(paper_mode=False),
        _funding_state_ready(mode="live"),
    )
    assert readiness["validation_ready"] is True
    assert readiness["can_activate_live_now"] is False
    assert "already_live_mode" in readiness["betfair"]["blockers"]
    assert "already_live_mode" in readiness["binance"]["blockers"]
