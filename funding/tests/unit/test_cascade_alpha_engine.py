from __future__ import annotations

import config
from funding.portfolios.cascade_alpha.engine import CascadeAlphaEngine


def test_cascade_alpha_engine_updates_learner(monkeypatch):
    monkeypatch.setattr(config, "CASCADE_ALPHA_MIN_SETTLED_FOR_CANDIDATE", 2)
    monkeypatch.setattr(config, "CASCADE_ALPHA_MIN_WIN_RATE_PCT", 50)
    monkeypatch.setattr(config, "CASCADE_ALPHA_MIN_AVG_NET_PNL_USD", 0)

    engine = CascadeAlphaEngine()
    engine._running = True
    engine._record_learning_sample(
        {
            "trade_id": "t1",
            "symbol": "AAAUSDT",
            "setup": "CONTINUATION",
            "side": "LONG",
            "signal_score": 5.0,
            "spread_bps": 2.0,
            "slippage_bps": 4.0,
            "taker_imbalance": 0.9,
            "net_pnl_usd": 15.0,
            "close_reason": "take_profit",
            "closed_at": "2026-03-06T00:00:00Z",
        }
    )
    engine._record_learning_sample(
        {
            "trade_id": "t2",
            "symbol": "BBBUSDT",
            "setup": "SNAPBACK",
            "side": "SHORT",
            "signal_score": 4.0,
            "spread_bps": 3.0,
            "slippage_bps": 5.0,
            "taker_imbalance": -0.8,
            "net_pnl_usd": 5.0,
            "close_reason": "take_profit",
            "closed_at": "2026-03-06T00:01:00Z",
        }
    )

    state = engine.get_state()
    learner = state["learner"]

    assert learner["settled_count"] == 2
    assert learner["strict_gate_pass"] is True
    assert learner["avg_net_pnl_usd"] > 0
    assert state["readiness"]["status"] == "candidate"
