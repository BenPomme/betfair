from datetime import datetime, timezone
from decimal import Decimal
import math

from core.types import PriceSnapshot, SelectionPrice
from data.price_cache import PriceCache
from strategy.prediction_engine import MultiModelPredictionManager, OnlinePredictionEngine


def _snapshot(
    market_id: str,
    a_price: str,
    b_price: str,
    market_status: str = "OPEN",
    a_status: str = "ACTIVE",
    b_status: str = "ACTIVE",
) -> PriceSnapshot:
    return PriceSnapshot(
        market_id=market_id,
        selections=(
            SelectionPrice("1", "A", Decimal(a_price), Decimal("100"), Decimal(a_price), Decimal("100"), runner_status=a_status),
            SelectionPrice("2", "B", Decimal(b_price), Decimal("100"), Decimal(b_price), Decimal("100"), runner_status=b_status),
        ),
        timestamp=datetime.now(timezone.utc),
        market_status=market_status,
    )


def test_prediction_engine_opens_and_settles_with_learning():
    engine = OnlinePredictionEngine(
        model_id="residual_test",
        model_kind="residual_logit",
        initial_balance_eur=100.0,
        stake_fraction=0.1,
        min_stake_eur=2.0,
        max_stake_eur=20.0,
        min_edge=-1.0,  # force opening for deterministic test
        min_liquidity_eur=1.0,
        model_path="data/prediction/test_online_model.json",
        save_every=1,
    )

    s1 = _snapshot("1.1", "3.0", "3.2")
    out1 = engine.process_snapshot("1.1", s1, "Event", None)
    assert any(e["kind"] == "prediction_open" for e in out1["events"])
    assert out1["state"]["total_bets"] == 1

    # Settlement now requires CLOSED market + runner statuses
    s2 = _snapshot("1.1", "2.8", "3.3", market_status="CLOSED", a_status="WINNER", b_status="LOSER")
    out2 = engine.process_snapshot("1.1", s2, "Event", None)
    assert out2["state"]["settled_bets"] >= 1
    assert out2["state"]["model_updates"] >= 1


def test_prediction_engine_resets_on_bust():
    engine = OnlinePredictionEngine(
        model_id="residual_reset",
        model_kind="residual_logit",
        initial_balance_eur=10.0,
        stake_fraction=1.0,
        min_stake_eur=2.0,
        max_stake_eur=10.0,
        min_edge=-1.0,
        min_liquidity_eur=1.0,
        model_path="data/prediction/test_online_model_reset.json",
        save_every=100,
    )
    s1 = _snapshot("1.2", "2.0", "2.2")
    engine.process_snapshot("1.2", s1, "Event2", None)
    # Closed market with first selection LOSER -> loss and potential bust
    s2 = _snapshot("1.2", "2.4", "2.5", market_status="CLOSED", a_status="LOSER", b_status="WINNER")
    out = engine.process_snapshot("1.2", s2, "Event2", None)
    assert out["state"]["resets"] >= 1
    assert out["state"]["balance_eur"] == out["state"]["initial_balance_eur"]


def test_multi_model_manager_parallel_accounts():
    manager = MultiModelPredictionManager(
        model_kinds=["implied_market", "residual_logit", "pure_logit"],
        initial_balance_eur=100000.0,
        stake_fraction=0.01,
        min_stake_eur=2.0,
        max_stake_eur=100.0,
        min_edge=-1.0,
        min_liquidity_eur=1.0,
        model_dir="data/prediction/test_models",
        save_every=100,
    )
    s1 = _snapshot("1.3", "2.9", "3.0")
    out1 = manager.process_snapshot("1.3", s1, "Event3", None)
    assert len(out1["models"]) == 3
    for model in out1["models"].values():
        assert model["initial_balance_eur"] == 100000.0


def test_prediction_engine_state_persists(tmp_path):
    model_path = tmp_path / "residual_model.json"
    state_path = tmp_path / "residual_state.json"
    engine1 = OnlinePredictionEngine(
        model_id="residual_persist",
        model_kind="residual_logit",
        initial_balance_eur=100.0,
        stake_fraction=0.1,
        min_stake_eur=2.0,
        max_stake_eur=20.0,
        min_edge=-1.0,
        min_liquidity_eur=1.0,
        model_path=str(model_path),
        state_path=str(state_path),
        save_every=1,
    )
    s1 = _snapshot("2.1", "3.0", "3.1")
    out1 = engine1.process_snapshot("2.1", s1, "Persist Event", None)
    assert any(e["kind"] == "prediction_open" for e in out1["events"])
    assert state_path.exists()

    engine2 = OnlinePredictionEngine(
        model_id="residual_persist",
        model_kind="residual_logit",
        initial_balance_eur=100.0,
        stake_fraction=0.1,
        min_stake_eur=2.0,
        max_stake_eur=20.0,
        min_edge=-1.0,
        min_liquidity_eur=1.0,
        model_path=str(model_path),
        state_path=str(state_path),
        save_every=1,
    )
    st2 = engine2.get_state()
    assert st2["total_bets"] >= 1
    assert st2["open_positions"] >= 1


def test_pending_market_settlement_watcher_resolves_closed_market(tmp_path):
    manager = MultiModelPredictionManager(
        model_kinds=["implied_market"],
        initial_balance_eur=1000.0,
        stake_fraction=0.05,
        min_stake_eur=2.0,
        max_stake_eur=50.0,
        min_edge=-1.0,
        min_liquidity_eur=1.0,
        model_dir=str(tmp_path / "models"),
        state_dir=str(tmp_path / "state"),
        save_every=10,
    )
    market_id = "9.1"
    s_open = _snapshot(market_id, "3.0", "3.2", market_status="OPEN", a_status="ACTIVE", b_status="ACTIVE")
    out_open = manager.process_snapshot(market_id, s_open, "Watcher Event", None)
    assert any(e["kind"] == "prediction_open" for e in out_open["events"])
    assert market_id in manager.pending_market_ids()

    cache = PriceCache(max_age_seconds=30)
    s_closed = _snapshot(market_id, "2.8", "3.3", market_status="CLOSED", a_status="WINNER", b_status="LOSER")
    cache.set_prices(s_closed)
    settle_out = manager.process_pending_settlements([market_id], price_cache=cache)

    assert any(e["kind"] == "prediction_settle" for e in settle_out["events"])
    model_state = next(iter(settle_out["models"].values()))
    assert model_state["settled_bets"] >= 1
    assert model_state["open_positions"] == 0


def test_pending_market_ids_union(tmp_path):
    manager = MultiModelPredictionManager(
        model_kinds=["implied_market", "pure_logit"],
        initial_balance_eur=1000.0,
        stake_fraction=0.05,
        min_stake_eur=2.0,
        max_stake_eur=50.0,
        min_edge=-1.0,
        min_liquidity_eur=1.0,
        model_dir=str(tmp_path / "models_union"),
        state_dir=str(tmp_path / "state_union"),
        save_every=10,
    )
    s1 = _snapshot("10.1", "2.9", "3.1")
    s2 = _snapshot("10.2", "3.4", "3.6")
    manager.process_snapshot("10.1", s1, "Union Event 1", None)
    manager.process_snapshot("10.2", s2, "Union Event 2", None)
    pending_ids = manager.pending_market_ids()
    assert "10.1" in pending_ids
    assert "10.2" in pending_ids


def test_residual_model_learns_without_opening_bets(tmp_path):
    engine = OnlinePredictionEngine(
        model_id="residual_shadow",
        model_kind="residual_logit",
        initial_balance_eur=100.0,
        stake_fraction=0.1,
        min_stake_eur=2.0,
        max_stake_eur=20.0,
        min_edge=0.5,  # force no open bets initially
        min_liquidity_eur=1.0,
        model_path=str(tmp_path / "residual_shadow_model.json"),
        save_every=1,
    )

    s_open = _snapshot("11.1", "2.0", "2.2", market_status="OPEN")
    out_open = engine.process_snapshot("11.1", s_open, "Shadow Event", None)
    assert not any(e["kind"] == "prediction_open" for e in out_open["events"])
    assert any(e["kind"] == "prediction_learning_track" for e in out_open["events"])
    assert out_open["state"]["total_bets"] == 0

    s_closed = _snapshot("11.1", "2.0", "2.2", market_status="CLOSED", a_status="WINNER", b_status="LOSER")
    out_closed = engine.process_snapshot("11.1", s_closed, "Shadow Event", None)
    assert any(e["kind"] == "prediction_learning_settle" for e in out_closed["events"])
    assert out_closed["state"]["total_bets"] == 0
    assert out_closed["state"]["learning_settled"] >= 1
    assert out_closed["state"]["learning_updates"] >= 1


def test_implied_model_tracks_learning_without_model_updates(tmp_path):
    engine = OnlinePredictionEngine(
        model_id="implied_shadow",
        model_kind="implied_market",
        initial_balance_eur=100.0,
        stake_fraction=0.1,
        min_stake_eur=2.0,
        max_stake_eur=20.0,
        min_edge=0.5,  # impossible for implied model
        min_liquidity_eur=1.0,
        model_path=str(tmp_path / "implied_shadow_model.json"),
        save_every=1,
    )
    s_open = _snapshot("12.1", "2.4", "2.6", market_status="OPEN")
    engine.process_snapshot("12.1", s_open, "Implied Shadow", None)
    s_closed = _snapshot("12.1", "2.4", "2.6", market_status="CLOSED", a_status="LOSER", b_status="WINNER")
    out = engine.process_snapshot("12.1", s_closed, "Implied Shadow", None)
    assert any(e["kind"] == "prediction_learning_settle" for e in out["events"])
    assert out["state"]["learning_settled"] >= 1
    assert out["state"]["learning_updates"] == 0


def test_strict_gate_pass_fail_matrix():
    engine = OnlinePredictionEngine(
        model_id="gate_matrix",
        model_kind="residual_logit",
        initial_balance_eur=1000.0,
        stake_fraction=0.05,
        min_stake_eur=2.0,
        max_stake_eur=20.0,
        min_edge=-1.0,
        min_liquidity_eur=1.0,
        model_path="data/prediction/test_gate_matrix.json",
        save_every=100,
    )
    # 200 settled points, all with positive brier lift and positive ROI.
    engine.settled_bets = 200
    engine._settled_history.clear()
    for _ in range(200):
        engine._settled_history.append(
            {
                "model_brier": 0.18,
                "baseline_brier": 0.22,
                "pnl": 0.15,
                "stake": 1.0,
                "clv": None,
            }
        )
    st = engine.get_state()
    assert st["strict_gate_pass"] is True
    # Flip ROI negative while keeping positive lift -> must fail strict.
    engine._settled_history.clear()
    for _ in range(200):
        engine._settled_history.append(
            {
                "model_brier": 0.18,
                "baseline_brier": 0.22,
                "pnl": -0.15,
                "stake": 1.0,
                "clv": None,
            }
        )
    st2 = engine.get_state()
    assert st2["strict_gate_pass"] is False


def test_settlement_update_rejected_on_non_finite_features(tmp_path):
    engine = OnlinePredictionEngine(
        model_id="reject_nonfinite",
        model_kind="residual_logit",
        initial_balance_eur=100.0,
        stake_fraction=0.1,
        min_stake_eur=2.0,
        max_stake_eur=20.0,
        min_edge=-1.0,
        min_liquidity_eur=1.0,
        model_path=str(tmp_path / "reject_nonfinite_model.json"),
        save_every=1,
    )
    engine._features_from_snapshot = lambda snapshot, market_start: {"spread_mean": float("inf")}  # type: ignore[method-assign]
    s_open = _snapshot("13.1", "3.0", "3.2", market_status="OPEN")
    out_open = engine.process_snapshot("13.1", s_open, "Reject", None)
    assert any(e["kind"] == "prediction_open" for e in out_open["events"])
    s_closed = _snapshot("13.1", "2.8", "3.3", market_status="CLOSED", a_status="WINNER", b_status="LOSER")
    out_closed = engine.process_snapshot("13.1", s_closed, "Reject", None)
    assert any(e["kind"] == "prediction_update_rejected" for e in out_closed["events"])


def test_saturation_and_frozen_detection_state():
    engine = OnlinePredictionEngine(
        model_id="sat_freeze",
        model_kind="pure_logit",
        initial_balance_eur=1000.0,
        stake_fraction=0.05,
        min_stake_eur=2.0,
        max_stake_eur=20.0,
        min_edge=-1.0,
        min_liquidity_eur=1.0,
        model_path="data/prediction/test_sat_freeze.json",
        save_every=100,
    )
    # Saturated + nearly frozen values
    engine._prediction_history.clear()
    for _ in range(120):
        engine._prediction_history.append(0.999)
    st = engine.get_state()
    assert st["prediction_saturation_rate"] > 0.7
    assert st["prediction_frozen"] is True
