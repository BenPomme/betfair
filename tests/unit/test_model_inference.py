from datetime import datetime, timezone
from decimal import Decimal
import importlib

from core.types import Opportunity, PriceSnapshot, SelectionPrice
from strategy.features import build_feature_vector
from strategy.model_inference import score_opportunity


def _build_case(net_profit: Decimal, net_roi: Decimal) -> tuple[Opportunity, PriceSnapshot]:
    snapshot = PriceSnapshot(
        market_id="1.model",
        selections=(
            SelectionPrice("1", "A", Decimal("3.00"), Decimal("200"), Decimal("3.05"), Decimal("180")),
            SelectionPrice("2", "B", Decimal("3.00"), Decimal("200"), Decimal("3.05"), Decimal("180")),
        ),
        timestamp=datetime.now(timezone.utc),
    )
    opp = Opportunity(
        market_id="1.model",
        event_name="Model Event",
        market_start=datetime.now(timezone.utc),
        arb_type="back_back",
        selections=(
            {"selection_id": "1", "name": "A", "back_price": 3.0, "stake_eur": 50.0, "liquidity_eur": 200.0},
            {"selection_id": "2", "name": "B", "back_price": 3.0, "stake_eur": 50.0, "liquidity_eur": 200.0},
        ),
        total_stake_eur=Decimal("100"),
        overround_raw=Decimal("0.6666"),
        gross_profit_eur=net_profit + Decimal("0.2"),
        commission_eur=Decimal("0.2"),
        net_profit_eur=net_profit,
        net_roi_pct=net_roi,
        liquidity_by_selection=(Decimal("200"), Decimal("200")),
    )
    return opp, snapshot


def test_score_returns_structured_decision():
    opp, snap = _build_case(Decimal("0.50"), Decimal("0.005"))
    features = build_feature_vector(snap, opp, market_start=opp.market_start)
    scored = score_opportunity(opp, features)
    assert scored.decision in {"EXECUTE", "DEFER", "SKIP"}
    assert scored.fill_prob >= Decimal("0")
    assert scored.fill_prob <= Decimal("1")
    assert scored.dynamic_threshold_eur >= Decimal("0")


def test_boundary_behavior_low_expected_value_not_execute():
    opp, snap = _build_case(Decimal("0.01"), Decimal("0.0001"))
    features = build_feature_vector(snap, opp, market_start=opp.market_start)
    scored = score_opportunity(opp, features)
    assert scored.decision in {"DEFER", "SKIP"}


def test_fallback_when_model_artifact_missing(monkeypatch):
    import strategy.model_inference as mi

    monkeypatch.setenv("ML_LINEAR_MODEL_PATH", "/tmp/does-not-exist-model.json")
    mi = importlib.reload(mi)

    opp, snap = _build_case(Decimal("0.40"), Decimal("0.004"))
    features = build_feature_vector(snap, opp, market_start=opp.market_start)
    scored = mi.score_opportunity(opp, features)
    assert scored.model_version in {"heuristic_v1", "linear_v1", "disabled"}


def test_prediction_boost_applies_when_calibrated():
    opp, snap = _build_case(Decimal("0.40"), Decimal("0.004"))
    features = build_feature_vector(snap, opp, market_start=opp.market_start)
    scored = score_opportunity(
        opp,
        features,
        prediction_confidence={
            "best_model_id": "residual_logit_1",
            "predicted_prob": 0.58,
            "edge_vs_market": 0.06,
            "model_brier": 0.21,
            "settled_bets": 100,
        },
    )
    assert scored.prediction_influence == "boosted"


def test_prediction_penalty_applies_when_negative_edge():
    opp, snap = _build_case(Decimal("0.40"), Decimal("0.004"))
    features = build_feature_vector(snap, opp, market_start=opp.market_start)
    scored = score_opportunity(
        opp,
        features,
        prediction_confidence={
            "best_model_id": "residual_logit_1",
            "predicted_prob": 0.30,
            "edge_vs_market": -0.04,
            "model_brier": 0.21,
            "settled_bets": 120,
        },
    )
    assert scored.prediction_influence == "penalized"


def test_prediction_ignored_when_insufficient_data():
    opp, snap = _build_case(Decimal("0.40"), Decimal("0.004"))
    features = build_feature_vector(snap, opp, market_start=opp.market_start)
    scored = score_opportunity(
        opp,
        features,
        prediction_confidence={
            "best_model_id": "residual_logit_1",
            "predicted_prob": 0.58,
            "edge_vs_market": 0.06,
            "model_brier": 0.21,
            "settled_bets": 5,
        },
    )
    assert scored.prediction_influence == "ignored_insufficient_data"


def test_stake_multiplier_increases_with_expected_value():
    high_opp, high_snap = _build_case(Decimal("0.80"), Decimal("0.008"))
    low_opp, low_snap = _build_case(Decimal("0.12"), Decimal("0.0012"))
    high_features = build_feature_vector(high_snap, high_opp, market_start=high_opp.market_start)
    low_features = build_feature_vector(low_snap, low_opp, market_start=low_opp.market_start)

    high_scored = score_opportunity(high_opp, high_features)
    low_scored = score_opportunity(low_opp, low_features)

    assert high_scored.stake_multiplier >= low_scored.stake_multiplier
    assert high_scored.stake_multiplier > Decimal("0")


def test_prediction_penalty_reduces_stake_multiplier():
    opp, snap = _build_case(Decimal("0.45"), Decimal("0.0045"))
    features = build_feature_vector(snap, opp, market_start=opp.market_start)

    boosted = score_opportunity(
        opp,
        features,
        prediction_confidence={
            "best_model_id": "residual_logit_1",
            "predicted_prob": 0.61,
            "edge_vs_market": 0.07,
            "model_brier": 0.20,
            "settled_bets": 100,
        },
    )
    penalized = score_opportunity(
        opp,
        features,
        prediction_confidence={
            "best_model_id": "residual_logit_1",
            "predicted_prob": 0.30,
            "edge_vs_market": -0.05,
            "model_brier": 0.20,
            "settled_bets": 100,
        },
    )

    assert boosted.stake_multiplier > penalized.stake_multiplier
