from strategy.predictive_model import (
    PredictionExample,
    PureLogitModel,
    ResidualLogitModel,
    evaluate_predictions,
    walk_forward_backtest,
)


def _make_examples(n: int = 300):
    out = []
    for i in range(n):
        imbalance = ((i % 20) - 10) / 10.0
        spread = 0.02 + ((i % 7) * 0.005)
        base_prob = 0.48 + 0.05 * imbalance
        base_prob = max(0.05, min(0.95, base_prob))
        odds = 1.0 / base_prob
        # Label pattern with residual dependency
        y = 1 if (0.6 * imbalance - 2.0 * spread + 0.3) > 0 else 0
        out.append(
            PredictionExample(
                timestamp=f"2026-01-01T00:{i:02d}:00Z",
                base_prob=base_prob,
                odds=odds,
                label=y,
                features={
                    "spread_mean": spread,
                    "imbalance": imbalance,
                    "depth_total_eur": 400 + i,
                    "price_velocity": 0.0,
                    "short_volatility": 0.02,
                    "time_to_start_sec": 3600,
                    "in_play": 0.0,
                },
            )
        )
    return out


def test_residual_model_train_and_predict():
    examples = _make_examples(200)
    model = ResidualLogitModel(
        feature_names=[
            "spread_mean",
            "imbalance",
            "depth_total_eur",
            "price_velocity",
            "short_volatility",
            "time_to_start_sec",
            "in_play",
        ]
    )
    model.fit(examples[:150], epochs=5, lr=0.01)
    p = model.predict_proba(examples[151].base_prob, examples[151].features)
    assert 0.0 < p < 1.0


def test_evaluation_metrics_shape():
    metrics = evaluate_predictions(
        probs=[0.7, 0.4, 0.6, 0.2],
        labels=[1, 0, 1, 0],
        odds=[2.0, 2.5, 1.8, 3.0],
        edge_threshold=0.0,
        stake=1.0,
    )
    assert metrics.brier >= 0
    assert metrics.logloss >= 0
    assert 0 <= metrics.accuracy <= 1


def test_pure_logit_train_and_predict():
    examples = _make_examples(160)
    model = PureLogitModel(
        feature_names=[
            "spread_mean",
            "imbalance",
            "depth_total_eur",
            "price_velocity",
            "short_volatility",
            "time_to_start_sec",
            "in_play",
        ]
    )
    model.fit(examples[:120], epochs=5, lr=0.01)
    p = model.predict_proba(examples[121].features)
    assert 0.0 < p < 1.0


def test_walk_forward_returns_metrics():
    examples = _make_examples(260)
    metrics, probs = walk_forward_backtest(
        examples=examples,
        train_window=120,
        test_window=40,
        epochs=4,
        lr=0.01,
        edge_threshold=0.0,
        stake=1.0,
    )
    assert len(probs) > 0
    assert metrics.brier >= 0
