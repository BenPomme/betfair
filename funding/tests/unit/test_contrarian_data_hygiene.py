from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from funding.ml.contrarian_features import build_contrarian_features, load_quarantined_symbols
from funding.ml.contrarian_learner import ContrarianOnlineLearner
from funding.ml.contrarian_xgb import ContrarianXGBoost
import funding.ml.contrarian_features as contrarian_features_module


def test_build_contrarian_features_quarantines_symbol_with_invalid_mark_price(tmp_path, monkeypatch):
    quarantine_path = tmp_path / "quarantine.jsonl"
    monkeypatch.setattr(contrarian_features_module, "CONTRARIAN_QUARANTINE_PATH", quarantine_path)

    funding_dir = tmp_path / "funding_rates"
    funding_dir.mkdir(parents=True)
    rows = []
    start_ms = 1700000000000
    for i in range(40):
        rows.append(
            {
                "symbol": "BADUSDT",
                "funding_rate": 0.0001,
                "funding_time": start_ms + (i * 8 * 3600 * 1000),
                "mark_price": 0.0 if i == 5 else 100.0 + i,
            }
        )
    pd.DataFrame(rows).to_csv(funding_dir / "BADUSDT.csv", index=False)

    result = build_contrarian_features("BADUSDT", data_dir=tmp_path)

    assert result.empty
    assert "BADUSDT" in load_quarantined_symbols()


def test_xgb_sanitizes_non_finite_targets_before_training():
    idx = pd.date_range("2026-01-01", periods=80, freq="8h", tz="UTC")
    df = pd.DataFrame(
        {
            "feat_a": np.linspace(0.0, 1.0, 80),
            "direction_24h": [0, 1] * 40,
            "price_return_24h_target": [0.01] * 79 + [np.inf],
        },
        index=idx,
    )
    model = ContrarianXGBoost()
    clean = model._sanitize_training_frame(df)

    assert len(clean) == 79
    assert np.isfinite(clean["price_return_24h_target"].values).all()


def test_contrarian_learner_rejects_non_finite_regression_targets():
    learner = ContrarianOnlineLearner()
    df = pd.DataFrame(
        {
            "feat_a": [1.0, 2.0],
            "price_return_24h_target": [0.02, np.inf],
            "direction_24h": [1, 0],
        }
    )
    reject = learner._validate_training_df(df)

    assert reject is not None
    assert reject["reason"] == "non_finite_regression_target"
