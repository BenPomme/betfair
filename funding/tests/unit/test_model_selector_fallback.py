from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

import funding.ml.model_selector as ms
from funding.ml.contrarian_baseline import ContrarianBaselineModel


def test_model_selector_uses_fallback_when_heavy_models_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(ms, "ContrarianXGBoost", None)
    monkeypatch.setattr(ms, "_DEEP_LEARNING_AVAILABLE", False)

    n = 120
    idx = [datetime.now(timezone.utc) - timedelta(hours=8 * (n - i)) for i in range(n)]
    df = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"] * n,
            "funding_zscore_30": [0.1 if i % 2 == 0 else -0.1 for i in range(n)],
            "rate_change_8h": [0.0002 if i % 3 == 0 else -0.0001 for i in range(n)],
            "direction_24h": [1 if i % 2 == 0 else 0 for i in range(n)],
            "price_return_24h_target": [0.01 if i % 2 == 0 else -0.01 for i in range(n)],
        },
        index=pd.DatetimeIndex(idx),
    )

    selector = ms.ModelSelector(model_dir=str(tmp_path))
    comparison = selector.compare(df, n_trials=1, tft_epochs=1)
    assert comparison.get("selected") == "fallback"
    model = selector.get_model()
    assert isinstance(model, ContrarianBaselineModel)

    selector2 = ms.ModelSelector(model_dir=str(tmp_path))
    selector2.load_comparison()
    model2 = selector2.get_model()
    assert isinstance(model2, ContrarianBaselineModel)
