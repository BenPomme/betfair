from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import config
from strategy.prediction_bootstrap import bootstrap_challenger_models, pooled_examples


def _write_examples(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_bootstrap_challenger_models_from_real_model_history(tmp_path, monkeypatch):
    prediction_root = tmp_path / "prediction"
    model_root = prediction_root / "models"
    state_root = prediction_root / "state"
    prediction_root.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)
    state_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        config,
        "PREDICTION_MODEL_KINDS",
        "implied_market,residual_logit,pure_logit,market_calibrated,hybrid_logit",
    )

    for model_id, kind in (
        ("implied_market_1", "implied_market"),
        ("residual_logit_2", "residual_logit"),
        ("pure_logit_3", "pure_logit"),
    ):
        (state_root / f"{model_id}.json").write_text(
            json.dumps({"model_id": model_id, "model_kind": kind, "min_edge": 0.03}),
            encoding="utf-8",
        )

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows: list[dict] = []
    for i in range(180):
        rows.append(
            {
                "timestamp": (start + timedelta(minutes=i)).isoformat(),
                "base_prob": 0.42,
                "odds": 2.8,
                "label": 1 if i % 3 == 0 else 0,
                "spread_mean": 0.08,
                "imbalance": 0.01,
                "depth_total_eur": 1200.0,
                "price_velocity": 0.02 if i % 2 == 0 else -0.02,
                "short_volatility": 0.01,
                "time_to_start_sec": 5400.0,
                "in_play": 0.0,
            }
        )

    for model_id in ("implied_market_1", "residual_logit_2", "pure_logit_3"):
        _write_examples(prediction_root / f"online_examples_{model_id}.jsonl", rows)

    monkeypatch.setattr(config, "PREDICTION_STATE_DIR", str(state_root))
    combined = pooled_examples(input_dir=str(prediction_root))
    assert len(combined) == len(rows)

    result = bootstrap_challenger_models(
        input_dir=str(prediction_root),
        model_dir=str(model_root),
        force=True,
        min_examples=100,
    )
    assert result["ok"] is True
    assert (model_root / "market_calibrated_4.json").exists()
    assert (model_root / "hybrid_logit_5.json").exists()
