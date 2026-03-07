from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import config
from strategy.prediction_policy_gate import train_from_examples


def _write_examples(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_prediction_policy_gate_promotes_positive_model_and_shadows_negative_model(tmp_path, monkeypatch):
    prediction_root = tmp_path / "prediction"
    state_root = prediction_root / "state"
    prediction_root.mkdir(parents=True, exist_ok=True)
    state_root.mkdir(parents=True, exist_ok=True)
    output = tmp_path / "models" / "prediction_policy_gate_v1.json"

    monkeypatch.setattr(config, "PREDICTION_STATE_DIR", str(state_root))
    monkeypatch.setattr(config, "PREDICTION_POLICY_GATE_MIN_EXAMPLES", 150)
    monkeypatch.setattr(config, "PREDICTION_POLICY_GATE_MIN_TEST_EXAMPLES", 40)
    monkeypatch.setattr(config, "PREDICTION_POLICY_GATE_MIN_TEST_BETS", 20)
    monkeypatch.setattr(config, "PREDICTION_POLICY_GATE_TRAIN_FRACTION", 0.7)
    monkeypatch.setattr(config, "PREDICTION_POLICY_GATE_MIN_ROI", 0.0)
    monkeypatch.setattr(config, "PREDICTION_POLICY_GATE_MIN_BRIER_LIFT", 0.0)
    monkeypatch.setattr(config, "PREDICTION_POLICY_GATE_EDGE_THRESHOLDS", "0.01,0.02,0.03")

    positive_state = {
        "model_id": "pure_logit_3",
        "model_kind": "pure_logit",
        "min_edge": 0.01,
    }
    negative_state = {
        "model_id": "residual_logit_2",
        "model_kind": "residual_logit",
        "min_edge": 0.01,
    }
    (state_root / "pure_logit_3.json").write_text(json.dumps(positive_state), encoding="utf-8")
    (state_root / "residual_logit_2.json").write_text(json.dumps(negative_state), encoding="utf-8")

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    positive_rows: list[dict] = []
    negative_rows: list[dict] = []
    for i in range(220):
        ts = (start + timedelta(minutes=i)).isoformat()
        signal = 1.0 if (i % 2 == 0) else -1.0
        positive_rows.append(
            {
                "timestamp": ts,
                "base_prob": 0.40,
                "odds": 3.20,
                "label": 1 if signal > 0 else 0,
                "price_velocity": signal,
                "spread_mean": 0.08,
                "depth_total_eur": 1400.0,
                "time_to_start_sec": 5400.0,
                "in_play": 0.0,
            }
        )
        negative_rows.append(
            {
                "timestamp": ts,
                "base_prob": 0.40,
                "odds": 2.20,
                "label": 1 if (i % 12 == 0) else 0,
                "price_velocity": signal,
                "spread_mean": 0.08,
                "depth_total_eur": 1400.0,
                "time_to_start_sec": 5400.0,
                "in_play": 0.0,
            }
        )

    _write_examples(prediction_root / "online_examples_pure_logit_3.jsonl", positive_rows)
    _write_examples(prediction_root / "online_examples_residual_logit_2.jsonl", negative_rows)

    result = train_from_examples(input_dir=str(prediction_root), output=str(output))
    assert result["ok"] is True

    payload = json.loads(output.read_text(encoding="utf-8"))
    positive = payload["model_policies"]["pure_logit_3"]
    negative = payload["model_policies"]["residual_logit_2"]

    assert positive["mode"] == "execute"
    assert positive["roi"] >= 0.0
    assert positive["brier_lift"] >= 0.0
    assert positive["edge_threshold"] <= 0.02
    assert negative["mode"] == "shadow_only"
