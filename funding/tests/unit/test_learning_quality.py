from __future__ import annotations

from pathlib import Path

import config
from funding.ml.learning_quality import FundingLearningQuality


def _mk(tmp_path: Path) -> FundingLearningQuality:
    return FundingLearningQuality(
        model_id="t1",
        model_family="funding",
        state_path=str(tmp_path / "state.json"),
    )


def test_strict_gate_pass_fail_matrix(tmp_path):
    q = _mk(tmp_path)
    for _ in range(200):
        q.record_settlement("BTCUSDT", stake=1.0, pnl=0.05, label=1, pred_prob=0.8, base_prob=0.6)
    gp, _, _, r100, r200 = q.strict_gate_status()
    assert gp is True
    assert r100["brier_lift_abs"] > 0
    assert r200["roi_pct"] >= 0

    q2 = _mk(tmp_path)
    for _ in range(200):
        q2.record_settlement("BTCUSDT", stake=1.0, pnl=-0.02, label=1, pred_prob=0.8, base_prob=0.6)
    gp2, reason2, _, _, _ = q2.strict_gate_status()
    assert gp2 is False
    assert reason2 == "negative_roi"

    q3 = _mk(tmp_path)
    for _ in range(200):
        q3.record_settlement("BTCUSDT", stake=1.0, pnl=0.01, label=1, pred_prob=0.4, base_prob=0.6)
    gp3, reason3, _, _, _ = q3.strict_gate_status()
    assert gp3 is False
    assert reason3 == "negative_brier_lift"


def test_rejects_non_finite_and_out_of_range(tmp_path):
    q = _mk(tmp_path)
    rej1 = q.validate_features({"a": float("nan")}, "BTCUSDT", "test")
    assert rej1 is not None
    assert rej1["reason"] == "non_finite_feature"
    rej2 = q.validate_features({"a": float(config.FUNDING_FEATURE_ABS_MAX) * 2}, "BTCUSDT", "test")
    assert rej2 is not None
    assert rej2["reason"] == "feature_abs_limit"


def test_frozen_and_saturation_detectors(tmp_path):
    q = _mk(tmp_path)
    for _ in range(int(config.FUNDING_FROZEN_WINDOW)):
        q.add_prediction(0.5)
    assert q.prediction_is_frozen() is True

    q2 = _mk(tmp_path)
    for _ in range(int(config.FUNDING_SATURATION_WINDOW)):
        q2.add_prediction(0.999)
    assert q2.saturation_rate() >= float(config.FUNDING_SATURATION_RATE_THRESHOLD)
