from __future__ import annotations

import json
from pathlib import Path

import config
from monitoring.research_bridge import load_latest_research_run, publish_research_run


def test_publish_research_run_updates_manifest_and_legacy_prediction_log(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESEARCH_RUNTIME_ROOT", str(tmp_path / "research"), raising=False)
    monkeypatch.setattr(config, "PREDICTION_EXPERIMENT_LOG_PATH", str(tmp_path / "prediction" / "experiments.jsonl"))

    payload = publish_research_run(
        family="betfair_prediction",
        run_id="pred-run-1",
        status="accepted",
        decision="publish",
        portfolios=["betfair_core", "betfair_prediction_league"],
        subject="betfair_prediction",
        git_sha="abc1234",
        metrics={"rolling_200": {"brier_lift_abs": 0.03}},
        gate={"strict_gate_pass": True},
        artifact_manifest={"artifact_path": "data/models/prediction/latest.json"},
    )

    latest = load_latest_research_run("betfair_core")
    log_path = Path(config.PREDICTION_EXPERIMENT_LOG_PATH)
    log_rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert payload["run_id"] == "pred-run-1"
    assert latest["decision"] == "publish"
    assert log_rows[-1]["research"]["family"] == "betfair_prediction"
    assert log_rows[-1]["gate"]["strict_gate_pass"] is True
