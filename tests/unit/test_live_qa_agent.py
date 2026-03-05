import asyncio

import config
from qa.live_qa_agent import LiveQAAgent


def test_rules_add_restart_when_runtime_degraded(monkeypatch):
    monkeypatch.setattr(config, "QA_RESTART_ON_DEGRADED_ENABLED", True)
    monkeypatch.setattr(config, "QA_DEGRADED_MIN_AGE_SECONDS", 60)
    agent = LiveQAAgent()
    metrics = {
        "candidate": {"stale_rate": 0.1, "fresh_snapshot_rate": 0.9},
        "artifacts": {"scoring_model_exists": True, "fill_model_exists": True},
        "runtime": {
            "running": True,
            "system_ok": False,
            "risk_ok": True,
            "degraded_seconds": 75.0,
        },
    }
    actions = agent._rules_actions(metrics)
    assert any(a.get("type") == "restart_runtime" for a in actions)


def test_rules_do_not_restart_when_risk_guard_failed(monkeypatch):
    monkeypatch.setattr(config, "QA_RESTART_ON_DEGRADED_ENABLED", True)
    monkeypatch.setattr(config, "QA_DEGRADED_MIN_AGE_SECONDS", 60)
    agent = LiveQAAgent()
    metrics = {
        "candidate": {"stale_rate": 0.1, "fresh_snapshot_rate": 0.9},
        "artifacts": {"scoring_model_exists": True, "fill_model_exists": True},
        "runtime": {
            "running": True,
            "system_ok": False,
            "risk_ok": False,
            "degraded_seconds": 75.0,
        },
    }
    actions = agent._rules_actions(metrics)
    assert not any(a.get("type") == "restart_runtime" for a in actions)


def test_sanitize_actions_allows_restart_runtime():
    agent = LiveQAAgent()
    actions = agent._sanitize_actions(
        [
            {"type": "restart_runtime", "reason": "health_degraded"},
            {"type": "unsupported", "reason": "x"},
        ]
    )
    assert actions == [{"type": "restart_runtime", "reason": "health_degraded"}]


def test_apply_restart_action_uses_callback():
    agent = LiveQAAgent()
    reasons = []

    async def _run():
        return await agent._apply_actions(
            [{"type": "restart_runtime", "reason": "health_degraded_while_running"}],
            request_restart=lambda reason: reasons.append(reason) or True,
        )

    results = asyncio.run(_run())
    assert reasons == ["health_degraded_while_running"]
    assert results[0]["ok"] is True
    assert results[0]["requires_restart"] is True
