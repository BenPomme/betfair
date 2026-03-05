from strategy.learning_architect import LearningArchitect


class _FakeEngine:
    def __init__(self, stake_fraction: float, min_edge: float):
        self.stake_fraction = stake_fraction
        self.min_edge = min_edge


class _FakeManager:
    def __init__(self):
        self.engines = {
            "m1": _FakeEngine(0.05, 0.03),
        }

    def initial_state(self):
        return {
            "m1": {
                "model_id": "m1",
                "stake_fraction": self.engines["m1"].stake_fraction,
                "min_edge": self.engines["m1"].min_edge,
                "settled_bets": 100,
                "roi_pct": -2.0,
                "avg_brier": 0.4,
            }
        }


def test_architect_rules_apply_bounded_changes(monkeypatch):
    architect = LearningArchitect()
    mgr = _FakeManager()
    decision = architect.evaluate_and_apply(mgr)
    assert decision.mode == "rules"
    assert isinstance(decision.proposals, list)
    assert mgr.engines["m1"].stake_fraction <= 0.05
    assert mgr.engines["m1"].min_edge >= 0.03
