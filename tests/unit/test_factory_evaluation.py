from __future__ import annotations

from factory.contracts import EvaluationBundle
from factory.evaluation import assign_pareto_ranks, best_bundle_by_lineage, compute_hard_vetoes


def _bundle(
    evaluation_id: str,
    *,
    lineage_id: str = "lineage-a",
    monthly_roi_pct: float = 6.0,
    calibration_lift_abs: float = 0.02,
    capacity_score: float = 0.6,
    regime_robustness: float = 0.7,
    max_drawdown_pct: float = 4.0,
    failure_rate: float = 0.01,
    slippage_headroom_pct: float = 1.0,
    trade_count: int = 80,
) -> EvaluationBundle:
    return EvaluationBundle(
        evaluation_id=evaluation_id,
        lineage_id=lineage_id,
        family_id="family-a",
        stage="paper",
        source="test",
        monthly_roi_pct=monthly_roi_pct,
        max_drawdown_pct=max_drawdown_pct,
        slippage_headroom_pct=slippage_headroom_pct,
        calibration_lift_abs=calibration_lift_abs,
        turnover=0.4,
        capacity_score=capacity_score,
        failure_rate=failure_rate,
        regime_robustness=regime_robustness,
        baseline_beaten_windows=3,
        stress_positive=True,
        trade_count=trade_count,
        settled_count=trade_count,
        paper_days=30,
        net_pnl=monthly_roi_pct,
    )


def test_assign_pareto_ranks_prefers_dominant_bundle():
    dominant = _bundle("eval-dominant")
    dominated = _bundle(
        "eval-dominated",
        lineage_id="lineage-b",
        monthly_roi_pct=5.0,
        calibration_lift_abs=0.01,
        capacity_score=0.4,
        regime_robustness=0.5,
        max_drawdown_pct=6.0,
        failure_rate=0.03,
        slippage_headroom_pct=0.5,
        trade_count=40,
    )

    ranked = assign_pareto_ranks([dominated, dominant])

    assert ranked[0].evaluation_id == "eval-dominant"
    assert ranked[0].pareto_rank == 1
    assert ranked[1].pareto_rank == 2
    assert float(ranked[0].fitness_score or 0.0) > float(ranked[1].fitness_score or 0.0)


def test_compute_hard_vetoes_flags_risk_failures():
    risky = _bundle(
        "eval-risky",
        max_drawdown_pct=14.0,
        failure_rate=0.2,
        slippage_headroom_pct=-0.3,
        capacity_score=0.1,
        regime_robustness=0.2,
    )

    vetoes = compute_hard_vetoes(risky)

    assert "drawdown_above_12pct" in vetoes
    assert "failure_rate_above_15pct" in vetoes
    assert "negative_slippage_stress" in vetoes
    assert "capacity_below_minimum" in vetoes
    assert "regime_robustness_below_minimum" in vetoes


def test_best_bundle_by_lineage_returns_highest_fitness_entry():
    weaker = _bundle("eval-1", lineage_id="lineage-a", monthly_roi_pct=4.0, calibration_lift_abs=0.005)
    stronger = _bundle("eval-2", lineage_id="lineage-a", monthly_roi_pct=7.0, calibration_lift_abs=0.02)
    other = _bundle("eval-3", lineage_id="lineage-b", monthly_roi_pct=6.5)

    selected = best_bundle_by_lineage([weaker, stronger, other])

    assert selected["lineage-a"].evaluation_id == "eval-2"
    assert selected["lineage-b"].evaluation_id == "eval-3"
