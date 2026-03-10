from __future__ import annotations

from factory.contracts import EvaluationBundle, LineageRecord, PromotionStage
from factory.promotion import PromotionController, PromotionGateConfig


def _lineage() -> LineageRecord:
    return LineageRecord(
        lineage_id="lineage-a",
        family_id="binance_funding_contrarian",
        label="Funding Champion",
        role="champion",
        current_stage=PromotionStage.IDEA.value,
        target_portfolios=["contrarian_legacy"],
        target_venues=["binance"],
        hypothesis_id="hypothesis-a",
        genome_id="genome-a",
        experiment_id="experiment-a",
        budget_bucket="incumbent",
        budget_weight_pct=20.0,
        connector_ids=["binance_core"],
        goldfish_workspace="research/goldfish/binance_funding_contrarian",
    )


def _bundle(stage: str = "paper", *, paper_days: int = 30, trade_count: int = 60, settled_count: int = 60) -> EvaluationBundle:
    return EvaluationBundle(
        evaluation_id=f"eval-{stage}",
        lineage_id="lineage-a",
        family_id="binance_funding_contrarian",
        stage=stage,
        source="test",
        monthly_roi_pct=6.0,
        max_drawdown_pct=3.0,
        slippage_headroom_pct=1.0,
        calibration_lift_abs=0.02,
        turnover=0.4,
        capacity_score=0.6,
        failure_rate=0.01,
        regime_robustness=0.7,
        baseline_beaten_windows=3,
        stress_positive=True,
        trade_count=trade_count,
        settled_count=settled_count,
        paper_days=paper_days,
        net_pnl=6.0,
    )


def test_paper_gate_blockers_require_sufficient_days_and_evidence():
    controller = PromotionController(
        PromotionGateConfig(
            monthly_roi_pct=5.0,
            max_drawdown_pct=8.0,
            min_paper_days=30,
            min_fast_trades=50,
            min_slow_settled=10,
        )
    )

    blockers = controller.paper_gate_blockers(
        _bundle(paper_days=12, trade_count=25, settled_count=25),
        slow_strategy=False,
    )

    assert "insufficient_paper_days" in blockers
    assert "insufficient_trade_count" in blockers


def test_decide_requires_human_signoff_before_live():
    controller = PromotionController()
    lineage = _lineage()

    decision = controller.decide(
        lineage,
        data_ready=True,
        workspace_ready=True,
        walkforward_bundle=_bundle("walkforward"),
        stress_bundle=_bundle("stress"),
        paper_bundle=_bundle("paper"),
        manifest_status="pending_approval",
        approved_by=None,
    )

    assert decision.next_stage == PromotionStage.LIVE_READY.value
    assert decision.requires_human_signoff is True
    assert "human_signoff_required" in decision.blockers


def test_decide_marks_lineage_approved_when_manifest_is_approved():
    controller = PromotionController()
    lineage = _lineage()

    decision = controller.decide(
        lineage,
        data_ready=True,
        workspace_ready=True,
        walkforward_bundle=_bundle("walkforward"),
        stress_bundle=_bundle("stress"),
        paper_bundle=_bundle("paper"),
        manifest_status="approved_live",
        approved_by="operator",
    )

    assert decision.next_stage == PromotionStage.APPROVED_LIVE.value
    assert decision.requires_human_signoff is False
    assert decision.blockers == []
