from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import config
from factory.contracts import (
    EvaluationBundle,
    LineageRecord,
    ManifestStatus,
    PromotionDecision,
    PromotionStage,
)


_STAGE_ORDER = [
    PromotionStage.IDEA.value,
    PromotionStage.SPEC.value,
    PromotionStage.DATA_CHECK.value,
    PromotionStage.GOLDFISH_RUN.value,
    PromotionStage.WALKFORWARD.value,
    PromotionStage.STRESS.value,
    PromotionStage.SHADOW.value,
    PromotionStage.PAPER.value,
    PromotionStage.CANARY_READY.value,
    PromotionStage.LIVE_READY.value,
    PromotionStage.APPROVED_LIVE.value,
]


@dataclass
class PromotionGateConfig:
    monthly_roi_pct: float = float(getattr(config, "FACTORY_PAPER_GATE_MONTHLY_ROI_PCT", 5.0))
    max_drawdown_pct: float = float(getattr(config, "FACTORY_PAPER_GATE_MAX_DRAWDOWN_PCT", 8.0))
    min_paper_days: int = int(getattr(config, "FACTORY_PAPER_GATE_MIN_DAYS", 30))
    min_fast_trades: int = int(getattr(config, "FACTORY_PAPER_GATE_MIN_FAST_TRADES", 50))
    min_slow_settled: int = int(getattr(config, "FACTORY_PAPER_GATE_MIN_SLOW_SETTLED", 10))


class PromotionController:
    def __init__(self, gate_config: Optional[PromotionGateConfig] = None):
        self.gates = gate_config or PromotionGateConfig()

    def paper_gate_blockers(self, bundle: EvaluationBundle, *, slow_strategy: bool) -> List[str]:
        blockers: List[str] = []
        if bundle.baseline_beaten_windows < 3:
            blockers.append("baseline_not_beaten_on_3_windows")
        if bundle.paper_days < self.gates.min_paper_days:
            blockers.append("insufficient_paper_days")
        if bundle.monthly_roi_pct < self.gates.monthly_roi_pct:
            blockers.append("monthly_roi_below_5pct")
        if bundle.max_drawdown_pct > self.gates.max_drawdown_pct:
            blockers.append("drawdown_above_8pct")
        if bundle.slippage_headroom_pct <= 0.0:
            blockers.append("slippage_stress_non_positive")
        if slow_strategy:
            if bundle.settled_count < self.gates.min_slow_settled:
                blockers.append("insufficient_settled_events")
        elif bundle.trade_count < self.gates.min_fast_trades:
            blockers.append("insufficient_trade_count")
        if bundle.hard_vetoes:
            blockers.extend(list(bundle.hard_vetoes))
        return blockers

    def decide(
        self,
        lineage: LineageRecord,
        *,
        data_ready: bool,
        workspace_ready: bool,
        walkforward_bundle: Optional[EvaluationBundle],
        stress_bundle: Optional[EvaluationBundle],
        paper_bundle: Optional[EvaluationBundle],
        manifest_status: Optional[str],
        approved_by: Optional[str],
    ) -> PromotionDecision:
        current_index = _STAGE_ORDER.index(lineage.current_stage)
        target_stage = lineage.current_stage
        blockers: List[str] = []

        if current_index < _STAGE_ORDER.index(PromotionStage.SPEC.value):
            target_stage = PromotionStage.SPEC.value
        if data_ready:
            target_stage = PromotionStage.DATA_CHECK.value
        else:
            blockers.append("connector_data_not_ready")
        if workspace_ready and data_ready:
            target_stage = PromotionStage.GOLDFISH_RUN.value
        else:
            blockers.append("goldfish_workspace_not_ready")
        if walkforward_bundle is not None:
            target_stage = PromotionStage.WALKFORWARD.value
            if walkforward_bundle.baseline_beaten_windows < 3:
                blockers.append("walkforward_window_coverage_insufficient")
        else:
            blockers.append("missing_walkforward_evidence")
        if stress_bundle is not None and stress_bundle.stress_positive:
            target_stage = PromotionStage.STRESS.value
            target_stage = PromotionStage.SHADOW.value
        elif stress_bundle is not None and not stress_bundle.stress_positive:
            blockers.append("stress_eval_negative")
        else:
            blockers.append("missing_stress_evidence")
        if paper_bundle is not None:
            target_stage = PromotionStage.PAPER.value
            slow_strategy = "slow" in lineage.family_id or "polymarket" in lineage.family_id
            paper_blockers = self.paper_gate_blockers(paper_bundle, slow_strategy=slow_strategy)
            if not paper_blockers:
                target_stage = PromotionStage.CANARY_READY.value
                target_stage = PromotionStage.LIVE_READY.value
            else:
                blockers.extend(paper_blockers)
        else:
            blockers.append("missing_paper_evidence")
        requires_human_signoff = target_stage in {
            PromotionStage.CANARY_READY.value,
            PromotionStage.LIVE_READY.value,
            PromotionStage.APPROVED_LIVE.value,
        }
        if manifest_status == ManifestStatus.APPROVED_LIVE.value and approved_by:
            target_stage = PromotionStage.APPROVED_LIVE.value
            blockers = []
            requires_human_signoff = False
        elif target_stage == PromotionStage.LIVE_READY.value:
            blockers.append("human_signoff_required")
        unique_blockers = list(dict.fromkeys(blockers))
        return PromotionDecision(
            lineage_id=lineage.lineage_id,
            current_stage=lineage.current_stage,
            next_stage=target_stage,
            allowed=target_stage != lineage.current_stage,
            requires_human_signoff=requires_human_signoff,
            blockers=unique_blockers,
            reasons=[f"target_stage={target_stage}"],
        )
