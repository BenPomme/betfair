from __future__ import annotations

from dataclasses import replace
from typing import Dict, Iterable, List

from factory.contracts import EvaluationBundle


def compute_hard_vetoes(bundle: EvaluationBundle) -> List[str]:
    vetoes: List[str] = []
    if bundle.failure_rate > 0.15:
        vetoes.append("failure_rate_above_15pct")
    if bundle.max_drawdown_pct > 12.0:
        vetoes.append("drawdown_above_12pct")
    if bundle.slippage_headroom_pct <= 0.0:
        vetoes.append("negative_slippage_stress")
    if bundle.capacity_score < 0.20:
        vetoes.append("capacity_below_minimum")
    if bundle.regime_robustness < 0.30:
        vetoes.append("regime_robustness_below_minimum")
    return vetoes


def compute_fitness_score(bundle: EvaluationBundle) -> float:
    score = 0.0
    score += bundle.monthly_roi_pct * 3.0
    score += bundle.calibration_lift_abs * 200.0
    score += bundle.capacity_score * 20.0
    score += bundle.regime_robustness * 25.0
    score -= bundle.max_drawdown_pct * 1.5
    score -= bundle.failure_rate * 100.0
    score += max(0.0, bundle.slippage_headroom_pct) * 2.0
    score += min(bundle.trade_count, 100) * 0.05
    score -= len(bundle.hard_vetoes) * 25.0
    return round(score, 6)


def dominates(left: EvaluationBundle, right: EvaluationBundle) -> bool:
    better_or_equal = (
        left.monthly_roi_pct >= right.monthly_roi_pct
        and left.calibration_lift_abs >= right.calibration_lift_abs
        and left.capacity_score >= right.capacity_score
        and left.regime_robustness >= right.regime_robustness
        and left.max_drawdown_pct <= right.max_drawdown_pct
        and left.failure_rate <= right.failure_rate
        and left.slippage_headroom_pct >= right.slippage_headroom_pct
    )
    strictly_better = (
        left.monthly_roi_pct > right.monthly_roi_pct
        or left.calibration_lift_abs > right.calibration_lift_abs
        or left.capacity_score > right.capacity_score
        or left.regime_robustness > right.regime_robustness
        or left.max_drawdown_pct < right.max_drawdown_pct
        or left.failure_rate < right.failure_rate
        or left.slippage_headroom_pct > right.slippage_headroom_pct
    )
    return better_or_equal and strictly_better


def assign_pareto_ranks(bundles: Iterable[EvaluationBundle]) -> List[EvaluationBundle]:
    pending = [replace(bundle) for bundle in bundles]
    output: List[EvaluationBundle] = []
    rank = 1
    while pending:
        current_front: List[EvaluationBundle] = []
        for bundle in pending:
            dominated = any(
                dominates(other, bundle)
                for other in pending
                if other.evaluation_id != bundle.evaluation_id
            )
            if not dominated:
                current_front.append(bundle)
        current_ids = {bundle.evaluation_id for bundle in current_front}
        for bundle in current_front:
            bundle.pareto_rank = rank
            bundle.hard_vetoes = compute_hard_vetoes(bundle)
            bundle.fitness_score = compute_fitness_score(bundle)
            output.append(bundle)
        pending = [bundle for bundle in pending if bundle.evaluation_id not in current_ids]
        rank += 1
    return sorted(
        output,
        key=lambda item: (
            item.pareto_rank or 999,
            -float(item.fitness_score or 0.0),
            -item.monthly_roi_pct,
        ),
    )


def best_bundle_by_lineage(bundles: Iterable[EvaluationBundle]) -> Dict[str, EvaluationBundle]:
    ranked = assign_pareto_ranks(bundles)
    by_lineage: Dict[str, EvaluationBundle] = {}
    for bundle in ranked:
        previous = by_lineage.get(bundle.lineage_id)
        if previous is None or float(bundle.fitness_score or 0.0) > float(previous.fitness_score or 0.0):
            by_lineage[bundle.lineage_id] = bundle
    return by_lineage
