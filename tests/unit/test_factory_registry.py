from __future__ import annotations

from factory.contracts import (
    AcceptedStrategyManifest,
    EvaluationBundle,
    EvaluationWindow,
    ExperimentSpec,
    LearningMemoryEntry,
    LineageRecord,
    MutationBounds,
    ResearchHypothesis,
    StrategyGenome,
)
from factory.registry import FactoryRegistry


def test_registry_round_trips_nested_factory_records(tmp_path):
    registry = FactoryRegistry(tmp_path / "factory")
    hypothesis = ResearchHypothesis(
        hypothesis_id="hypothesis-a",
        family_id="family-a",
        title="Hypothesis A",
        thesis="Test thesis",
        scientific_domains=["econometrics"],
        lead_agent_role="Director",
        success_metric="paper_monthly_roi_pct",
        guardrails=["paper_only"],
    )
    genome = StrategyGenome(
        genome_id="genome-a",
        lineage_id="lineage-a",
        family_id="family-a",
        parent_genome_id=None,
        role="champion",
        parameters={"feature_set": "baseline"},
        mutation_bounds=MutationBounds(
            horizons_seconds=[120, 600],
            feature_subsets=["baseline", "microstructure"],
            model_classes=["logit", "gbdt"],
            execution_thresholds={"min_edge": [0.01, 0.08]},
            hyperparameter_ranges={"learning_rate": [0.001, 0.05]},
        ),
        scientific_domains=["econometrics"],
        budget_bucket="incumbent",
        resource_profile="local-first-hybrid",
        budget_weight_pct=20.0,
    )
    experiment = ExperimentSpec(
        experiment_id="experiment-a",
        lineage_id="lineage-a",
        family_id="family-a",
        hypothesis_id="hypothesis-a",
        genome_id="genome-a",
        goldfish_workspace="research/goldfish/family-a",
        pipeline_stages=["dataset", "train", "package"],
        backend_mode="goldfish_sidecar",
        resource_profile="local-first-hybrid",
    )
    lineage = LineageRecord(
        lineage_id="lineage-a",
        family_id="family-a",
        label="Champion",
        role="champion",
        current_stage="idea",
        target_portfolios=["betfair_core"],
        target_venues=["betfair"],
        hypothesis_id="hypothesis-a",
        genome_id="genome-a",
        experiment_id="experiment-a",
        budget_bucket="incumbent",
        budget_weight_pct=20.0,
        connector_ids=["betfair_core"],
        goldfish_workspace="research/goldfish/family-a",
    )

    registry.save_research_pack(
        hypothesis=hypothesis,
        genome=genome,
        experiment=experiment,
        lineage=lineage,
    )
    registry.save_evaluation(
        EvaluationBundle(
            evaluation_id="eval-a",
            lineage_id="lineage-a",
            family_id="family-a",
            stage="paper",
            source="test",
            windows=[
                EvaluationWindow(
                    label="window-a",
                    settled_count=60,
                    monthly_roi_pct=6.0,
                    baseline_roi_pct=0.0,
                    brier_lift_abs=0.02,
                    drawdown_pct=3.0,
                    slippage_headroom_pct=1.2,
                    failure_rate=0.01,
                    regime_robustness=0.7,
                )
            ],
            monthly_roi_pct=6.0,
            max_drawdown_pct=3.0,
            slippage_headroom_pct=1.2,
            calibration_lift_abs=0.02,
            turnover=0.5,
            capacity_score=0.6,
            failure_rate=0.01,
            regime_robustness=0.7,
            baseline_beaten_windows=3,
            stress_positive=True,
            trade_count=60,
            settled_count=60,
            paper_days=30,
            net_pnl=6.0,
        )
    )

    loaded_genome = registry.load_genome("lineage-a")
    loaded_experiment = registry.load_experiment("lineage-a")
    loaded_bundles = registry.evaluations("lineage-a")

    assert loaded_genome is not None
    assert loaded_experiment is not None
    assert isinstance(loaded_genome.mutation_bounds, MutationBounds)
    assert loaded_genome.mutation_bounds.horizons_seconds == [120, 600]
    assert isinstance(loaded_bundles[0].windows[0], EvaluationWindow)
    assert loaded_experiment.pipeline_stages == ["dataset", "train", "package"]


def test_registry_manifest_approval_controls_live_exposure(tmp_path):
    registry = FactoryRegistry(tmp_path / "factory")
    manifest = AcceptedStrategyManifest(
        manifest_id="manifest-a",
        lineage_id="lineage-a",
        family_id="family-a",
        portfolio_targets=["betfair_core"],
        venue_targets=["betfair"],
        approved_stage="live_ready",
        status="pending_approval",
        artifact_refs={"workspace": "research/goldfish/family-a"},
        runtime_overrides={"resource_profile": "local-first-hybrid"},
    )

    registry.save_manifest(manifest)

    assert registry.live_manifests_for_portfolio("betfair_core") == []

    approved = registry.approve_manifest("manifest-a", approved_by="operator", note="ready for canary")

    assert approved is not None
    assert approved.is_live_loadable() is True
    assert len(registry.live_manifests_for_portfolio("betfair_core")) == 1


def test_registry_persists_learning_memory_entries(tmp_path):
    registry = FactoryRegistry(tmp_path / "factory")
    memory = LearningMemoryEntry(
        memory_id="memory-a",
        family_id="family-a",
        lineage_id="lineage-a",
        hypothesis_id="hypothesis-a",
        outcome="retired_underperformance",
        summary="Retired after two tweaks",
        scientific_domains=["econometrics", "microstructure"],
        lead_agent_role="Econometrics Researcher",
        tweak_count=2,
        decision_stage="paper",
        recommendations=["tighten thresholds"],
    )

    registry.save_learning_memory(memory)
    memories = registry.learning_memories(family_id="family-a")

    assert len(memories) == 1
    assert memories[0].memory_id == "memory-a"
    assert memories[0].recommendations == ["tighten thresholds"]
