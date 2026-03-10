from __future__ import annotations

from factory.contracts import FactoryFamily, LearningMemoryEntry, MutationBounds, ResearchHypothesis, StrategyGenome
from factory.strategy_inventor import ScientificStrategyInventor


def test_scientific_strategy_inventor_generates_cross_domain_bounded_proposal():
    inventor = ScientificStrategyInventor()
    family = FactoryFamily(
        family_id="betfair_prediction_value_league",
        label="Betfair Prediction/Value League",
        thesis="Evolve probability models with disciplined policy gates.",
        target_portfolios=["betfair_core"],
        target_venues=["betfair"],
        primary_connector_ids=["betfair_core"],
        champion_lineage_id="betfair_prediction_value_league:champion",
        shadow_challenger_ids=[],
        paper_challenger_ids=[],
        budget_split={"incumbent": 70.0, "adjacent": 20.0, "moonshot": 10.0},
        queue_stage="idea",
        explainer="Prediction family",
    )
    hypothesis = ResearchHypothesis(
        hypothesis_id="betfair_prediction_value_league:hypothesis",
        family_id=family.family_id,
        title=family.label,
        thesis=family.thesis,
        scientific_domains=["econometrics", "microstructure"],
        lead_agent_role="Director",
        success_metric="paper_monthly_roi_pct",
        guardrails=["paper-first"],
    )
    genome = StrategyGenome(
        genome_id="betfair_prediction_value_league:genome",
        lineage_id=family.champion_lineage_id,
        family_id=family.family_id,
        parent_genome_id=None,
        role="champion",
        parameters={"selected_model_class": "logit"},
        mutation_bounds=MutationBounds(
            horizons_seconds=[120, 600, 1800],
            feature_subsets=["baseline", "microstructure", "cross_science", "regime"],
            model_classes=["logit", "gbdt", "tft", "transformer", "rules"],
            execution_thresholds={"min_edge": [0.01, 0.1], "stake_fraction": [0.01, 0.1]},
            hyperparameter_ranges={"learning_rate": [0.001, 0.1], "lookback_hours": [6.0, 168.0]},
        ),
        scientific_domains=["econometrics", "microstructure", "information_theory"],
        budget_bucket="incumbent",
        resource_profile="local-first-hybrid",
        budget_weight_pct=16.0,
    )
    proposal = inventor.generate_proposal(
        family=family,
        champion_hypothesis=hypothesis,
        champion_genome=genome,
        learning_memory=[
            LearningMemoryEntry(
                memory_id="memory-a",
                family_id=family.family_id,
                lineage_id="old-lineage",
                hypothesis_id="old-hypothesis",
                outcome="retired_underperformance",
                summary="old swarm failed",
                scientific_domains=["network_epidemiology", "game_theory_behavioral", "information_theory"],
                lead_agent_role="Network/Epidemiology Researcher",
                tweak_count=2,
                decision_stage="paper",
                recommendations=["avoid repeating network-heavy swarm without structural change"],
            )
        ],
        cycle_count=1,
        proposal_index=2,
    )

    assert proposal.origin == "scientific_agent_collective"
    assert len(proposal.scientific_domains) >= 2
    assert proposal.lead_agent_role
    assert proposal.collaborating_agent_roles
    assert proposal.parameter_overrides["selected_feature_subset"] in genome.mutation_bounds.feature_subsets
    assert proposal.parameter_overrides["selected_model_class"] in genome.mutation_bounds.model_classes
    assert proposal.parameter_overrides["selected_horizon_seconds"] in genome.mutation_bounds.horizons_seconds
    assert 0.01 <= proposal.parameter_overrides["selected_min_edge"] <= 0.1
    assert 0.01 <= proposal.parameter_overrides["selected_stake_fraction"] <= 0.1
    assert proposal.parameter_overrides["selected_model_class"] == "gbdt"
    assert proposal.parameter_overrides["selected_min_edge"] >= 0.05
    assert proposal.parameter_overrides["selected_stake_fraction"] <= 0.02
