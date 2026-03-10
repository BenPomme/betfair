from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from dataclasses import replace
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import config
from factory.connectors import default_connector_catalog
from factory.contracts import (
    EvaluationBundle,
    EvaluationStage,
    ExperimentQueueEntry,
    EvaluationWindow,
    ExecutionTier,
    ExperimentSpec,
    FactoryFamily,
    FactoryJournal,
    LearningMemoryEntry,
    LineageRecord,
    LineageRole,
    MutationBounds,
    PromotionStage,
    ResearchHypothesis,
    StrategyGenome,
    utc_now_iso,
)
from factory.execution_targets import resolve_target_portfolio
from factory.evaluation import assign_pareto_ranks, compute_hard_vetoes
from factory.experiment_runner import FactoryExperimentRunner
from factory.goldfish_bridge import GoldfishBridge
from factory.promotion import PromotionController
from factory.registry import FactoryRegistry
from factory.runtime_mode import current_agentic_factory_runtime_mode
from factory.strategy_inventor import ScientificAgentProposal, ScientificStrategyInventor
from portfolio.state_store import PortfolioStateStore


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _budget_split() -> Dict[str, float]:
    return {"incumbent": 70.0, "adjacent": 20.0, "moonshot": 10.0}


def _factory_roles() -> Dict[str, List[str]]:
    return {
        ExecutionTier.TIER0.value: [
            "Director",
            "Budget Allocator",
            "Venue/Data Curator",
            "Genome Mutator",
            "Evaluator",
            "Risk Governor",
            "Promotion Arbiter",
            "Goldfish Bridge",
        ],
        ExecutionTier.TIER1.value: [
            "hypothesis_author",
            "feature_ideator",
            "test_scaffold",
            "doc_scribe",
        ],
        ExecutionTier.TIER2.value: [
            "pipeline_assembler",
            "genome_mutation_runner",
            "evaluation_integrator",
        ],
        ExecutionTier.TIER3.value: [
            "capital_risk_reviewer",
            "promotion_policy_reviewer",
            "execution_path_reviewer",
        ],
    }


def _scientific_researchers() -> List[str]:
    return [
        "econometrics",
        "microstructure",
        "bayesian_causal",
        "statistical_physics",
        "network_epidemiology",
        "ecology_evolution",
        "information_theory",
        "control_rl",
        "game_theory_behavioral",
        "signal_processing_neuroscience",
    ]


class FactoryOrchestrator:
    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root)
        factory_root = Path(getattr(config, "FACTORY_ROOT", "data/factory"))
        goldfish_root = Path(getattr(config, "FACTORY_GOLDFISH_ROOT", "research/goldfish"))
        if not factory_root.is_absolute():
            factory_root = self.project_root / factory_root
        if not goldfish_root.is_absolute():
            goldfish_root = self.project_root / goldfish_root
        self.registry = FactoryRegistry(factory_root)
        self.bridge = GoldfishBridge(goldfish_root)
        self.experiment_runner = FactoryExperimentRunner(self.project_root)
        self.strategy_inventor = ScientificStrategyInventor()
        self.promotion = PromotionController()
        self.connectors = default_connector_catalog(self.project_root)
        self._events: Deque[Dict[str, Any]] = deque(maxlen=200)
        self._cycle_count = 0
        self._last_state: Dict[str, Any] = {}
        self.bootstrap()

    def _runtime_mode(self):
        return current_agentic_factory_runtime_mode()

    def _with_runtime_mode(self, state: Dict[str, Any], *, pause_reason: str | None = None) -> Dict[str, Any]:
        runtime_mode = self._runtime_mode()
        state.update(runtime_mode.to_dict())
        if pause_reason:
            state["pause_reason"] = pause_reason
        return state

    def _latest_manifest_by_lineage(self) -> Dict[str, Any]:
        latest: Dict[str, Any] = {}
        for manifest in self.registry.manifests():
            previous = latest.get(manifest.lineage_id)
            if previous is None or str(manifest.created_at) > str(previous.created_at):
                latest[manifest.lineage_id] = manifest
        return latest

    def _hard_stop_state(self) -> Dict[str, Any]:
        state = dict(self.registry.read_state() or {})
        state.setdefault("portfolio_id", getattr(config, "RESEARCH_FACTORY_PORTFOLIO_ID", "research_factory"))
        state.setdefault("mode", "research")
        state.setdefault(
            "explainer",
            "Research-only control plane for multi-family strategy discovery, evaluation, and approval-gated promotion.",
        )
        state["running"] = False
        state["status"] = "paused"
        readiness = dict(state.get("readiness") or {})
        checks = list(readiness.get("checks") or [])
        if not any(item.get("name") == "agentic_factory_runtime_mode" for item in checks):
            checks.append(
                {
                    "name": "agentic_factory_runtime_mode",
                    "ok": False,
                    "reason": "Runtime mode is hard_stop, so factory orchestration and runner influence are paused intentionally.",
                }
            )
        readiness["status"] = "research_only"
        readiness["blockers"] = list(dict.fromkeys(list(readiness.get("blockers") or []) + ["agentic_factory_hard_stopped"]))
        readiness["checks"] = checks
        readiness["eta_to_readiness"] = "hard_stop"
        readiness["score_pct"] = round(
            (sum(1 for item in checks if item.get("ok")) / len(checks)) * 100.0,
            2,
        ) if checks else 0.0
        state["readiness"] = readiness
        research_summary = dict(state.get("research_summary") or {})
        research_summary["hard_stop_active"] = True
        state["research_summary"] = research_summary
        self._last_state = self._with_runtime_mode(
            state,
            pause_reason="agentic_factory_hard_stopped",
        )
        return dict(self._last_state)

    def bootstrap(self) -> None:
        if self.registry.families():
            return
        family_specs = [
            {
                "family_id": "binance_funding_contrarian",
                "label": "Binance Funding Contrarian",
                "thesis": "Exploit funding extremes, regime shifts, and cross-science features to improve directional funding trades.",
                "target_portfolios": ["hedge_validation", "hedge_research", "contrarian_legacy"],
                "target_venues": ["binance"],
                "connectors": ["binance_core"],
                "budget_bucket": "incumbent",
                "budget_weight_pct": 14.0,
                "role": LineageRole.CHAMPION.value,
                "explainer": "Uses funding, basis, open interest, and regime features to rank contrarian directional setups.",
            },
            {
                "family_id": "binance_cascade_regime",
                "label": "Binance Cascade/Regime",
                "thesis": "Detect fragile market states and liquidation cascades early enough to survive and exploit dislocations.",
                "target_portfolios": ["cascade_alpha"],
                "target_venues": ["binance"],
                "connectors": ["binance_core"],
                "budget_bucket": "adjacent",
                "budget_weight_pct": 18.0,
                "role": LineageRole.CHAMPION.value,
                "explainer": "Uses liquidation, depth collapse, and regime features for short-horizon cascade alpha.",
            },
            {
                "family_id": "betfair_prediction_value_league",
                "label": "Betfair Prediction/Value League",
                "thesis": "Evolve parallel probability models and policy gates until the best value-betting league earns execution rights.",
                "target_portfolios": ["betfair_core"],
                "target_venues": ["betfair"],
                "connectors": ["betfair_core"],
                "budget_bucket": "incumbent",
                "budget_weight_pct": 16.0,
                "role": LineageRole.CHAMPION.value,
                "explainer": "Compares prediction, calibration, and policy-gated value-betting lineages on the same snapshots.",
            },
            {
                "family_id": "betfair_information_lag",
                "label": "Betfair Information-Lag Books",
                "thesis": "Cross external event signals, time-zone maintenance patterns, and book synchronization delays to find stale pricing.",
                "target_portfolios": ["betfair_execution_book", "betfair_suspension_lag", "betfair_crossbook_consensus", "betfair_timezone_decay"],
                "target_venues": ["betfair", "polymarket"],
                "connectors": ["betfair_core", "polymarket_core"],
                "budget_bucket": "moonshot",
                "budget_weight_pct": 22.0,
                "role": LineageRole.CHAMPION.value,
                "explainer": "Tracks related-market lag and cross-book stale pricing patterns before they are trusted for execution.",
            },
            {
                "family_id": "polymarket_cross_venue",
                "label": "Polymarket Cross-Venue Signals",
                "thesis": "Use Polymarket microstructure and cross-venue event matching to produce robust paper-only signal layers.",
                "target_portfolios": ["polymarket_quantum_fold", "polymarket_binary_research"],
                "target_venues": ["polymarket", "betfair"],
                "connectors": ["polymarket_core", "betfair_core"],
                "budget_bucket": "adjacent",
                "budget_weight_pct": 30.0,
                "role": LineageRole.CHAMPION.value,
                "explainer": "Runs signal leagues on Polymarket quotes and cross-venue confirmations to rank paper-only opportunities.",
            },
        ]
        for spec in family_specs:
            family_id = spec["family_id"]
            hypothesis_id = f"{family_id}:hypothesis"
            lineage_id = f"{family_id}:champion"
            genome_id = f"{family_id}:genome:champion"
            experiment_id = f"{family_id}:experiment:champion"
            hypothesis = ResearchHypothesis(
                hypothesis_id=hypothesis_id,
                family_id=family_id,
                title=spec["label"],
                thesis=spec["thesis"],
                scientific_domains=_scientific_researchers()[:4],
                lead_agent_role="Director",
                success_metric="paper_monthly_roi_pct",
                guardrails=[
                    "No live promotion without human approval.",
                    "Mutation bounds may not touch credentials or hard risk caps.",
                    "Paper-first and net-of-costs only.",
                ],
                origin="seeded_family",
                agent_notes=["Initial seeded champion for family bootstrap."],
            )
            genome = StrategyGenome(
                genome_id=genome_id,
                lineage_id=lineage_id,
                family_id=family_id,
                parent_genome_id=None,
                role=spec["role"],
                parameters={
                    "resource_profile": "local-first-hybrid",
                    "budget_mix": _budget_split(),
                    "max_shadow_challengers": 5,
                    "max_paper_challengers": 2,
                },
                mutation_bounds=MutationBounds(
                    horizons_seconds=[120, 600, 1800, 14400],
                    feature_subsets=["baseline", "microstructure", "cross_science", "regime"],
                    model_classes=["logit", "gbdt", "tft", "transformer", "rules"],
                    execution_thresholds={"min_edge": [0.01, 0.10], "stake_fraction": [0.01, 0.10]},
                    hyperparameter_ranges={"learning_rate": [0.001, 0.1], "lookback_hours": [6, 168]},
                ),
                scientific_domains=_scientific_researchers(),
                budget_bucket=spec["budget_bucket"],
                resource_profile="local-first-hybrid",
                budget_weight_pct=spec["budget_weight_pct"],
            )
            experiment = ExperimentSpec(
                experiment_id=experiment_id,
                lineage_id=lineage_id,
                family_id=family_id,
                hypothesis_id=hypothesis_id,
                genome_id=genome_id,
                goldfish_workspace=str(self.bridge.workspace_path(family_id)),
                pipeline_stages=["dataset", "features", "train", "walkforward", "stress", "package"],
                backend_mode="goldfish_sidecar",
                resource_profile="local-first-hybrid",
            )
            lineage = LineageRecord(
                lineage_id=lineage_id,
                family_id=family_id,
                label=f"{spec['label']} Champion",
                role=spec["role"],
                current_stage=PromotionStage.IDEA.value,
                target_portfolios=list(spec["target_portfolios"]),
                target_venues=list(spec["target_venues"]),
                hypothesis_id=hypothesis_id,
                genome_id=genome_id,
                experiment_id=experiment_id,
                budget_bucket=spec["budget_bucket"],
                budget_weight_pct=spec["budget_weight_pct"],
                connector_ids=list(spec["connectors"]),
                goldfish_workspace=str(self.bridge.workspace_path(family_id)),
                iteration_status="seeded_champion",
            )
            family = FactoryFamily(
                family_id=family_id,
                label=spec["label"],
                thesis=spec["thesis"],
                target_portfolios=list(spec["target_portfolios"]),
                target_venues=list(spec["target_venues"]),
                primary_connector_ids=list(spec["connectors"]),
                champion_lineage_id=lineage_id,
                shadow_challenger_ids=[],
                paper_challenger_ids=[],
                budget_split=_budget_split(),
                queue_stage=PromotionStage.IDEA.value,
                explainer=spec["explainer"],
            )
            self.registry.save_family(family)
            self.registry.save_research_pack(
                hypothesis=hypothesis,
                genome=genome,
                experiment=experiment,
                lineage=lineage,
            )
        self.registry.write_journal(
            FactoryJournal(
                active_goal="Build a reproducible strategy factory with shadow, paper, and approval-gated live promotion.",
                recent_actions=["[bootstrap] Seeded default factory families and champion lineages."],
            )
        )

    def _lineages_by_family(self) -> Dict[str, List[LineageRecord]]:
        grouped: Dict[str, List[LineageRecord]] = defaultdict(list)
        for lineage in self.registry.lineages():
            grouped[lineage.family_id].append(lineage)
        return grouped

    def _mutation_choice(self, options: List[Any], mutation_index: int, offset: int = 0) -> Any:
        if not options:
            return None
        return options[(mutation_index + offset) % len(options)]

    def _mutation_range_value(self, bounds: List[float], mutation_index: int, *, divisor: int = 5) -> float:
        if not bounds:
            return 0.0
        if len(bounds) == 1:
            return float(bounds[0])
        low = float(bounds[0])
        high = float(bounds[-1])
        slots = max(1, divisor - 1)
        position = (mutation_index % divisor) / slots
        return round(low + ((high - low) * position), 6)

    def _preferred_horizon(self, family_id: str) -> int:
        mapping = {
            "binance_funding_contrarian": 600,
            "binance_cascade_regime": 120,
            "betfair_prediction_value_league": 1800,
            "betfair_information_lag": 600,
            "polymarket_cross_venue": 600,
        }
        return mapping.get(family_id, 600)

    def _family_budget_weight(self, family_id: str) -> float:
        family = self.registry.load_family(family_id)
        if family is None:
            return 5.0
        champion = self.registry.load_lineage(family.champion_lineage_id)
        if champion is None:
            return 5.0
        return float(champion.budget_weight_pct or 5.0)

    def _active_lineages(self, family_id: str) -> List[LineageRecord]:
        return [
            lineage
            for lineage in self.registry.lineages()
            if lineage.family_id == family_id and lineage.active
        ]

    def _nearest_choice(self, options: List[int], value: int) -> int:
        if not options:
            return int(value)
        normalized = [int(option) for option in options]
        return min(normalized, key=lambda option: abs(option - int(value)))

    def _allowed_choice(self, options: List[str], value: str, *, fallback: str) -> str:
        normalized = [str(option) for option in options]
        if str(value) in normalized:
            return str(value)
        if fallback in normalized:
            return fallback
        return normalized[0] if normalized else fallback

    def _rotate_choice(self, options: List[str], current: str, *, step: int = 1, fallback: str) -> str:
        normalized = [str(option) for option in options]
        if not normalized:
            return fallback
        if current not in normalized:
            return normalized[0]
        index = normalized.index(current)
        return normalized[(index + step) % len(normalized)]

    def _clip_bounds(self, bounds: List[float], value: float, *, fallback: float) -> float:
        if not bounds:
            return round(float(value if value else fallback), 6)
        low = float(bounds[0])
        high = float(bounds[-1])
        return round(min(max(float(value), low), high), 6)

    def _mutate_genome(
        self,
        *,
        parent_genome: StrategyGenome,
        lineage_id: str,
        role: str,
        mutation_index: int,
        budget_bucket: str,
        budget_weight_pct: float,
        scientific_domains: Optional[List[str]] = None,
        parameter_overrides: Optional[Dict[str, Any]] = None,
        mutation_source: str = "deterministic_factory_mutation",
    ) -> StrategyGenome:
        bounds = parent_genome.mutation_bounds
        selected_horizon = int(self._mutation_choice(bounds.horizons_seconds, mutation_index) or self._preferred_horizon(parent_genome.family_id))
        selected_feature_subset = str(self._mutation_choice(bounds.feature_subsets, mutation_index, offset=1) or "baseline")
        selected_model_class = str(self._mutation_choice(bounds.model_classes, mutation_index, offset=2) or "logit")
        selected_min_edge = self._mutation_range_value(bounds.execution_thresholds.get("min_edge") or [0.01, 0.1], mutation_index)
        selected_stake_fraction = self._mutation_range_value(
            bounds.execution_thresholds.get("stake_fraction") or [0.01, 0.1],
            mutation_index + 1,
        )
        selected_learning_rate = self._mutation_range_value(
            bounds.hyperparameter_ranges.get("learning_rate") or [0.001, 0.1],
            mutation_index + 2,
        )
        selected_lookback_hours = self._mutation_range_value(
            bounds.hyperparameter_ranges.get("lookback_hours") or [6.0, 168.0],
            mutation_index + 3,
            divisor=7,
        )
        parameters = dict(parent_genome.parameters)
        parameters.update(
            {
                "mutation_index": mutation_index,
                "mutation_source": mutation_source,
                "selected_horizon_seconds": selected_horizon,
                "selected_feature_subset": selected_feature_subset,
                "selected_model_class": selected_model_class,
                "selected_min_edge": selected_min_edge,
                "selected_stake_fraction": selected_stake_fraction,
                "selected_learning_rate": selected_learning_rate,
                "selected_lookback_hours": selected_lookback_hours,
            }
        )
        for key, value in dict(parameter_overrides or {}).items():
            parameters[key] = value
        parameters["selected_horizon_seconds"] = self._nearest_choice(
            bounds.horizons_seconds,
            int(parameters.get("selected_horizon_seconds", selected_horizon) or selected_horizon),
        )
        parameters["selected_feature_subset"] = self._allowed_choice(
            bounds.feature_subsets,
            str(parameters.get("selected_feature_subset", selected_feature_subset) or selected_feature_subset),
            fallback="baseline",
        )
        parameters["selected_model_class"] = self._allowed_choice(
            bounds.model_classes,
            str(parameters.get("selected_model_class", selected_model_class) or selected_model_class),
            fallback="logit",
        )
        parameters["selected_min_edge"] = self._clip_bounds(
            bounds.execution_thresholds.get("min_edge") or [0.01, 0.1],
            float(parameters.get("selected_min_edge", selected_min_edge) or selected_min_edge),
            fallback=0.03,
        )
        parameters["selected_stake_fraction"] = self._clip_bounds(
            bounds.execution_thresholds.get("stake_fraction") or [0.01, 0.1],
            float(parameters.get("selected_stake_fraction", selected_stake_fraction) or selected_stake_fraction),
            fallback=0.03,
        )
        parameters["selected_learning_rate"] = self._clip_bounds(
            bounds.hyperparameter_ranges.get("learning_rate") or [0.001, 0.1],
            float(parameters.get("selected_learning_rate", selected_learning_rate) or selected_learning_rate),
            fallback=0.02,
        )
        parameters["selected_lookback_hours"] = self._clip_bounds(
            bounds.hyperparameter_ranges.get("lookback_hours") or [6.0, 168.0],
            float(parameters.get("selected_lookback_hours", selected_lookback_hours) or selected_lookback_hours),
            fallback=48.0,
        )
        return StrategyGenome(
            genome_id=f"{lineage_id}:genome",
            lineage_id=lineage_id,
            family_id=parent_genome.family_id,
            parent_genome_id=parent_genome.genome_id,
            role=role,
            parameters=parameters,
            mutation_bounds=bounds,
            scientific_domains=list(scientific_domains or parent_genome.scientific_domains),
            budget_bucket=budget_bucket,
            resource_profile=parent_genome.resource_profile,
            budget_weight_pct=budget_weight_pct,
        )

    def _create_challenger(
        self,
        family: FactoryFamily,
        *,
        parent_lineage: LineageRecord,
        mutation_index: int,
        budget_bucket: str,
        proposal: Optional[ScientificAgentProposal] = None,
    ) -> LineageRecord:
        lineage_id = f"{family.family_id}:challenger:{mutation_index}"
        hypothesis_id = f"{lineage_id}:hypothesis"
        experiment_id = f"{lineage_id}:experiment"
        parent_genome = self.registry.load_genome(parent_lineage.lineage_id)
        if parent_genome is None:
            raise ValueError(f"Missing genome for {parent_lineage.lineage_id}")
        budget_weight_pct = round(max(2.0, self._family_budget_weight(family.family_id) / 3.0), 2)
        role = LineageRole.MOONSHOT.value if budget_bucket == "moonshot" else LineageRole.SHADOW_CHALLENGER.value
        genome = self._mutate_genome(
            parent_genome=parent_genome,
            lineage_id=lineage_id,
            role=role,
            mutation_index=mutation_index,
            budget_bucket=budget_bucket,
            budget_weight_pct=budget_weight_pct,
            scientific_domains=list(proposal.scientific_domains) if proposal else None,
            parameter_overrides=dict(proposal.parameter_overrides) if proposal else None,
            mutation_source=proposal.origin if proposal else "deterministic_factory_mutation",
        )
        hypothesis = ResearchHypothesis(
            hypothesis_id=hypothesis_id,
            family_id=family.family_id,
            title=proposal.title if proposal else f"{family.label} Challenger {mutation_index}",
            thesis=proposal.thesis if proposal else f"{family.thesis} Mutated challenger {mutation_index} probes bounded changes to horizon, features, model class, and execution thresholds.",
            scientific_domains=list(proposal.scientific_domains if proposal else (genome.scientific_domains[(mutation_index - 1) % max(1, len(genome.scientific_domains)) :] or genome.scientific_domains)),
            lead_agent_role=proposal.lead_agent_role if proposal else "Genome Mutator",
            success_metric="paper_monthly_roi_pct",
            guardrails=[
                "Mutation remains inside declared bounds.",
                "No live promotion without human approval.",
                "No credentials or hard caps may be mutated.",
            ],
            collaborating_agent_roles=list(proposal.collaborating_agent_roles) if proposal else [],
            origin=proposal.origin if proposal else "deterministic_mutation",
            agent_notes=list(proposal.agent_notes) if proposal else [f"mutation_index={mutation_index}"],
        )
        experiment = ExperimentSpec(
            experiment_id=experiment_id,
            lineage_id=lineage_id,
            family_id=family.family_id,
            hypothesis_id=hypothesis_id,
            genome_id=genome.genome_id,
            goldfish_workspace=str(self.bridge.workspace_path(family.family_id)),
            pipeline_stages=["dataset", "features", "train", "walkforward", "stress", "package"],
            backend_mode="goldfish_sidecar",
            resource_profile=genome.resource_profile,
            inputs={
                "parent_lineage_id": parent_lineage.lineage_id,
                "mutation_index": mutation_index,
                "budget_bucket": budget_bucket,
                "proposal_id": proposal.proposal_id if proposal else None,
                "lead_agent_role": proposal.lead_agent_role if proposal else "Genome Mutator",
                "collaborating_agent_roles": list(proposal.collaborating_agent_roles) if proposal else [],
            },
        )
        lineage = LineageRecord(
            lineage_id=lineage_id,
            family_id=family.family_id,
            label=proposal.title if proposal else f"{family.label} Challenger {mutation_index}",
            role=role,
            current_stage=PromotionStage.IDEA.value,
            target_portfolios=list(family.target_portfolios),
            target_venues=list(family.target_venues),
            hypothesis_id=hypothesis_id,
            genome_id=genome.genome_id,
            experiment_id=experiment_id,
            budget_bucket=budget_bucket,
            budget_weight_pct=budget_weight_pct,
            connector_ids=list(family.primary_connector_ids),
            goldfish_workspace=str(self.bridge.workspace_path(family.family_id)),
            parent_lineage_id=parent_lineage.lineage_id,
            iteration_status="new_candidate",
        )
        self.registry.save_research_pack(
            hypothesis=hypothesis,
            genome=genome,
            experiment=experiment,
            lineage=lineage,
        )
        return lineage

    def _seed_challengers(
        self,
        family: FactoryFamily,
        lineages_by_family: Dict[str, List[LineageRecord]],
        *,
        runtime_mode_value: str,
        recent_actions: List[str],
    ) -> None:
        if runtime_mode_value != "full":
            return
        active = [lineage for lineage in lineages_by_family.get(family.family_id, []) if lineage.active]
        champion = next((lineage for lineage in active if lineage.lineage_id == family.champion_lineage_id), None)
        if champion is None and active:
            champion = active[0]
        if champion is None:
            return
        champion_genome = self.registry.load_genome(champion.lineage_id)
        if champion_genome is None:
            return
        max_shadow = int(champion_genome.parameters.get("max_shadow_challengers", 5) or 5)
        desired_shadow = min(max_shadow, max(1, self._cycle_count))
        existing_shadow = [
            lineage
            for lineage in active
            if lineage.role in {LineageRole.SHADOW_CHALLENGER.value, LineageRole.MOONSHOT.value}
        ]
        if len(existing_shadow) >= desired_shadow:
            return
        budget_sequence = ["incumbent", "adjacent", "moonshot", "incumbent", "adjacent"]
        mutation_index = len(lineages_by_family.get(family.family_id, []))
        proposal = self.strategy_inventor.generate_proposal(
            family=family,
            champion_hypothesis=self.registry.load_hypothesis(champion.lineage_id),
            champion_genome=champion_genome,
            learning_memory=self.registry.learning_memories(family_id=family.family_id, limit=12),
            cycle_count=self._cycle_count,
            proposal_index=mutation_index,
        )
        budget_bucket = proposal.budget_bucket or budget_sequence[(mutation_index - 1) % len(budget_sequence)]
        created = self._create_challenger(
            family,
            parent_lineage=champion,
            mutation_index=mutation_index,
            budget_bucket=budget_bucket,
            proposal=proposal,
        )
        family.shadow_challenger_ids = sorted(set(list(family.shadow_challenger_ids) + [created.lineage_id]))
        self.registry.save_family(family)
        lineages_by_family[family.family_id].append(created)
        recent_actions.append(
            f"[cycle {self._cycle_count}] Seeded challenger {created.lineage_id} for {family.family_id} from {proposal.lead_agent_role} with collaborators {','.join(proposal.collaborating_agent_roles) or 'none'}."
        )

    def _lineage_variant(self, lineage: LineageRecord) -> Dict[str, float]:
        genome = self.registry.load_genome(lineage.lineage_id)
        if genome is None:
            return {
                "roi_delta": 0.0,
                "drawdown_delta": 0.0,
                "slippage_delta": 0.0,
                "calibration_delta": 0.0,
                "capacity_delta": 0.0,
                "failure_delta": 0.0,
                "regime_delta": 0.0,
                "baseline_delta": 0.0,
            }
        parameters = dict(genome.parameters)
        mutation_index = int(parameters.get("mutation_index", 0) or 0)
        if mutation_index <= 0:
            return {
                "roi_delta": 0.0,
                "drawdown_delta": 0.0,
                "slippage_delta": 0.0,
                "calibration_delta": 0.0,
                "capacity_delta": 0.0,
                "failure_delta": 0.0,
                "regime_delta": 0.0,
                "baseline_delta": 0.0,
            }
        feature_bonus = {
            "baseline": 0.0,
            "microstructure": 0.25,
            "cross_science": 0.55,
            "regime": 0.2,
        }.get(str(parameters.get("selected_feature_subset", "baseline")), 0.0)
        model_bonus = {
            "logit": 0.0,
            "gbdt": 0.2,
            "tft": 0.35,
            "transformer": 0.1,
            "rules": -0.25,
        }.get(str(parameters.get("selected_model_class", "logit")), 0.0)
        horizon = int(parameters.get("selected_horizon_seconds", self._preferred_horizon(lineage.family_id)) or self._preferred_horizon(lineage.family_id))
        preferred_horizon = self._preferred_horizon(lineage.family_id)
        horizon_bonus = max(-0.35, 0.25 - abs(horizon - preferred_horizon) / max(preferred_horizon, 1200))
        min_edge = float(parameters.get("selected_min_edge", 0.02) or 0.02)
        stake_fraction = float(parameters.get("selected_stake_fraction", 0.03) or 0.03)
        learning_rate = float(parameters.get("selected_learning_rate", 0.01) or 0.01)
        lookback_hours = float(parameters.get("selected_lookback_hours", 24.0) or 24.0)
        edge_penalty = max(0.0, (min_edge - 0.04) * 25.0)
        stake_penalty = max(0.0, (stake_fraction - 0.05) * 18.0)
        lr_penalty = max(0.0, abs(learning_rate - 0.02) * 12.0)
        lookback_bonus = max(-0.25, 0.25 - abs(lookback_hours - 48.0) / 96.0)
        slot_bias = ((mutation_index % 5) - 2) * 0.18
        digest = int(hashlib.sha1(lineage.lineage_id.encode("utf-8")).hexdigest()[:8], 16)
        hash_bias = ((digest % 11) - 5) / 40.0
        roi_delta = round(feature_bonus + model_bonus + horizon_bonus + lookback_bonus + slot_bias + hash_bias - edge_penalty - stake_penalty - lr_penalty, 4)
        calibration_delta = round((feature_bonus + model_bonus + lookback_bonus + hash_bias) / 20.0, 4)
        drawdown_delta = round(max(-1.0, stake_penalty + lr_penalty - horizon_bonus - 0.2), 4)
        slippage_delta = round((feature_bonus * 0.6) - (edge_penalty * 0.8) - 0.15 + hash_bias, 4)
        capacity_delta = round((lookback_bonus * 0.4) + (0.2 if str(parameters.get("selected_model_class")) != "transformer" else -0.15), 4)
        failure_delta = round(max(-0.02, 0.005 + (stake_penalty * 0.01) + (lr_penalty * 0.01) - (feature_bonus * 0.008)), 4)
        regime_delta = round(max(-0.2, min(0.2, horizon_bonus + (feature_bonus * 0.2) - 0.05)), 4)
        baseline_delta = 1.0 if roi_delta > 0.45 else (-1.0 if roi_delta < -0.35 else 0.0)
        return {
            "roi_delta": roi_delta,
            "drawdown_delta": drawdown_delta,
            "slippage_delta": slippage_delta,
            "calibration_delta": calibration_delta,
            "capacity_delta": capacity_delta,
            "failure_delta": failure_delta,
            "regime_delta": regime_delta,
            "baseline_delta": baseline_delta,
        }

    def _adjust_bundle_for_lineage(self, lineage: LineageRecord, bundle: EvaluationBundle) -> EvaluationBundle:
        variant = self._lineage_variant(lineage)
        monthly_roi = round(float(bundle.monthly_roi_pct) + variant["roi_delta"], 4)
        drawdown = round(max(0.0, float(bundle.max_drawdown_pct) + variant["drawdown_delta"]), 4)
        slippage = round(float(bundle.slippage_headroom_pct) + variant["slippage_delta"], 4)
        calibration = round(float(bundle.calibration_lift_abs) + variant["calibration_delta"], 4)
        capacity = round(max(0.0, min(1.0, float(bundle.capacity_score) + variant["capacity_delta"])), 4)
        failure_rate = round(max(0.0, min(1.0, float(bundle.failure_rate) + variant["failure_delta"])), 4)
        regime = round(max(0.0, min(1.0, float(bundle.regime_robustness) + variant["regime_delta"])), 4)
        baseline_beaten_windows = max(0, min(3, int(bundle.baseline_beaten_windows + variant["baseline_delta"])))
        windows = [
            replace(
                window,
                monthly_roi_pct=round(float(window.monthly_roi_pct) + variant["roi_delta"], 4),
                brier_lift_abs=round(float(window.brier_lift_abs) + variant["calibration_delta"], 4),
                drawdown_pct=round(max(0.0, float(window.drawdown_pct) + variant["drawdown_delta"]), 4),
                slippage_headroom_pct=round(float(window.slippage_headroom_pct) + variant["slippage_delta"], 4),
                failure_rate=failure_rate,
                regime_robustness=regime,
            )
            for window in bundle.windows
        ]
        return replace(
            bundle,
            evaluation_id=f"{bundle.evaluation_id}:{lineage.lineage_id.split(':')[-1]}",
            windows=windows,
            monthly_roi_pct=monthly_roi,
            max_drawdown_pct=drawdown,
            slippage_headroom_pct=slippage,
            calibration_lift_abs=calibration,
            capacity_score=capacity,
            failure_rate=failure_rate,
            regime_robustness=regime,
            baseline_beaten_windows=baseline_beaten_windows,
            stress_positive=bool(bundle.stress_positive and (monthly_roi > 0.0) and (slippage > 0.0)),
            net_pnl=round(float(bundle.net_pnl) + variant["roi_delta"], 4),
            notes=list(bundle.notes) + [f"variant_applied={lineage.lineage_id}"],
        )

    def _queue_status(self, lineage: LineageRecord) -> str:
        if not lineage.active:
            return "retired"
        if lineage.current_stage in {PromotionStage.CANARY_READY.value, PromotionStage.LIVE_READY.value}:
            return "promotion_candidate"
        if lineage.current_stage == PromotionStage.APPROVED_LIVE.value:
            return "approved_live"
        if lineage.current_stage == PromotionStage.PAPER.value:
            return "paper"
        if lineage.current_stage == PromotionStage.SHADOW.value:
            return "shadow"
        return "queued"

    def _queue_priority(self, lineage: LineageRecord) -> int:
        base = {
            LineageRole.CHAMPION.value: 10,
            LineageRole.PAPER_CHALLENGER.value: 20,
            LineageRole.SHADOW_CHALLENGER.value: 30,
            LineageRole.MOONSHOT.value: 40,
        }.get(lineage.role, 50)
        if not lineage.active:
            base += 50
        return base

    def _refresh_queue_entries(self, entries: List[ExperimentQueueEntry]) -> List[ExperimentQueueEntry]:
        refreshed_entries: List[ExperimentQueueEntry] = []
        for entry in entries:
            lineage = self.registry.load_lineage(entry.lineage_id)
            if lineage is None:
                refreshed_entries.append(entry)
                continue
            refreshed_entries.append(
                replace(
                    entry,
                    role=lineage.role,
                    current_stage=lineage.current_stage,
                    status=self._queue_status(lineage),
                    priority=self._queue_priority(lineage),
                    updated_at=utc_now_iso(),
                    notes=[
                        f"loss_streak={int(lineage.loss_streak or 0)}",
                        f"tweak_count={int(lineage.tweak_count or 0)}/{int(lineage.max_tweaks or 2)}",
                    ],
                )
            )
        return refreshed_entries

    def _reclassify_family(
        self,
        family: FactoryFamily,
        ranked_rows: List[Dict[str, Any]],
        *,
        recent_actions: List[str],
    ) -> None:
        active_ranked = [row for row in ranked_rows if row.get("active", True)]
        if not active_ranked:
            return
        new_champion_id = str(active_ranked[0]["lineage_id"])
        if family.champion_lineage_id != new_champion_id:
            recent_actions.append(
                f"[cycle {self._cycle_count}] Champion rotated for {family.family_id}: {family.champion_lineage_id} -> {new_champion_id}."
            )
        family.champion_lineage_id = new_champion_id
        paper_candidates = [
            row["lineage_id"]
            for row in active_ranked[1:]
            if row.get("current_stage") in {
                PromotionStage.PAPER.value,
                PromotionStage.CANARY_READY.value,
                PromotionStage.LIVE_READY.value,
                PromotionStage.APPROVED_LIVE.value,
            }
        ][:2]
        shadow_candidates = [
            row["lineage_id"]
            for row in active_ranked[1:]
            if row["lineage_id"] not in paper_candidates
        ][:5]
        family.paper_challenger_ids = paper_candidates
        family.shadow_challenger_ids = shadow_candidates
        self.registry.save_family(family)

        for row in ranked_rows:
            lineage = self.registry.load_lineage(str(row["lineage_id"]))
            if lineage is None:
                continue
            if not lineage.active:
                continue
            if lineage.lineage_id == family.champion_lineage_id:
                lineage.role = LineageRole.CHAMPION.value
                lineage.loss_streak = 0
                lineage.iteration_status = "champion"
            elif lineage.lineage_id in family.paper_challenger_ids:
                lineage.role = LineageRole.PAPER_CHALLENGER.value
                if lineage.iteration_status == "new_candidate":
                    lineage.iteration_status = "paper_candidate"
            else:
                lineage.role = LineageRole.SHADOW_CHALLENGER.value
            self.registry.save_lineage(lineage)
            genome = self.registry.load_genome(lineage.lineage_id)
            if genome is not None and genome.role != lineage.role:
                genome.role = lineage.role
                self.registry.save_genome(lineage.lineage_id, genome)

    def _tweak_lineage_for_underperformance(
        self,
        lineage: LineageRecord,
        row: Dict[str, Any],
        *,
        recent_actions: List[str],
    ) -> None:
        genome = self.registry.load_genome(lineage.lineage_id)
        hypothesis = self.registry.load_hypothesis(lineage.lineage_id)
        experiment = self.registry.load_experiment(lineage.lineage_id)
        if genome is None:
            return
        tweak_number = int(lineage.tweak_count or 0) + 1
        hard_vetoes = list(row.get("hard_vetoes") or [])
        reason = hard_vetoes[0] if hard_vetoes else "underperforming_vs_champion"
        parameters = dict(genome.parameters)
        current_feature = str(parameters.get("selected_feature_subset", "baseline") or "baseline")
        current_model = str(parameters.get("selected_model_class", "logit") or "logit")
        current_horizon = int(parameters.get("selected_horizon_seconds", self._preferred_horizon(lineage.family_id)) or self._preferred_horizon(lineage.family_id))
        current_lookback = float(parameters.get("selected_lookback_hours", 48.0) or 48.0)
        current_min_edge = float(parameters.get("selected_min_edge", 0.03) or 0.03)
        current_stake = float(parameters.get("selected_stake_fraction", 0.03) or 0.03)
        if "drawdown" in reason.lower() or float(row.get("max_drawdown_pct", 0.0) or 0.0) > float(getattr(config, "FACTORY_PAPER_GATE_MAX_DRAWDOWN_PCT", 8.0)):
            parameters["selected_stake_fraction"] = self._clip_bounds(
                genome.mutation_bounds.execution_thresholds.get("stake_fraction") or [0.01, 0.1],
                current_stake * 0.75,
                fallback=current_stake,
            )
            parameters["selected_min_edge"] = self._clip_bounds(
                genome.mutation_bounds.execution_thresholds.get("min_edge") or [0.01, 0.1],
                current_min_edge + 0.01,
                fallback=current_min_edge,
            )
            parameters["selected_feature_subset"] = self._allowed_choice(
                genome.mutation_bounds.feature_subsets,
                "regime",
                fallback=current_feature,
            )
        elif float(row.get("calibration_lift_abs", 0.0) or 0.0) > 0.0:
            parameters["selected_model_class"] = self._rotate_choice(
                genome.mutation_bounds.model_classes,
                current_model,
                step=tweak_number,
                fallback="logit",
            )
            parameters["selected_lookback_hours"] = self._clip_bounds(
                genome.mutation_bounds.hyperparameter_ranges.get("lookback_hours") or [6.0, 168.0],
                current_lookback + 12.0,
                fallback=current_lookback,
            )
            parameters["selected_min_edge"] = self._clip_bounds(
                genome.mutation_bounds.execution_thresholds.get("min_edge") or [0.01, 0.1],
                max(0.01, current_min_edge - 0.005),
                fallback=current_min_edge,
            )
        else:
            parameters["selected_feature_subset"] = self._rotate_choice(
                genome.mutation_bounds.feature_subsets,
                current_feature,
                step=tweak_number,
                fallback="baseline",
            )
            parameters["selected_horizon_seconds"] = self._nearest_choice(
                genome.mutation_bounds.horizons_seconds,
                current_horizon + (300 * tweak_number),
            )
            parameters["selected_lookback_hours"] = self._clip_bounds(
                genome.mutation_bounds.hyperparameter_ranges.get("lookback_hours") or [6.0, 168.0],
                max(6.0, current_lookback - 6.0),
                fallback=current_lookback,
            )
        parameters["mutation_source"] = "underperformance_tweak"
        parameters["last_tweak_reason"] = reason
        parameters["last_tweak_cycle"] = self._cycle_count
        genome.parameters = parameters
        self.registry.save_genome(lineage.lineage_id, genome)
        lineage.tweak_count = tweak_number
        lineage.loss_streak = int(lineage.loss_streak or 0) + 1
        lineage.iteration_status = "tweaked"
        lineage.updated_at = utc_now_iso()
        self.registry.save_lineage(lineage)
        if hypothesis is not None:
            hypothesis.agent_notes = list(hypothesis.agent_notes) + [
                f"tweak_{tweak_number}: {reason}",
            ]
            self.registry.save_hypothesis(lineage.lineage_id, hypothesis)
        if experiment is not None:
            experiment.inputs = dict(experiment.inputs or {})
            experiment.inputs["tweak_count"] = lineage.tweak_count
            experiment.inputs["last_tweak_reason"] = reason
            self.registry.save_experiment(lineage.lineage_id, experiment)
        recent_actions.append(
            f"[cycle {self._cycle_count}] Tweaked {lineage.lineage_id} ({tweak_number}/{int(lineage.max_tweaks or 2)}) after {reason}."
        )

    def _record_learning_memory(
        self,
        lineage: LineageRecord,
        row: Dict[str, Any],
        *,
        reason: str,
    ) -> None:
        hypothesis = self.registry.load_hypothesis(lineage.lineage_id)
        summary = (
            f"{lineage.lineage_id} retired after {int(lineage.tweak_count or 0)} tweaks. "
            f"ROI={float(row.get('monthly_roi_pct', 0.0) or 0.0):.4f}, "
            f"fitness={float(row.get('fitness_score', 0.0) or 0.0):.4f}, "
            f"reason={reason}."
        )
        recommendations = []
        if row.get("hard_vetoes"):
            recommendations.append(f"avoid veto pattern {list(row.get('hard_vetoes') or [reason])[0]}")
        if float(row.get("calibration_lift_abs", 0.0) or 0.0) <= 0.0:
            recommendations.append("prefer higher-information or microstructure features next")
        if float(row.get("monthly_roi_pct", 0.0) or 0.0) < 0.0:
            recommendations.append("tighten edge thresholds and reduce stake fraction in successor")
        memory = LearningMemoryEntry(
            memory_id=f"{lineage.lineage_id}:memory:{int(lineage.tweak_count or 0)}",
            family_id=lineage.family_id,
            lineage_id=lineage.lineage_id,
            hypothesis_id=lineage.hypothesis_id,
            outcome="retired_underperformance",
            summary=summary,
            scientific_domains=list((hypothesis.scientific_domains if hypothesis else []) or []),
            lead_agent_role=str((hypothesis.lead_agent_role if hypothesis else "unknown") or "unknown"),
            tweak_count=int(lineage.tweak_count or 0),
            decision_stage=lineage.current_stage,
            metrics={
                "monthly_roi_pct": float(row.get("monthly_roi_pct", 0.0) or 0.0),
                "fitness_score": float(row.get("fitness_score", 0.0) or 0.0),
                "pareto_rank": row.get("pareto_rank"),
            },
            blockers=list(row.get("hard_vetoes") or []),
            recommendations=recommendations or ["change scientific collaboration mix before retrying"],
        )
        self.registry.save_learning_memory(memory)
        lineage.last_memory_id = memory.memory_id

    def _retire_or_update_lineages(
        self,
        family: FactoryFamily,
        ranked_rows: List[Dict[str, Any]],
        *,
        recent_actions: List[str],
    ) -> None:
        active_ranked = [row for row in ranked_rows if row.get("active", True)]
        if not active_ranked:
            return
        champion_row = active_ranked[0]
        champion_score = float(champion_row.get("fitness_score", 0.0) or 0.0)
        retired_ids = set(family.retired_lineage_ids)
        for row in active_ranked[1:]:
            lineage = self.registry.load_lineage(str(row["lineage_id"]))
            if lineage is None or not lineage.active:
                continue
            execution_signal_ready = bool(row.get("execution_has_signal"))
            if not execution_signal_ready:
                lineage.blockers = list(dict.fromkeys(list(lineage.blockers) + ["awaiting_execution_validation"]))
                lineage.iteration_status = "awaiting_execution_validation"
                self.registry.save_lineage(lineage)
                recent_actions.append(
                    f"[cycle {self._cycle_count}] Deferred tweak/retirement for {lineage.lineage_id} until execution validation is present."
                )
                continue
            monthly_roi = float(row.get("monthly_roi_pct", 0.0) or 0.0)
            hard_vetoes = list(row.get("hard_vetoes") or [])
            score = float(row.get("fitness_score", 0.0) or 0.0)
            underperforming = bool(hard_vetoes) or (score < (champion_score - 0.25)) or monthly_roi < 0.0
            if underperforming:
                if int(lineage.tweak_count or 0) < int(lineage.max_tweaks or 2):
                    self._tweak_lineage_for_underperformance(
                        lineage,
                        row,
                        recent_actions=recent_actions,
                    )
                    continue
                lineage.loss_streak = int(lineage.loss_streak or 0) + 1
            else:
                lineage.loss_streak = 0
                if lineage.iteration_status == "tweaked":
                    lineage.iteration_status = "stabilized_after_tweak"
            if underperforming and int(lineage.tweak_count or 0) >= int(lineage.max_tweaks or 2):
                lineage.active = False
                lineage.retired_at = utc_now_iso()
                lineage.iteration_status = "retired"
                lineage.retirement_reason = "max_tweaks_exhausted_underperforming"
                lineage.blockers = list(dict.fromkeys(list(lineage.blockers) + [lineage.retirement_reason]))
                self._record_learning_memory(
                    lineage,
                    row,
                    reason=hard_vetoes[0] if hard_vetoes else "score_and_roi_underperformance",
                )
                retired_ids.add(lineage.lineage_id)
                recent_actions.append(
                    f"[cycle {self._cycle_count}] Retired {lineage.lineage_id} from {family.family_id}: {lineage.retirement_reason}."
                )
            self.registry.save_lineage(lineage)
        family.retired_lineage_ids = sorted(retired_ids)
        family.shadow_challenger_ids = [lineage_id for lineage_id in family.shadow_challenger_ids if lineage_id not in retired_ids]
        family.paper_challenger_ids = [lineage_id for lineage_id in family.paper_challenger_ids if lineage_id not in retired_ids]
        self.registry.save_family(family)

    def _connector_snapshots(self) -> List[Dict[str, Any]]:
        return [adapter.snapshot().to_dict() for adapter in self.connectors]

    def _execution_validation_snapshot(self, lineage: LineageRecord) -> Dict[str, Any]:
        targets: List[Dict[str, Any]] = []
        running_target_count = 0
        recent_trade_count = 0
        recent_event_count = 0
        for requested_target in list(lineage.target_portfolios):
            resolved_target = resolve_target_portfolio(str(requested_target))
            store = PortfolioStateStore(resolved_target)
            heartbeat = store.read_heartbeat()
            target_state = store.read_state()
            trades = store.read_trades(limit=20)
            events = store.read_events(limit=20)
            target_running = bool(target_state.get("running")) or str(heartbeat.get("status") or "").lower() == "running"
            if target_running:
                running_target_count += 1
            recent_trade_count += len(trades)
            recent_event_count += len(events)
            targets.append(
                {
                    "requested_target": str(requested_target),
                    "resolved_target": resolved_target,
                    "running": target_running,
                    "heartbeat_ts": heartbeat.get("ts"),
                    "recent_trade_count": len(trades),
                    "recent_event_count": len(events),
                    "status": target_state.get("status"),
                }
            )
        return {
            "targets": targets,
            "running_target_count": running_target_count,
            "recent_trade_count": recent_trade_count,
            "recent_event_count": recent_event_count,
            "has_execution_signal": bool(running_target_count or recent_trade_count or recent_event_count),
        }

    def _run_experiment(self, lineage: LineageRecord) -> Dict[str, Any]:
        genome = self.registry.load_genome(lineage.lineage_id)
        experiment = self.registry.load_experiment(lineage.lineage_id)
        if genome is None or experiment is None:
            return {"mode": "missing_inputs", "bundles": [], "artifact_summary": None}
        result = self.experiment_runner.run(
            lineage=lineage,
            genome=genome,
            experiment=experiment,
        )
        artifact_summary = dict(result.get("artifact_summary") or {})
        if artifact_summary:
            experiment.expected_outputs = dict(experiment.expected_outputs or {})
            experiment.expected_outputs["latest_run"] = artifact_summary
            self.registry.save_experiment(lineage.lineage_id, experiment)
        return result

    def _blend_runtime_and_offline_bundle(
        self,
        *,
        lineage: LineageRecord,
        runtime_bundle: EvaluationBundle,
        walkforward_bundle: Optional[EvaluationBundle],
        stress_bundle: Optional[EvaluationBundle],
        stage: str,
    ) -> EvaluationBundle:
        if walkforward_bundle is None and stress_bundle is None:
            return runtime_bundle
        walkforward = walkforward_bundle or runtime_bundle
        stress = stress_bundle or walkforward
        artifact_notes = [
            note
            for note in list(walkforward.notes) + list(stress.notes)
            if "package_path=" in str(note)
        ]
        return replace(
            runtime_bundle,
            evaluation_id=f"{runtime_bundle.evaluation_id}:{stage}:artifact",
            stage=stage,
            source="factory_runtime_plus_artifact",
            windows=list(walkforward.windows or runtime_bundle.windows),
            monthly_roi_pct=round(
                (float(runtime_bundle.monthly_roi_pct) * 0.65) + (float(walkforward.monthly_roi_pct) * 0.35),
                4,
            ),
            max_drawdown_pct=round(
                max(float(runtime_bundle.max_drawdown_pct), float(walkforward.max_drawdown_pct)),
                4,
            ),
            slippage_headroom_pct=round(
                min(float(runtime_bundle.slippage_headroom_pct), float(stress.slippage_headroom_pct)),
                4,
            ),
            calibration_lift_abs=round(
                float(runtime_bundle.calibration_lift_abs) + (float(walkforward.calibration_lift_abs) * 0.5),
                6,
            ),
            turnover=round(
                max(float(runtime_bundle.turnover), float(walkforward.turnover)),
                4,
            ),
            capacity_score=round(
                min(1.0, (float(runtime_bundle.capacity_score) * 0.5) + (float(walkforward.capacity_score) * 0.5)),
                4,
            ),
            failure_rate=round(
                max(float(runtime_bundle.failure_rate), float(stress.failure_rate)),
                4,
            ),
            regime_robustness=round(
                min(1.0, (float(runtime_bundle.regime_robustness) * 0.5) + (float(walkforward.regime_robustness) * 0.5)),
                4,
            ),
            baseline_beaten_windows=max(
                int(runtime_bundle.baseline_beaten_windows),
                int(walkforward.baseline_beaten_windows),
            ),
            stress_positive=bool(stress.stress_positive and runtime_bundle.slippage_headroom_pct > 0.0),
            trade_count=max(int(runtime_bundle.trade_count), int(walkforward.trade_count)),
            settled_count=max(int(runtime_bundle.settled_count), int(walkforward.settled_count)),
            paper_days=max(int(runtime_bundle.paper_days), int(walkforward.paper_days)),
            net_pnl=round(
                float(runtime_bundle.net_pnl) + (float(walkforward.net_pnl) * 0.25),
                4,
            ),
            notes=list(dict.fromkeys(list(runtime_bundle.notes) + artifact_notes + [f"artifact_lineage={lineage.lineage_id}"])),
        )

    def _workspace_status(self, families: Iterable[FactoryFamily]) -> Dict[str, Dict[str, Any]]:
        status: Dict[str, Dict[str, Any]] = {}
        for family in families:
            status[family.family_id] = self.bridge.ensure_workspace(
                family.family_id,
                thesis=family.thesis,
                pipeline_stages=["dataset", "features", "train", "walkforward", "stress", "package"],
            )
        return status

    def _prediction_paper_bundle(self, lineage: LineageRecord) -> Optional[EvaluationBundle]:
        log_path = self.project_root / "data/prediction/experiments.jsonl"
        rows = _load_jsonl(log_path)
        if not rows:
            return None
        rows = rows[-25:]
        last = rows[-1]
        metrics = dict(last.get("metrics") or {})
        rolling = metrics.get("rolling_200") or metrics.get("rolling_100") or {}
        settled = int(rolling.get("settled", 0) or 0)
        monthly_roi = float(rolling.get("roi_pct", 0.0) or 0.0)
        brier_lift = float(rolling.get("brier_lift_abs", 0.0) or 0.0)
        windows = [
            EvaluationWindow(
                label="prediction_rolling",
                settled_count=settled,
                monthly_roi_pct=monthly_roi,
                baseline_roi_pct=0.0,
                brier_lift_abs=brier_lift,
                drawdown_pct=abs(min(0.0, monthly_roi)) * 0.8,
                slippage_headroom_pct=max(-5.0, monthly_roi - 1.5),
                failure_rate=0.02,
                regime_robustness=0.35,
            )
        ]
        stage = EvaluationStage.PAPER.value if settled else EvaluationStage.WALKFORWARD.value
        return EvaluationBundle(
            evaluation_id=f"{lineage.lineage_id}:prediction:{settled}",
            lineage_id=lineage.lineage_id,
            family_id=lineage.family_id,
            stage=stage,
            source="repo_prediction_experiments",
            windows=windows,
            monthly_roi_pct=monthly_roi,
            max_drawdown_pct=abs(min(0.0, monthly_roi)) * 0.8,
            slippage_headroom_pct=max(-5.0, monthly_roi - 1.5),
            calibration_lift_abs=brier_lift,
            turnover=min(1.0, settled / 200.0),
            capacity_score=0.45,
            failure_rate=0.02,
            regime_robustness=0.35,
            baseline_beaten_windows=3 if brier_lift > 0 and monthly_roi > 0 else 0,
            stress_positive=monthly_roi > 2.0,
            trade_count=settled,
            settled_count=settled,
            paper_days=min(30, max(1, settled // 4)),
            net_pnl=monthly_roi,
        )

    def _funding_bundle(self, lineage: LineageRecord) -> Optional[EvaluationBundle]:
        log_path = self.project_root / "data/funding/experiments.jsonl"
        rows = _load_jsonl(log_path)
        if not rows:
            return None
        rows = rows[-25:]
        last = rows[-1]
        metrics = dict(last.get("metrics") or {})
        rolling = metrics.get("rolling_200") or metrics.get("rolling_100") or {}
        settled = int(rolling.get("settled", 0) or 0)
        monthly_roi = float(rolling.get("roi_pct", 0.0) or 0.0)
        brier_lift = float(rolling.get("brier_lift_abs", 0.0) or 0.0)
        windows = [
            EvaluationWindow(
                label="funding_rolling",
                settled_count=settled,
                monthly_roi_pct=monthly_roi,
                baseline_roi_pct=0.0,
                brier_lift_abs=brier_lift,
                drawdown_pct=max(0.0, 6.0 - monthly_roi),
                slippage_headroom_pct=max(-2.0, monthly_roi - 0.5),
                failure_rate=0.01,
                regime_robustness=0.60,
            )
        ]
        return EvaluationBundle(
            evaluation_id=f"{lineage.lineage_id}:funding:{settled}",
            lineage_id=lineage.lineage_id,
            family_id=lineage.family_id,
            stage=EvaluationStage.PAPER.value if settled else EvaluationStage.WALKFORWARD.value,
            source="repo_funding_experiments",
            windows=windows,
            monthly_roi_pct=monthly_roi,
            max_drawdown_pct=max(0.0, 6.0 - monthly_roi),
            slippage_headroom_pct=max(-2.0, monthly_roi - 0.5),
            calibration_lift_abs=brier_lift,
            turnover=min(1.0, settled / 150.0),
            capacity_score=0.65,
            failure_rate=0.01,
            regime_robustness=0.60,
            baseline_beaten_windows=3 if brier_lift > 0 and monthly_roi >= 5.0 else 1,
            stress_positive=monthly_roi > 0.0,
            trade_count=settled,
            settled_count=settled,
            paper_days=min(30, max(1, settled // 4)),
            net_pnl=monthly_roi,
        )

    def _portfolio_state_bundle(
        self,
        lineage: LineageRecord,
        *,
        portfolio_id: str,
        stage: str,
        capacity_score: float,
        regime_robustness: float,
    ) -> Optional[EvaluationBundle]:
        store = PortfolioStateStore(portfolio_id)
        account = store.read_account()
        if account is None:
            return None
        trades = store.read_trades(limit=500)
        settled = sum(1 for trade in trades if str(trade.get("status", "")).upper() in {"CLOSED", "SETTLED"})
        trade_count = len(trades)
        monthly_roi = float(account.roi_pct)
        drawdown = float(account.drawdown_pct)
        windows = [
            EvaluationWindow(
                label=f"{portfolio_id}_runtime",
                settled_count=max(settled, trade_count),
                monthly_roi_pct=monthly_roi,
                baseline_roi_pct=0.0,
                brier_lift_abs=0.0,
                drawdown_pct=drawdown,
                slippage_headroom_pct=max(-5.0, monthly_roi - 2.0),
                failure_rate=0.01 if trade_count else 0.05,
                regime_robustness=regime_robustness,
            )
        ]
        return EvaluationBundle(
            evaluation_id=f"{lineage.lineage_id}:{portfolio_id}:{trade_count}",
            lineage_id=lineage.lineage_id,
            family_id=lineage.family_id,
            stage=stage,
            source=f"portfolio_state:{portfolio_id}",
            windows=windows,
            monthly_roi_pct=monthly_roi,
            max_drawdown_pct=drawdown,
            slippage_headroom_pct=max(-5.0, monthly_roi - 2.0),
            calibration_lift_abs=0.0,
            turnover=min(1.0, trade_count / 100.0),
            capacity_score=capacity_score,
            failure_rate=0.01 if trade_count else 0.05,
            regime_robustness=regime_robustness,
            baseline_beaten_windows=3 if monthly_roi > 0 else 0,
            stress_positive=monthly_roi > 0.0,
            trade_count=trade_count,
            settled_count=max(settled, trade_count),
            paper_days=min(30, max(1, trade_count // 2)) if trade_count else 0,
            net_pnl=float(account.realized_pnl),
        )

    def _collect_evidence(self, lineage: LineageRecord) -> List[EvaluationBundle]:
        bundles: List[EvaluationBundle] = []
        experiment_result = self._run_experiment(lineage)
        experiment_bundles = list(experiment_result.get("bundles") or [])
        experiment_by_stage = {bundle.stage: bundle for bundle in experiment_bundles}
        if lineage.family_id == "binance_funding_contrarian":
            bundles.extend(experiment_bundles)
            bundle = self._funding_bundle(lineage)
            runtime_bundle = self._portfolio_state_bundle(
                lineage,
                portfolio_id="contrarian_legacy",
                stage=EvaluationStage.PAPER.value,
                capacity_score=0.55,
                regime_robustness=0.55,
            )
            if runtime_bundle:
                bundles.extend(
                    [
                        self._blend_runtime_and_offline_bundle(
                            lineage=lineage,
                            runtime_bundle=runtime_bundle,
                            walkforward_bundle=experiment_by_stage.get(EvaluationStage.WALKFORWARD.value),
                            stress_bundle=experiment_by_stage.get(EvaluationStage.STRESS.value),
                            stage=EvaluationStage.SHADOW.value,
                        ),
                        self._blend_runtime_and_offline_bundle(
                            lineage=lineage,
                            runtime_bundle=runtime_bundle,
                            walkforward_bundle=experiment_by_stage.get(EvaluationStage.WALKFORWARD.value),
                            stress_bundle=experiment_by_stage.get(EvaluationStage.STRESS.value),
                            stage=EvaluationStage.PAPER.value,
                        ),
                    ]
                )
            elif bundle:
                bundles.extend(
                    [
                        replace(bundle, stage=EvaluationStage.WALKFORWARD.value, evaluation_id=f"{bundle.evaluation_id}:wf"),
                        replace(bundle, stage=EvaluationStage.STRESS.value, evaluation_id=f"{bundle.evaluation_id}:stress"),
                        replace(bundle, stage=EvaluationStage.PAPER.value, evaluation_id=f"{bundle.evaluation_id}:paper"),
                    ]
                )
        elif lineage.family_id == "binance_cascade_regime":
            bundles.extend(experiment_bundles)
            runtime_bundle = self._portfolio_state_bundle(
                lineage,
                portfolio_id="cascade_alpha",
                stage=EvaluationStage.PAPER.value,
                capacity_score=0.50,
                regime_robustness=0.65,
            )
            if runtime_bundle:
                bundles.extend(
                    [
                        self._blend_runtime_and_offline_bundle(
                            lineage=lineage,
                            runtime_bundle=runtime_bundle,
                            walkforward_bundle=experiment_by_stage.get(EvaluationStage.WALKFORWARD.value),
                            stress_bundle=experiment_by_stage.get(EvaluationStage.STRESS.value),
                            stage=EvaluationStage.SHADOW.value,
                        ),
                        self._blend_runtime_and_offline_bundle(
                            lineage=lineage,
                            runtime_bundle=runtime_bundle,
                            walkforward_bundle=experiment_by_stage.get(EvaluationStage.WALKFORWARD.value),
                            stress_bundle=experiment_by_stage.get(EvaluationStage.STRESS.value),
                            stage=EvaluationStage.PAPER.value,
                        ),
                    ]
                )
        elif lineage.family_id == "betfair_prediction_value_league":
            bundles.extend(experiment_bundles)
            bundle = self._prediction_paper_bundle(lineage)
            if bundle:
                shadow_bundle = self._blend_runtime_and_offline_bundle(
                    lineage=lineage,
                    runtime_bundle=bundle,
                    walkforward_bundle=experiment_by_stage.get(EvaluationStage.WALKFORWARD.value),
                    stress_bundle=experiment_by_stage.get(EvaluationStage.STRESS.value),
                    stage=EvaluationStage.SHADOW.value,
                )
                paper_bundle = self._blend_runtime_and_offline_bundle(
                    lineage=lineage,
                    runtime_bundle=bundle,
                    walkforward_bundle=experiment_by_stage.get(EvaluationStage.WALKFORWARD.value),
                    stress_bundle=experiment_by_stage.get(EvaluationStage.STRESS.value),
                    stage=EvaluationStage.PAPER.value,
                )
                bundles.extend(
                    [
                        shadow_bundle,
                        paper_bundle,
                    ]
                )
        elif lineage.family_id == "betfair_information_lag":
            bundles.extend(experiment_bundles)
            bundle = self._portfolio_state_bundle(
                lineage,
                portfolio_id="betfair_core",
                stage=EvaluationStage.SHADOW.value,
                capacity_score=0.30,
                regime_robustness=0.45,
            )
            if bundle:
                bundles.extend(
                    [
                        self._blend_runtime_and_offline_bundle(
                            lineage=lineage,
                            runtime_bundle=replace(bundle, stage=EvaluationStage.PAPER.value),
                            walkforward_bundle=experiment_by_stage.get(EvaluationStage.WALKFORWARD.value),
                            stress_bundle=experiment_by_stage.get(EvaluationStage.STRESS.value),
                            stage=EvaluationStage.SHADOW.value,
                        ),
                        self._blend_runtime_and_offline_bundle(
                            lineage=lineage,
                            runtime_bundle=replace(bundle, stage=EvaluationStage.PAPER.value),
                            walkforward_bundle=experiment_by_stage.get(EvaluationStage.WALKFORWARD.value),
                            stress_bundle=experiment_by_stage.get(EvaluationStage.STRESS.value),
                            stage=EvaluationStage.PAPER.value,
                        ),
                    ]
                )
        elif lineage.family_id == "polymarket_cross_venue":
            bundles.extend(experiment_bundles)
            bundle = self._portfolio_state_bundle(
                lineage,
                portfolio_id="polymarket_quantum_fold",
                stage=EvaluationStage.PAPER.value,
                capacity_score=0.40,
                regime_robustness=0.40,
            )
            if bundle:
                bundles.extend(
                    [
                        self._blend_runtime_and_offline_bundle(
                            lineage=lineage,
                            runtime_bundle=bundle,
                            walkforward_bundle=experiment_by_stage.get(EvaluationStage.WALKFORWARD.value),
                            stress_bundle=experiment_by_stage.get(EvaluationStage.STRESS.value),
                            stage=EvaluationStage.SHADOW.value,
                        ),
                        self._blend_runtime_and_offline_bundle(
                            lineage=lineage,
                            runtime_bundle=bundle,
                            walkforward_bundle=experiment_by_stage.get(EvaluationStage.WALKFORWARD.value),
                            stress_bundle=experiment_by_stage.get(EvaluationStage.STRESS.value),
                            stage=EvaluationStage.PAPER.value,
                        ),
                    ]
                )
        if lineage.family_id not in {
            "betfair_prediction_value_league",
            "binance_funding_contrarian",
            "binance_cascade_regime",
            "betfair_information_lag",
            "polymarket_cross_venue",
        }:
            bundles = [self._adjust_bundle_for_lineage(lineage, bundle) for bundle in bundles]
        ranked = assign_pareto_ranks(bundles)
        for bundle in ranked:
            bundle.hard_vetoes = compute_hard_vetoes(bundle)
        return ranked

    def _save_evidence(self, lineage: LineageRecord) -> Dict[str, EvaluationBundle]:
        by_stage: Dict[str, EvaluationBundle] = {}
        for bundle in self._collect_evidence(lineage):
            self.registry.save_evaluation(bundle)
            previous = by_stage.get(bundle.stage)
            if previous is None or str(bundle.generated_at) > str(previous.generated_at):
                by_stage[bundle.stage] = bundle
        return by_stage

    def _maybe_publish_manifest(
        self,
        lineage: LineageRecord,
        paper_bundle: Optional[EvaluationBundle],
    ) -> Optional[str]:
        if paper_bundle is None:
            return None
        blockers = self.promotion.paper_gate_blockers(
            paper_bundle,
            slow_strategy="polymarket" in lineage.family_id or "information" in lineage.family_id,
        )
        if blockers:
            return None
        existing = [
            manifest for manifest in self.registry.manifests()
            if manifest.lineage_id == lineage.lineage_id
        ]
        if existing:
            return existing[-1].manifest_id
        artifact_refs = {
            "workspace": lineage.goldfish_workspace,
            "paper_bundle_id": paper_bundle.evaluation_id,
        }
        experiment = self.registry.load_experiment(lineage.lineage_id)
        latest_run = dict((experiment.expected_outputs or {}).get("latest_run") or {}) if experiment else {}
        if latest_run.get("package_path"):
            artifact_refs["package"] = str(latest_run["package_path"])
            artifact_refs["run_id"] = latest_run.get("run_id")
        if lineage.family_id == "binance_funding_contrarian":
            artifact_refs["model_meta"] = "data/funding_models/funding_predictor_meta.json"
        elif lineage.family_id == "betfair_prediction_value_league":
            artifact_refs["policy_gate"] = str(getattr(config, "PREDICTION_POLICY_GATE_PATH", "data/models/prediction_policy_gate_v1.json"))
        manifest = self.bridge.publish_candidate_manifest(
            lineage_id=lineage.lineage_id,
            family_id=lineage.family_id,
            portfolio_targets=list(lineage.target_portfolios),
            venue_targets=list(lineage.target_venues),
            artifact_refs=artifact_refs,
            runtime_overrides={"resource_profile": "local-first-hybrid"},
            notes=["Candidate manifest published by Goldfish sidecar bridge; human approval required for live use."],
        )
        self.registry.save_manifest(manifest)
        return manifest.manifest_id

    def run_cycle(self) -> Dict[str, Any]:
        runtime_mode = self._runtime_mode()
        if runtime_mode.is_hard_stop:
            return self._hard_stop_state()
        self._cycle_count += 1
        families = self.registry.families()
        workspace_status = self._workspace_status(families)
        connector_snapshots = self._connector_snapshots()
        manifests_by_lineage = self._latest_manifest_by_lineage()
        connector_ready = {
            snapshot["connector_id"]: bool(snapshot.get("ready"))
            for snapshot in connector_snapshots
        }
        lineages_by_family = self._lineages_by_family()
        family_rankings: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        total_paper_pnl = 0.0
        live_loadable_manifests = []
        lineage_summaries: List[Dict[str, Any]] = []
        lineage_summary_by_id: Dict[str, Dict[str, Any]] = {}
        queue_entries: List[ExperimentQueueEntry] = []
        recent_actions = [f"[cycle {self._cycle_count}] Connector snapshots refreshed and factory evidence re-evaluated."]
        if runtime_mode.is_cost_saver:
            recent_actions.append(
                f"[cycle {self._cycle_count}] Runtime mode cost_saver kept deterministic evaluation active while token-consuming agentic work stayed paused."
            )
        for family in families:
            self._seed_challengers(
                family,
                lineages_by_family,
                runtime_mode_value=runtime_mode.value,
                recent_actions=recent_actions,
            )
        for lineage in self.registry.lineages():
            latest_by_stage = (
                self._save_evidence(lineage)
                if lineage.active
                else self.registry.latest_evaluation_by_stage(lineage.lineage_id)
            )
            data_ready = all(connector_ready.get(connector_id, False) for connector_id in lineage.connector_ids)
            workspace_ready = bool(workspace_status.get(lineage.family_id, {}).get("ready"))
            manifest = manifests_by_lineage.get(lineage.lineage_id)
            if runtime_mode.is_full and lineage.active:
                manifest_id = self._maybe_publish_manifest(lineage, latest_by_stage.get(EvaluationStage.PAPER.value))
                if manifest_id:
                    manifest = self.registry.load_manifest(manifest_id) or manifest
                    if manifest is not None:
                        manifests_by_lineage[lineage.lineage_id] = manifest
                decision = self.promotion.decide(
                    lineage,
                    data_ready=data_ready,
                    workspace_ready=workspace_ready,
                    walkforward_bundle=latest_by_stage.get(EvaluationStage.WALKFORWARD.value),
                    stress_bundle=latest_by_stage.get(EvaluationStage.STRESS.value),
                    paper_bundle=latest_by_stage.get(EvaluationStage.PAPER.value),
                    manifest_status=manifest.status if manifest is not None else None,
                    approved_by=manifest.approved_by if manifest is not None else None,
                )
                self.registry.cas_transition(
                    lineage.lineage_id,
                    expected_stage=lineage.current_stage,
                    next_stage=decision.next_stage,
                    blockers=decision.blockers,
                    decision=decision.to_dict(),
                )
                refreshed = self.registry.load_lineage(lineage.lineage_id) or lineage
            else:
                refreshed = self.registry.load_lineage(lineage.lineage_id) or lineage
            manifest_id = manifest.manifest_id if manifest is not None else None
            experiment = self.registry.load_experiment(refreshed.lineage_id)
            hypothesis = self.registry.load_hypothesis(refreshed.lineage_id)
            latest_run = dict((experiment.expected_outputs or {}).get("latest_run") or {}) if experiment else {}
            latest_bundle = (
                latest_by_stage.get(EvaluationStage.PAPER.value)
                or latest_by_stage.get(EvaluationStage.STRESS.value)
                or latest_by_stage.get(EvaluationStage.WALKFORWARD.value)
            )
            execution_validation = self._execution_validation_snapshot(refreshed)
            if latest_bundle is not None and lineage.active:
                total_paper_pnl += float(latest_bundle.net_pnl or 0.0)
            lineage_summary = {
                "lineage_id": refreshed.lineage_id,
                "family_id": refreshed.family_id,
                "label": refreshed.label,
                "role": refreshed.role,
                "current_stage": refreshed.current_stage,
                "active": bool(refreshed.active),
                "loss_streak": int(refreshed.loss_streak or 0),
                "tweak_count": int(refreshed.tweak_count or 0),
                "max_tweaks": int(refreshed.max_tweaks or 2),
                "iteration_status": refreshed.iteration_status,
                "parent_lineage_id": refreshed.parent_lineage_id,
                "retired_at": refreshed.retired_at,
                "retirement_reason": refreshed.retirement_reason,
                "last_memory_id": refreshed.last_memory_id,
                "budget_bucket": refreshed.budget_bucket,
                "budget_weight_pct": float(refreshed.budget_weight_pct or 0.0),
                "connector_ids": list(refreshed.connector_ids),
                "target_portfolios": list(refreshed.target_portfolios),
                "lead_agent_role": hypothesis.lead_agent_role if hypothesis else None,
                "collaborating_agent_roles": list((hypothesis.collaborating_agent_roles if hypothesis else []) or []),
                "scientific_domains": list((hypothesis.scientific_domains if hypothesis else []) or []),
                "hypothesis_origin": hypothesis.origin if hypothesis else None,
                "blockers": list(refreshed.blockers),
                "fitness_score": float((latest_bundle.fitness_score if latest_bundle else 0.0) or 0.0),
                "pareto_rank": latest_bundle.pareto_rank if latest_bundle else None,
                "monthly_roi_pct": float((latest_bundle.monthly_roi_pct if latest_bundle else 0.0) or 0.0),
                "calibration_lift_abs": float((latest_bundle.calibration_lift_abs if latest_bundle else 0.0) or 0.0),
                "net_pnl": float((latest_bundle.net_pnl if latest_bundle else 0.0) or 0.0),
                "trade_count": int((latest_bundle.trade_count if latest_bundle else 0) or 0),
                "settled_count": int((latest_bundle.settled_count if latest_bundle else 0) or 0),
                "paper_days": int((latest_bundle.paper_days if latest_bundle else 0) or 0),
                "hard_vetoes": list((latest_bundle.hard_vetoes if latest_bundle else []) or []),
                "latest_artifact_mode": latest_run.get("mode"),
                "latest_artifact_package": latest_run.get("package_path"),
                "latest_artifact_run_id": latest_run.get("run_id"),
                "strict_gate_pass": refreshed.current_stage in {
                    PromotionStage.CANARY_READY.value,
                    PromotionStage.LIVE_READY.value,
                    PromotionStage.APPROVED_LIVE.value,
                },
                "manifest_id": manifest_id,
                "execution_validation": execution_validation,
                "execution_running_target_count": int(execution_validation["running_target_count"]),
                "execution_recent_trade_count": int(execution_validation["recent_trade_count"]),
                "execution_recent_event_count": int(execution_validation["recent_event_count"]),
                "execution_has_signal": bool(execution_validation["has_execution_signal"]),
            }
            lineage_summaries.append(lineage_summary)
            lineage_summary_by_id[refreshed.lineage_id] = lineage_summary
            family_rankings[refreshed.family_id].append(lineage_summary)
            queue_entries.append(
                ExperimentQueueEntry(
                    queue_id=f"{refreshed.lineage_id}:{refreshed.current_stage}",
                    family_id=refreshed.family_id,
                    lineage_id=refreshed.lineage_id,
                    experiment_id=refreshed.experiment_id,
                    role=refreshed.role,
                    current_stage=refreshed.current_stage,
                    status=self._queue_status(refreshed),
                    priority=self._queue_priority(refreshed),
                    notes=[
                        f"loss_streak={int(refreshed.loss_streak or 0)}",
                        f"tweak_count={int(refreshed.tweak_count or 0)}/{int(refreshed.max_tweaks or 2)}",
                    ],
                )
            )
            if manifest is not None and manifest.is_live_loadable() and runtime_mode.factory_influence_allowed:
                live_loadable_manifests.append(manifest.to_dict())
        family_summaries: List[Dict[str, Any]] = []
        for family in families:
            ranked = sorted(
                family_rankings.get(family.family_id, []),
                key=lambda item: (
                    0 if item.get("active", True) else 1,
                    item.get("pareto_rank") if item.get("pareto_rank") is not None else 999,
                    -float(item.get("fitness_score", 0.0) or 0.0),
                ),
            )
            if runtime_mode.is_full and ranked:
                self._retire_or_update_lineages(family, ranked, recent_actions=recent_actions)
                refreshed_ranked: List[Dict[str, Any]] = []
                for row in ranked:
                    lineage = self.registry.load_lineage(str(row["lineage_id"]))
                    if lineage is None:
                        continue
                    summary = lineage_summary_by_id.get(lineage.lineage_id)
                    if summary is not None:
                        summary["active"] = bool(lineage.active)
                        summary["loss_streak"] = int(lineage.loss_streak or 0)
                        summary["tweak_count"] = int(lineage.tweak_count or 0)
                        summary["max_tweaks"] = int(lineage.max_tweaks or 2)
                        summary["iteration_status"] = lineage.iteration_status
                        summary["role"] = lineage.role
                        summary["retired_at"] = lineage.retired_at
                        summary["retirement_reason"] = lineage.retirement_reason
                        summary["last_memory_id"] = lineage.last_memory_id
                    refreshed_ranked.append(dict(summary or row))
                ranked = sorted(
                    refreshed_ranked,
                    key=lambda item: (
                        0 if item.get("active", True) else 1,
                        item.get("pareto_rank") if item.get("pareto_rank") is not None else 999,
                        -float(item.get("fitness_score", 0.0) or 0.0),
                    ),
                )
                self._reclassify_family(family, ranked, recent_actions=recent_actions)
                refreshed_ranked = []
                for row in ranked:
                    lineage = self.registry.load_lineage(str(row["lineage_id"]))
                    if lineage is None:
                        continue
                    summary = lineage_summary_by_id.get(lineage.lineage_id)
                    if summary is not None:
                        summary["role"] = lineage.role
                        summary["active"] = bool(lineage.active)
                        summary["loss_streak"] = int(lineage.loss_streak or 0)
                        summary["tweak_count"] = int(lineage.tweak_count or 0)
                        summary["max_tweaks"] = int(lineage.max_tweaks or 2)
                        summary["iteration_status"] = lineage.iteration_status
                        summary["last_memory_id"] = lineage.last_memory_id
                    refreshed_ranked.append(dict(summary or row))
                ranked = sorted(
                    refreshed_ranked,
                    key=lambda item: (
                        0 if item.get("active", True) else 1,
                        item.get("pareto_rank") if item.get("pareto_rank") is not None else 999,
                        -float(item.get("fitness_score", 0.0) or 0.0),
                    ),
                )
            champion = next((item for item in ranked if item.get("active", True)), None) or {"lineage_id": family.champion_lineage_id, "current_stage": family.queue_stage}
            family.queue_stage = str(champion.get("current_stage") or family.queue_stage)
            family.last_cycle_at = utc_now_iso()
            self.registry.save_family(family)
            family_summaries.append(
                {
                    "family_id": family.family_id,
                    "label": family.label,
                    "explainer": family.explainer,
                    "queue_stage": family.queue_stage,
                    "champion": champion,
                    "lineage_count": len(ranked),
                    "active_lineage_count": sum(1 for item in ranked if item.get("active", True)),
                    "retired_lineage_count": len(family.retired_lineage_ids),
                    "shadow_challenger_ids": list(family.shadow_challenger_ids),
                    "paper_challenger_ids": list(family.paper_challenger_ids),
                    "target_portfolios": list(family.target_portfolios),
                    "budget_split": dict(family.budget_split),
                }
            )
        queue_entries = self._refresh_queue_entries(queue_entries)
        if runtime_mode.factory_influence_allowed:
            active_lineage_ids = {
                item["lineage_id"]
                for item in lineage_summaries
                if item.get("active", True)
            }
            live_loadable_manifests = [
                manifest.to_dict()
                for manifest in self.registry.manifests()
                if manifest.is_live_loadable() and manifest.lineage_id in active_lineage_ids
            ]
        else:
            live_loadable_manifests = []
        readiness_checks = [
            {
                "name": "connector_catalog_ready",
                "ok": all(item.get("ready") for item in connector_snapshots),
                "reason": "All venue connector snapshots should resolve at least one local evidence source.",
            },
            {
                "name": "goldfish_workspaces_ready",
                "ok": all(item.get("ready") for item in workspace_status.values()),
                "reason": "Each family should have a Goldfish sidecar workspace scaffold.",
            },
            {
                "name": "no_live_without_human_signoff",
                "ok": all(bool(item.get("approved_by")) and bool(item.get("approved_at")) for item in live_loadable_manifests),
                "reason": "Any live-loadable manifest must include explicit human approval metadata.",
            },
            {
                "name": "agentic_factory_runtime_mode",
                "ok": True,
                "reason": (
                    "Runtime mode is full."
                    if runtime_mode.is_full
                    else "Runtime mode is cost_saver, so token-consuming agentic work and new manifest publication are paused intentionally."
                ),
            },
        ]
        readiness_status = "research_only"
        if any(summary.get("strict_gate_pass") for summary in lineage_summaries):
            readiness_status = "paper_validating"
        if live_loadable_manifests:
            readiness_status = "live_ready"
        journal = FactoryJournal(
            active_goal="Promote only reproducible, net-of-costs lineages through paper gates and human-approved live manifests.",
            recent_actions=(self.registry.read_journal().recent_actions + recent_actions)[-20:],
        )
        self.registry.write_journal(journal)
        learning_memory = [memory.to_dict() for memory in self.registry.learning_memories(limit=20)]
        state = {
            "portfolio_id": getattr(config, "RESEARCH_FACTORY_PORTFOLIO_ID", "research_factory"),
            "running": True,
            "mode": "research",
            "status": "running",
            "explainer": "Research-only control plane for multi-family strategy discovery, evaluation, and approval-gated promotion.",
            "cycle_count": self._cycle_count,
            "last_cycle_at": utc_now_iso(),
            "budget_policy": {
                "split": _budget_split(),
                "max_shadow_challengers_per_family": 5,
                "max_paper_challengers_per_family": 2,
                "compute_posture": "local_first_hybrid",
            },
            "agent_roles": _factory_roles(),
            "scientific_researchers": _scientific_researchers(),
            "connectors": connector_snapshots,
            "goldfish": {
                "mode": "sidecar",
                "workspaces": workspace_status,
            },
            "families": family_summaries,
            "lineages": lineage_summaries,
            "manifests": {
                "pending": [manifest.to_dict() for manifest in self.registry.manifests() if not manifest.is_live_loadable()],
                "live_loadable": live_loadable_manifests,
            },
            "queue": [
                entry.to_dict()
                for entry in sorted(
                    queue_entries,
                    key=lambda item: (item.priority, item.family_id, item.lineage_id),
                )
            ],
            "learning_memory": learning_memory,
            "readiness": {
                "status": readiness_status,
                "blockers": (
                    ["human_signoff_required_for_live"]
                    if readiness_status != "live_ready" and not live_loadable_manifests and not runtime_mode.is_cost_saver
                    else []
                ),
                "warnings": ["agentic_factory_cost_saver"] if runtime_mode.is_cost_saver else [],
                "checks": readiness_checks,
                "score_pct": round((sum(1 for item in readiness_checks if item["ok"]) / len(readiness_checks)) * 100.0, 2),
                "eta_to_readiness": (
                    "agentic_tokens_paused"
                    if runtime_mode.is_cost_saver
                    else ("human_signoff_required" if readiness_status == "paper_validating" else "research_continuous")
                ),
            },
            "research_summary": {
                "family_count": len(family_summaries),
                "lineage_count": len(lineage_summaries),
                "active_lineage_count": sum(1 for item in lineage_summaries if item.get("active", True)),
                "artifact_backed_lineage_count": sum(1 for item in lineage_summaries if item.get("latest_artifact_package")),
                "challenge_count": sum(
                    1
                    for item in lineage_summaries
                    if item.get("role") in {LineageRole.SHADOW_CHALLENGER.value, LineageRole.PAPER_CHALLENGER.value}
                ),
                "agent_generated_lineage_count": sum(1 for item in lineage_summaries if item.get("hypothesis_origin") == "scientific_agent_collective"),
                "tweaked_lineage_count": sum(1 for item in lineage_summaries if int(item.get("tweak_count", 0) or 0) > 0),
                "retired_lineage_count": sum(1 for item in lineage_summaries if not item.get("active", True)),
                "learning_memory_count": len(learning_memory),
                "paper_pnl": round(total_paper_pnl, 4),
                "ready_for_canary": sum(1 for item in lineage_summaries if item.get("current_stage") == PromotionStage.CANARY_READY.value),
                "live_loadable_manifest_count": len(live_loadable_manifests),
                "manifest_publication_paused": runtime_mode.is_cost_saver,
            },
        }
        state = self._with_runtime_mode(state)
        self.registry.write_state(state)
        self._last_state = state
        return state

    def get_state(self) -> Dict[str, Any]:
        if not self._last_state:
            self._last_state = self.registry.read_state()
        return dict(self._last_state)
