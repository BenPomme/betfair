from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from factory.contracts import FactoryFamily, LearningMemoryEntry, MutationBounds, ResearchHypothesis, StrategyGenome


@dataclass(frozen=True)
class _DomainAgentProfile:
    domain: str
    role: str
    angle: str
    feature_subset: str
    model_class: str
    horizon_seconds: int
    lookback_hours: float
    min_edge: float
    stake_fraction: float


@dataclass(frozen=True)
class ScientificAgentProposal:
    proposal_id: str
    family_id: str
    title: str
    thesis: str
    scientific_domains: List[str]
    lead_agent_role: str
    collaborating_agent_roles: List[str]
    parameter_overrides: Dict[str, Any]
    budget_bucket: str
    origin: str = "scientific_agent_collective"
    agent_notes: List[str] = field(default_factory=list)


_DOMAIN_PROFILES: Dict[str, _DomainAgentProfile] = {
    "econometrics": _DomainAgentProfile(
        domain="econometrics",
        role="Econometrics Researcher",
        angle="stability-aware factor combinations and regime-conditioned edge persistence",
        feature_subset="baseline",
        model_class="logit",
        horizon_seconds=1800,
        lookback_hours=72.0,
        min_edge=0.03,
        stake_fraction=0.03,
    ),
    "microstructure": _DomainAgentProfile(
        domain="microstructure",
        role="Microstructure Researcher",
        angle="order-book stress, queue pressure, and execution-quality asymmetry",
        feature_subset="microstructure",
        model_class="gbdt",
        horizon_seconds=600,
        lookback_hours=36.0,
        min_edge=0.025,
        stake_fraction=0.025,
    ),
    "bayesian_causal": _DomainAgentProfile(
        domain="bayesian_causal",
        role="Bayesian/Causal Researcher",
        angle="causal priors and uncertainty-aware edge filtering",
        feature_subset="cross_science",
        model_class="logit",
        horizon_seconds=1800,
        lookback_hours=96.0,
        min_edge=0.035,
        stake_fraction=0.03,
    ),
    "statistical_physics": _DomainAgentProfile(
        domain="statistical_physics",
        role="Statistical Physics Researcher",
        angle="phase-transition detection and cascade susceptibility before dislocations accelerate",
        feature_subset="regime",
        model_class="transformer",
        horizon_seconds=120,
        lookback_hours=24.0,
        min_edge=0.04,
        stake_fraction=0.02,
    ),
    "network_epidemiology": _DomainAgentProfile(
        domain="network_epidemiology",
        role="Network/Epidemiology Researcher",
        angle="contagion paths, information spread, and lag propagation across correlated books",
        feature_subset="cross_science",
        model_class="transformer",
        horizon_seconds=600,
        lookback_hours=48.0,
        min_edge=0.035,
        stake_fraction=0.025,
    ),
    "ecology_evolution": _DomainAgentProfile(
        domain="ecology_evolution",
        role="Ecology/Evolution Researcher",
        angle="survival pressure, niche competition, and adaptive response to decaying edges",
        feature_subset="cross_science",
        model_class="gbdt",
        horizon_seconds=600,
        lookback_hours=84.0,
        min_edge=0.03,
        stake_fraction=0.025,
    ),
    "information_theory": _DomainAgentProfile(
        domain="information_theory",
        role="Information Theory Researcher",
        angle="entropy compression, surprise detection, and cross-feed information gain",
        feature_subset="cross_science",
        model_class="gbdt",
        horizon_seconds=600,
        lookback_hours=48.0,
        min_edge=0.03,
        stake_fraction=0.025,
    ),
    "control_rl": _DomainAgentProfile(
        domain="control_rl",
        role="Control/RL Researcher",
        angle="feedback-aware thresholding and action policies that stay robust under execution drift",
        feature_subset="regime",
        model_class="tft",
        horizon_seconds=600,
        lookback_hours=60.0,
        min_edge=0.035,
        stake_fraction=0.02,
    ),
    "game_theory_behavioral": _DomainAgentProfile(
        domain="game_theory_behavioral",
        role="Game Theory/Behavioral Researcher",
        angle="opponent adaptation, crowd reflexivity, and exploitability of consensus breakdowns",
        feature_subset="cross_science",
        model_class="rules",
        horizon_seconds=1800,
        lookback_hours=60.0,
        min_edge=0.04,
        stake_fraction=0.02,
    ),
    "signal_processing_neuroscience": _DomainAgentProfile(
        domain="signal_processing_neuroscience",
        role="Signal Processing/Neuroscience Researcher",
        angle="multi-timescale filtering, burst detection, and low-latency state transitions",
        feature_subset="microstructure",
        model_class="gbdt",
        horizon_seconds=120,
        lookback_hours=24.0,
        min_edge=0.03,
        stake_fraction=0.02,
    ),
}


_FAMILY_SWARMS: Dict[str, List[List[str]]] = {
    "binance_funding_contrarian": [
        ["econometrics", "microstructure", "ecology_evolution"],
        ["bayesian_causal", "information_theory", "control_rl"],
        ["statistical_physics", "network_epidemiology", "microstructure"],
    ],
    "binance_cascade_regime": [
        ["statistical_physics", "network_epidemiology", "signal_processing_neuroscience"],
        ["microstructure", "control_rl", "information_theory"],
        ["econometrics", "statistical_physics", "control_rl"],
    ],
    "betfair_prediction_value_league": [
        ["bayesian_causal", "information_theory", "econometrics"],
        ["microstructure", "signal_processing_neuroscience", "control_rl"],
        ["game_theory_behavioral", "bayesian_causal", "ecology_evolution"],
    ],
    "betfair_information_lag": [
        ["network_epidemiology", "game_theory_behavioral", "information_theory"],
        ["signal_processing_neuroscience", "microstructure", "bayesian_causal"],
        ["ecology_evolution", "network_epidemiology", "control_rl"],
    ],
    "polymarket_cross_venue": [
        ["information_theory", "signal_processing_neuroscience", "network_epidemiology"],
        ["microstructure", "game_theory_behavioral", "control_rl"],
        ["bayesian_causal", "information_theory", "econometrics"],
    ],
}

_FAMILY_TUNING: Dict[str, Dict[str, Any]] = {
    "binance_funding_contrarian": {
        "feature_subset": "regime",
        "model_class": "gbdt",
        "horizon_seconds": 600,
        "lookback_hours": 96.0,
        "min_edge": 0.05,
        "stake_fraction": 0.015,
    },
    "binance_cascade_regime": {
        "feature_subset": "microstructure",
        "model_class": "gbdt",
        "horizon_seconds": 120,
        "lookback_hours": 36.0,
        "min_edge": 0.06,
        "stake_fraction": 0.012,
    },
    "betfair_prediction_value_league": {
        "feature_subset": "microstructure",
        "model_class": "gbdt",
        "horizon_seconds": 1800,
        "lookback_hours": 120.0,
        "min_edge": 0.05,
        "stake_fraction": 0.015,
    },
    "betfair_information_lag": {
        "feature_subset": "cross_science",
        "model_class": "gbdt",
        "horizon_seconds": 600,
        "lookback_hours": 96.0,
        "min_edge": 0.055,
        "stake_fraction": 0.012,
    },
    "polymarket_cross_venue": {
        "feature_subset": "cross_science",
        "model_class": "gbdt",
        "horizon_seconds": 600,
        "lookback_hours": 84.0,
        "min_edge": 0.055,
        "stake_fraction": 0.012,
    },
}


class ScientificStrategyInventor:
    def generate_proposal(
        self,
        *,
        family: FactoryFamily,
        champion_hypothesis: ResearchHypothesis | None,
        champion_genome: StrategyGenome,
        learning_memory: Sequence[LearningMemoryEntry],
        cycle_count: int,
        proposal_index: int,
    ) -> ScientificAgentProposal:
        swarms = list(_FAMILY_SWARMS.get(family.family_id) or [["econometrics", "microstructure", "information_theory"]])
        recent_signatures = {
            tuple(sorted(memory.scientific_domains))
            for memory in learning_memory
            if memory.outcome.startswith("retired")
        }
        selected_domains: List[str] = []
        start_index = (cycle_count + proposal_index - 1) % max(1, len(swarms))
        for offset in range(len(swarms)):
            candidate = list(swarms[(start_index + offset) % len(swarms)])
            if tuple(sorted(candidate)) not in recent_signatures:
                selected_domains = candidate
                break
        if not selected_domains:
            selected_domains = list(swarms[start_index % len(swarms)])
        profiles = [_DOMAIN_PROFILES[domain] for domain in selected_domains]
        feature_subset = self._dominant_value([profile.feature_subset for profile in profiles], fallback="baseline")
        model_class = self._dominant_value([profile.model_class for profile in profiles], fallback="logit")
        horizons = [profile.horizon_seconds for profile in profiles]
        lookbacks = [profile.lookback_hours for profile in profiles]
        min_edges = [profile.min_edge for profile in profiles]
        stake_fractions = [profile.stake_fraction for profile in profiles]
        bounds = champion_genome.mutation_bounds
        tuning = dict(_FAMILY_TUNING.get(family.family_id) or {})
        if tuning:
            feature_subset = str(tuning.get("feature_subset", feature_subset) or feature_subset)
            model_class = str(tuning.get("model_class", model_class) or model_class)
        memory_adjustments = self._memory_adjustments(learning_memory)
        parameter_overrides = {
            "selected_horizon_seconds": self._nearest_choice(
                bounds.horizons_seconds,
                int(tuning.get("horizon_seconds") or int(round(sum(horizons) / len(horizons)))),
            ) or horizons[0],
            "selected_feature_subset": self._allowed_choice(bounds.feature_subsets, feature_subset, fallback="baseline"),
            "selected_model_class": self._allowed_choice(bounds.model_classes, model_class, fallback="logit"),
            "selected_lookback_hours": self._clip_range(
                bounds.hyperparameter_ranges.get("lookback_hours"),
                float(tuning.get("lookback_hours") or (sum(lookbacks) / len(lookbacks))),
                fallback=48.0,
            ),
            "selected_min_edge": self._clip_range(
                bounds.execution_thresholds.get("min_edge"),
                max(float(tuning.get("min_edge", 0.0) or 0.0), (sum(min_edges) / len(min_edges)) + memory_adjustments["edge_bump"]),
                fallback=0.03,
            ),
            "selected_stake_fraction": self._clip_range(
                bounds.execution_thresholds.get("stake_fraction"),
                min(float(tuning.get("stake_fraction", 1.0) or 1.0), max(0.01, (sum(stake_fractions) / len(stake_fractions)) - memory_adjustments["stake_reduction"])),
                fallback=0.03,
            ),
            "selected_learning_rate": self._learning_rate_for_model(
                self._allowed_choice(bounds.model_classes, model_class, fallback="logit"),
                bounds,
            ),
        }
        if memory_adjustments["prefer_information"]:
            parameter_overrides["selected_feature_subset"] = self._allowed_choice(
                bounds.feature_subsets,
                "microstructure" if "microstructure" in bounds.feature_subsets else "cross_science",
                fallback=parameter_overrides["selected_feature_subset"],
            )
        if memory_adjustments["avoid_high_failure_models"]:
            parameter_overrides["selected_model_class"] = self._allowed_choice(
                bounds.model_classes,
                "gbdt",
                fallback=parameter_overrides["selected_model_class"],
            )
        lead_profile = profiles[0]
        collaborator_roles = [profile.role for profile in profiles[1:]]
        thesis_parts = [profile.angle for profile in profiles]
        thesis = (
            f"{family.thesis} This variant fuses {lead_profile.role.lower()} guidance with "
            f"{', '.join(role.lower() for role in collaborator_roles)} to test "
            f"{'; '.join(thesis_parts)}."
        )
        memory_hint = self._memory_hint(learning_memory)
        notes = [
            f"lead_agent={lead_profile.role}",
            f"collaborators={','.join(collaborator_roles) or 'none'}",
        ]
        if memory_hint:
            notes.append(memory_hint)
        return ScientificAgentProposal(
            proposal_id=f"{family.family_id}:proposal:{proposal_index}",
            family_id=family.family_id,
            title=f"{family.label} Scientific Collaboration {proposal_index}",
            thesis=thesis,
            scientific_domains=selected_domains,
            lead_agent_role=lead_profile.role,
            collaborating_agent_roles=collaborator_roles,
            parameter_overrides=parameter_overrides,
            budget_bucket=self._budget_bucket(selected_domains, family.budget_split),
            agent_notes=notes,
        )

    def _budget_bucket(self, domains: Sequence[str], budget_split: Dict[str, float]) -> str:
        if any(domain in {"statistical_physics", "network_epidemiology", "game_theory_behavioral"} for domain in domains):
            return "moonshot" if budget_split.get("moonshot", 0.0) > 0.0 else "adjacent"
        if any(domain in {"control_rl", "information_theory", "signal_processing_neuroscience"} for domain in domains):
            return "adjacent" if budget_split.get("adjacent", 0.0) > 0.0 else "incumbent"
        return "incumbent"

    def _memory_hint(self, learning_memory: Sequence[LearningMemoryEntry]) -> str:
        if not learning_memory:
            return ""
        latest = learning_memory[-1]
        if latest.recommendations:
            return f"memory_hint={latest.recommendations[0]}"
        return f"memory_hint=avoid repeating {','.join(latest.scientific_domains)} without a structural change"

    def _memory_adjustments(self, learning_memory: Sequence[LearningMemoryEntry]) -> Dict[str, Any]:
        edge_bump = 0.0
        stake_reduction = 0.0
        prefer_information = False
        avoid_high_failure_models = False
        for memory in learning_memory[-5:]:
            recs = " | ".join(memory.recommendations).lower()
            blockers = " | ".join(memory.blockers).lower()
            if "tighten edge thresholds" in recs:
                edge_bump = max(edge_bump, 0.01)
            if "reduce stake fraction" in recs:
                stake_reduction = max(stake_reduction, 0.01)
            if "prefer higher-information or microstructure features" in recs:
                prefer_information = True
            if "failure_rate_above_15pct" in blockers:
                avoid_high_failure_models = True
        return {
            "edge_bump": edge_bump,
            "stake_reduction": stake_reduction,
            "prefer_information": prefer_information,
            "avoid_high_failure_models": avoid_high_failure_models,
        }

    def _dominant_value(self, values: Sequence[str], *, fallback: str) -> str:
        if not values:
            return fallback
        counts = Counter(values)
        return counts.most_common(1)[0][0]

    def _nearest_choice(self, options: Sequence[int], value: int) -> int | None:
        if not options:
            return None
        return min((int(option) for option in options), key=lambda option: abs(option - int(value)))

    def _allowed_choice(self, options: Sequence[str], candidate: str, *, fallback: str) -> str:
        normalized = [str(option) for option in options]
        if candidate in normalized:
            return candidate
        if fallback in normalized:
            return fallback
        return normalized[0] if normalized else fallback

    def _clip_range(self, bounds: Sequence[float] | None, value: float, *, fallback: float) -> float:
        if not bounds:
            return round(float(value if value else fallback), 6)
        low = float(bounds[0])
        high = float(bounds[-1])
        return round(min(max(float(value), low), high), 6)

    def _learning_rate_for_model(self, model_class: str, bounds: MutationBounds) -> float:
        preferred = {
            "logit": 0.02,
            "gbdt": 0.04,
            "tft": 0.01,
            "transformer": 0.008,
            "rules": 0.015,
        }.get(str(model_class or "logit").lower(), 0.02)
        return self._clip_range(bounds.hyperparameter_ranges.get("learning_rate"), preferred, fallback=0.02)
