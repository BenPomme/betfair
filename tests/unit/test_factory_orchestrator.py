from __future__ import annotations

import json
from pathlib import Path

import config
from factory.orchestrator import FactoryOrchestrator
from portfolio.accounting import build_strategy_account
from portfolio.runner_base import PortfolioRunnerBase
from portfolio.state_store import PortfolioStateStore
from portfolio.types import PortfolioRunnerSpec


class _DummyRunner(PortfolioRunnerBase):
    def run(self) -> None:
        raise NotImplementedError


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _seed_prediction_examples(project_root: Path, *, count: int = 720) -> None:
    rows: list[dict] = []
    for idx in range(count):
        timestamp = f"2026-03-{1 + (idx // 24):02d}T{idx % 24:02d}:00:00+00:00"
        signal = 1 if idx % 6 in {0, 1, 3, 4} else 0
        base_prob = 0.47 + (0.02 if idx % 9 in {0, 1, 2} else -0.015)
        odds = 2.08 if signal else 1.92
        label = 1 if (signal and idx % 5 != 0) or ((not signal) and idx % 11 == 0) else 0
        rows.append(
            {
                "timestamp": timestamp,
                "base_prob": base_prob,
                "odds": odds,
                "label": label,
                "spread_mean": 0.18 + ((idx % 7) * 0.01),
                "imbalance": 0.65 if signal else -0.55,
                "depth_total_eur": 900.0 + (idx % 8) * 35.0,
                "price_velocity": 0.012 if signal else -0.009,
                "short_volatility": 0.03 + ((idx % 5) * 0.002),
                "time_to_start_sec": float(21600 - ((idx % 24) * 900)),
                "in_play": 1.0 if idx % 10 >= 6 else 0.0,
                "weighted_spread": 0.24 + ((idx % 6) * 0.015),
                "lay_back_ratio": 1.65 if signal else 0.72,
                "top_of_book_concentration": 0.58 + ((idx % 4) * 0.03),
                "selection_count": 2.0 + float(idx % 3),
            }
        )
    _write_jsonl(project_root / "data/prediction/online_examples_implied_market_1.jsonl", rows)


def _seed_portfolio(store: PortfolioStateStore, *, currency: str, starting_balance: float, realized_pnl: float, trade_count: int) -> None:
    balance_history = [
        {"ts": "2026-03-01T00:00:00Z", "balance": starting_balance},
        {"ts": "2026-03-15T00:00:00Z", "balance": starting_balance + realized_pnl + 10.0},
        {"ts": "2026-03-30T00:00:00Z", "balance": starting_balance + realized_pnl},
    ]
    store.write_account(
        build_strategy_account(
            portfolio_id=store.portfolio_id,
            currency=currency,
            starting_balance=starting_balance,
            current_balance=starting_balance + realized_pnl,
            realized_pnl=realized_pnl,
            trade_count=trade_count,
            balance_history=balance_history,
        )
    )
    store.write_trades(
        [
            {
                "trade_id": f"{store.portfolio_id}-{idx}",
                "status": "CLOSED",
                "net_pnl_usd": 1.0,
            }
            for idx in range(trade_count)
        ]
    )


def _prepare_factory_inputs(project_root: Path) -> None:
    for rel_path in [
        "data/funding_history",
        "data/funding_models",
        "data/candidates",
        "data/prediction",
        "data/state",
        "data/portfolios/betfair_core",
        "data/portfolios/polymarket_quantum_fold",
    ]:
        (project_root / rel_path).mkdir(parents=True, exist_ok=True)

    (project_root / "data/funding_models/funding_predictor_meta.json").write_text("{}", encoding="utf-8")
    (project_root / "data/portfolios/betfair_core/runtime").mkdir(parents=True, exist_ok=True)
    (project_root / "data/portfolios/betfair_core/runtime/polymarket_binary_research_state.json").write_text(
        "{}",
        encoding="utf-8",
    )
    _write_jsonl(
        project_root / "data/funding/experiments.jsonl",
        [{"metrics": {"rolling_200": {"settled": 120, "roi_pct": 6.5, "brier_lift_abs": 0.025}}}],
    )
    _write_jsonl(
        project_root / "data/prediction/experiments.jsonl",
        [{"metrics": {"rolling_200": {"settled": 120, "roi_pct": 6.2, "brier_lift_abs": 0.018}}}],
    )
    _seed_prediction_examples(project_root)

    _seed_portfolio(PortfolioStateStore("contrarian_legacy"), currency="USD", starting_balance=1000.0, realized_pnl=60.0, trade_count=60)
    _seed_portfolio(PortfolioStateStore("cascade_alpha"), currency="USD", starting_balance=1000.0, realized_pnl=55.0, trade_count=60)
    _seed_portfolio(PortfolioStateStore("betfair_core"), currency="EUR", starting_balance=1000.0, realized_pnl=30.0, trade_count=30)
    _seed_portfolio(PortfolioStateStore("polymarket_quantum_fold"), currency="USD", starting_balance=1000.0, realized_pnl=58.0, trade_count=60)


def test_factory_orchestrator_publishes_pending_manifests_and_promotes_after_approval(tmp_path, monkeypatch):
    project_root = tmp_path / "repo"
    project_root.mkdir(parents=True, exist_ok=True)
    portfolio_root = tmp_path / "portfolio_state"
    factory_root = tmp_path / "factory"
    goldfish_root = project_root / "research" / "goldfish"

    monkeypatch.setattr(config, "PORTFOLIO_STATE_ROOT", str(portfolio_root))
    monkeypatch.setattr(config, "FACTORY_ROOT", str(factory_root))
    monkeypatch.setattr(config, "FACTORY_GOLDFISH_ROOT", str(goldfish_root))
    _prepare_factory_inputs(project_root)

    orchestrator = FactoryOrchestrator(project_root)

    first_state = orchestrator.run_cycle()

    assert first_state["research_summary"]["family_count"] == 5
    assert first_state["manifests"]["pending"]
    assert first_state["manifests"]["live_loadable"] == []
    assert any(lineage["current_stage"] == "live_ready" for lineage in first_state["lineages"])
    assert any("budget_weight_pct" in lineage for lineage in first_state["lineages"])
    assert any(manifest["artifact_refs"].get("package") for manifest in first_state["manifests"]["pending"])

    selected_manifest = next(
        manifest
        for manifest in first_state["manifests"]["pending"]
        if manifest["artifact_refs"].get("package")
    )
    pending_manifest_id = selected_manifest["manifest_id"]
    approved = orchestrator.registry.approve_manifest(pending_manifest_id, approved_by="operator")

    assert approved is not None
    assert approved.is_live_loadable() is True

    second_state = orchestrator.run_cycle()

    assert second_state["manifests"]["live_loadable"]
    assert any(lineage["current_stage"] == "approved_live" for lineage in second_state["lineages"])

    runner = _DummyRunner(
        PortfolioRunnerSpec(
            portfolio_id=selected_manifest["portfolio_targets"][0],
            label="Approved Manifest Portfolio",
            category="research_validation",
            control_mode="local_managed",
            currency="USD",
            initial_balance=1000.0,
        )
    )
    snapshot = runner.build_config_snapshot()

    assert snapshot["factory_live_manifest_count"] >= 1
    assert any(item["manifest_id"] == pending_manifest_id for item in snapshot["factory_live_manifests"])
    assert snapshot["factory_live_manifests"][0]["package"]["package_found"] is True
    assert snapshot["factory_live_artifact_packages"]
    assert snapshot["factory_live_contexts"]
    assert snapshot["factory_live_artifact_payloads"]
    assert snapshot["factory_live_strategy_families"]


def test_factory_orchestrator_creates_bounded_challengers_and_queue_entries(tmp_path, monkeypatch):
    project_root = tmp_path / "repo"
    project_root.mkdir(parents=True, exist_ok=True)
    portfolio_root = tmp_path / "portfolio_state"
    factory_root = tmp_path / "factory"
    goldfish_root = project_root / "research" / "goldfish"

    monkeypatch.setattr(config, "PORTFOLIO_STATE_ROOT", str(portfolio_root))
    monkeypatch.setattr(config, "FACTORY_ROOT", str(factory_root))
    monkeypatch.setattr(config, "FACTORY_GOLDFISH_ROOT", str(goldfish_root))
    _prepare_factory_inputs(project_root)

    orchestrator = FactoryOrchestrator(project_root)

    state = orchestrator.run_cycle()

    challengers = [
        row for row in state["lineages"]
        if row["role"] in {"shadow_challenger", "paper_challenger"} and row.get("parent_lineage_id") is not None
    ]
    prediction_lineages = [
        row for row in state["lineages"] if row["family_id"] == "betfair_prediction_value_league"
    ]
    funding_lineages = [
        row for row in state["lineages"] if row["family_id"] == "binance_funding_contrarian"
    ]

    assert state["research_summary"]["challenge_count"] >= 4
    assert state["research_summary"]["artifact_backed_lineage_count"] >= 2
    assert state["research_summary"]["agent_generated_lineage_count"] >= 1
    assert challengers
    assert state["queue"]
    assert all(item["family_id"] for item in state["queue"])
    assert prediction_lineages
    assert funding_lineages
    assert any(lineage["latest_artifact_package"] for lineage in prediction_lineages)
    assert any(lineage["latest_artifact_package"] for lineage in funding_lineages)

    challenger = challengers[0]
    genome = orchestrator.registry.load_genome(challenger["lineage_id"])
    hypothesis = orchestrator.registry.load_hypothesis(challenger["lineage_id"])

    assert challenger["parent_lineage_id"] is not None
    assert genome is not None
    assert hypothesis is not None
    assert challenger["max_tweaks"] == 2
    assert challenger["lead_agent_role"]
    assert challenger["hypothesis_origin"] == "scientific_agent_collective"
    assert hypothesis.origin == "scientific_agent_collective"
    assert hypothesis.collaborating_agent_roles
    assert genome.parameters["selected_horizon_seconds"] in genome.mutation_bounds.horizons_seconds
    assert genome.parameters["selected_feature_subset"] in genome.mutation_bounds.feature_subsets
    assert genome.parameters["selected_model_class"] in genome.mutation_bounds.model_classes
    min_edge_bounds = genome.mutation_bounds.execution_thresholds["min_edge"]
    assert min_edge_bounds[0] <= genome.parameters["selected_min_edge"] <= min_edge_bounds[-1]

    prediction_experiment = orchestrator.registry.load_experiment("betfair_prediction_value_league:champion")
    latest_run = dict((prediction_experiment.expected_outputs or {}).get("latest_run") or {})
    assert latest_run["mode"] == "prediction_walkforward"
    assert Path(latest_run["package_path"]).exists()

    funding_experiment = orchestrator.registry.load_experiment("binance_funding_contrarian:champion")
    funding_run = dict((funding_experiment.expected_outputs or {}).get("latest_run") or {})
    assert funding_run["mode"] == "funding_contrarian"
    assert Path(funding_run["package_path"]).exists()


def test_factory_orchestrator_retires_repeat_losers_after_three_cycles(tmp_path, monkeypatch):
    project_root = tmp_path / "repo"
    project_root.mkdir(parents=True, exist_ok=True)
    portfolio_root = tmp_path / "portfolio_state"
    factory_root = tmp_path / "factory"
    goldfish_root = project_root / "research" / "goldfish"

    monkeypatch.setattr(config, "PORTFOLIO_STATE_ROOT", str(portfolio_root))
    monkeypatch.setattr(config, "FACTORY_ROOT", str(factory_root))
    monkeypatch.setattr(config, "FACTORY_GOLDFISH_ROOT", str(goldfish_root))
    _prepare_factory_inputs(project_root)

    orchestrator = FactoryOrchestrator(project_root)

    for _ in range(4):
        state = orchestrator.run_cycle()

    retired = [row for row in state["lineages"] if row.get("active") is False]

    assert retired
    assert any(row.get("retirement_reason") for row in retired)
    assert state["research_summary"]["retired_lineage_count"] >= 1
    assert state["research_summary"]["learning_memory_count"] >= 1
    assert state["research_summary"]["tweaked_lineage_count"] >= 1
    assert any(int(row.get("tweak_count", 0) or 0) >= 2 for row in retired)
    assert any(memory["outcome"] == "retired_underperformance" for memory in state["learning_memory"])


def test_factory_orchestrator_cost_saver_preserves_learning_but_freezes_manifests_and_promotions(tmp_path, monkeypatch):
    project_root = tmp_path / "repo"
    project_root.mkdir(parents=True, exist_ok=True)
    portfolio_root = tmp_path / "portfolio_state"
    factory_root = tmp_path / "factory"
    goldfish_root = project_root / "research" / "goldfish"

    monkeypatch.setattr(config, "PORTFOLIO_STATE_ROOT", str(portfolio_root))
    monkeypatch.setattr(config, "FACTORY_ROOT", str(factory_root))
    monkeypatch.setattr(config, "FACTORY_GOLDFISH_ROOT", str(goldfish_root))
    monkeypatch.setattr(config, "AGENTIC_FACTORY_MODE", "cost_saver")
    _prepare_factory_inputs(project_root)

    orchestrator = FactoryOrchestrator(project_root)

    state = orchestrator.run_cycle()

    assert state["agentic_factory_mode"] == "cost_saver"
    assert state["agentic_tokens_allowed"] is False
    assert state["factory_influence_allowed"] is True
    assert state["research_summary"]["manifest_publication_paused"] is True
    assert state["manifests"]["pending"] == []
    assert state["manifests"]["live_loadable"] == []
    assert state["lineages"]
    assert all(lineage["current_stage"] == "idea" for lineage in state["lineages"])
    assert orchestrator.registry.evaluations("binance_funding_contrarian:champion")
    assert any(lineage["latest_artifact_package"] for lineage in state["lineages"] if lineage["family_id"] == "betfair_prediction_value_league")
    assert any(lineage["latest_artifact_package"] for lineage in state["lineages"] if lineage["family_id"] == "binance_funding_contrarian")
    assert state["research_summary"]["agent_generated_lineage_count"] == 0


def test_factory_orchestrator_hard_stop_returns_paused_state_without_new_activity(tmp_path, monkeypatch):
    project_root = tmp_path / "repo"
    project_root.mkdir(parents=True, exist_ok=True)
    factory_root = tmp_path / "factory"
    goldfish_root = project_root / "research" / "goldfish"

    monkeypatch.setattr(config, "FACTORY_ROOT", str(factory_root))
    monkeypatch.setattr(config, "FACTORY_GOLDFISH_ROOT", str(goldfish_root))
    monkeypatch.setattr(config, "AGENTIC_FACTORY_MODE", "hard_stop")

    orchestrator = FactoryOrchestrator(project_root)

    state = orchestrator.run_cycle()

    assert state["status"] == "paused"
    assert state["running"] is False
    assert state["agentic_factory_mode"] == "hard_stop"
    assert state["factory_influence_allowed"] is False
    assert "agentic_factory_hard_stopped" in (state.get("readiness") or {}).get("blockers", [])
    assert orchestrator.registry.evaluations() == []
