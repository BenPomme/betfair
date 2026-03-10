from __future__ import annotations

import config
from factory.contracts import AcceptedStrategyManifest
from factory.manifests import candidate_context_refs_for_portfolio, live_manifest_refs_for_portfolio
from factory.orchestrator import FactoryOrchestrator
from factory.registry import FactoryRegistry
from factory.runtime_mode import AgenticFactoryRuntimeMode, normalize_agentic_factory_mode


def test_runtime_mode_normalizes_and_exposes_flags():
    assert normalize_agentic_factory_mode("FULL") == "full"
    assert normalize_agentic_factory_mode("cost_saver") == "cost_saver"
    assert normalize_agentic_factory_mode("unknown") == "full"

    full = AgenticFactoryRuntimeMode("full")
    cost_saver = AgenticFactoryRuntimeMode("cost_saver")
    hard_stop = AgenticFactoryRuntimeMode("hard_stop")

    assert full.is_full is True
    assert full.agentic_tokens_allowed is True
    assert full.factory_influence_allowed is True

    assert cost_saver.is_cost_saver is True
    assert cost_saver.agentic_tokens_allowed is False
    assert cost_saver.factory_influence_allowed is True

    assert hard_stop.is_hard_stop is True
    assert hard_stop.agentic_tokens_allowed is False
    assert hard_stop.factory_influence_allowed is False


def test_live_manifest_refs_are_hidden_when_factory_is_hard_stopped(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FACTORY_ROOT", str(tmp_path / "factory"))
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
    registry.approve_manifest("manifest-a", approved_by="operator")

    monkeypatch.setattr(config, "AGENTIC_FACTORY_MODE", "cost_saver")
    assert len(live_manifest_refs_for_portfolio("betfair_core")) == 1

    monkeypatch.setattr(config, "AGENTIC_FACTORY_MODE", "hard_stop")
    assert live_manifest_refs_for_portfolio("betfair_core") == []


def test_candidate_contexts_are_visible_in_cost_saver_and_hidden_in_hard_stop(tmp_path, monkeypatch):
    project_root = tmp_path / "repo"
    project_root.mkdir(parents=True, exist_ok=True)
    portfolio_root = tmp_path / "portfolio_state"
    factory_root = tmp_path / "factory"
    goldfish_root = project_root / "research" / "goldfish"

    monkeypatch.setattr(config, "PORTFOLIO_STATE_ROOT", str(portfolio_root))
    monkeypatch.setattr(config, "FACTORY_ROOT", str(factory_root))
    monkeypatch.setattr(config, "FACTORY_GOLDFISH_ROOT", str(goldfish_root))

    from tests.unit.test_factory_orchestrator import _prepare_factory_inputs

    _prepare_factory_inputs(project_root)
    orchestrator = FactoryOrchestrator(project_root)
    orchestrator.run_cycle()

    monkeypatch.setattr(config, "AGENTIC_FACTORY_MODE", "cost_saver")
    assert candidate_context_refs_for_portfolio("betfair_core")

    monkeypatch.setattr(config, "AGENTIC_FACTORY_MODE", "hard_stop")
    assert candidate_context_refs_for_portfolio("betfair_core") == []
