from __future__ import annotations

import importlib
from typing import Dict, List

import config
from factory.runtime_mode import research_factory_start_blocker
from portfolio.types import PortfolioRunnerSpec


_REGISTRY: Dict[str, PortfolioRunnerSpec] = {
    "betfair_core": PortfolioRunnerSpec(
        portfolio_id="betfair_core",
        label="Betfair Core",
        category="betfair",
        control_mode="local_managed",
        currency="EUR",
        initial_balance=float(config.INITIAL_BALANCE_EUR),
        runner_path="portfolio.runners.betfair_runner:BetfairPortfolioRunner",
        autostart=False,
        description="Primary Betfair arbitrage portfolio.",
        ui_group="Betfair",
    ),
    "research_factory": PortfolioRunnerSpec(
        portfolio_id="research_factory",
        label="Research Factory",
        category="research_factory",
        control_mode="local_managed",
        currency="USD",
        initial_balance=float(getattr(config, "RESEARCH_FACTORY_INITIAL_BALANCE_USD", 100000.0)),
        runner_path="portfolio.runners.research_factory_runner:ResearchFactoryPortfolioRunner",
        autostart=False,
        enabled=bool(getattr(config, "RESEARCH_FACTORY_ENABLED", True)),
        description="Agentic strategy factory control plane with Goldfish sidecar workspaces and approval-gated manifest promotion.",
        ui_group="Research Factory",
    ),
    "hedge_validation": PortfolioRunnerSpec(
        portfolio_id="hedge_validation",
        label="Hedge Validation",
        category="crypto_hedge",
        control_mode="local_managed",
        currency="USD",
        initial_balance=float(getattr(config, "HEDGE_PORTFOLIO_INITIAL_BALANCE_USD", config.FUNDING_MAX_TOTAL_EXPOSURE_USD)),
        runner_path="portfolio.runners.hedge_runner:HedgeValidationPortfolioRunner",
        autostart=False,
        description="Strict hedge-only Binance funding validation.",
        ui_group="Hedge Validation",
    ),
    "hedge_research": PortfolioRunnerSpec(
        portfolio_id="hedge_research",
        label="Hedge Research",
        category="crypto_hedge",
        control_mode="local_managed",
        currency="USD",
        initial_balance=float(getattr(config, "HEDGE_RESEARCH_INITIAL_BALANCE_USD", 25000.0)),
        runner_path="portfolio.runners.hedge_research_runner:HedgeResearchPortfolioRunner",
        autostart=False,
        enabled=bool(getattr(config, "HEDGE_RESEARCH_ENABLED", True)),
        description="Looser funding research portfolio that trains the shared learner for strict validation.",
        ui_group="Hedge Research",
    ),
    "cascade_alpha": PortfolioRunnerSpec(
        portfolio_id="cascade_alpha",
        label="Cascade Alpha",
        category="crypto_alpha",
        control_mode="local_managed",
        currency="USD",
        initial_balance=float(getattr(config, "CASCADE_ALPHA_INITIAL_BALANCE_USD", 15000.0)),
        runner_path="portfolio.runners.cascade_alpha_runner:CascadeAlphaPortfolioRunner",
        autostart=False,
        enabled=bool(getattr(config, "CASCADE_ALPHA_ENABLED", False)),
        description="Short-horizon Binance liquidation/dislocation alpha portfolio.",
        ui_group="Cascade Alpha",
    ),
    "mev_scout_sol": PortfolioRunnerSpec(
        portfolio_id="mev_scout_sol",
        label="MEV Scout",
        category="onchain_research",
        control_mode="local_managed",
        currency="USD",
        initial_balance=float(getattr(config, "MEV_SCOUT_SOL_SHADOW_BALANCE_USD", 5000.0)),
        runner_path="portfolio.runners.mev_scout_sol_runner:MevScoutSolPortfolioRunner",
        autostart=False,
        enabled=bool(getattr(config, "MEV_SCOUT_SOL_ENABLED", False)),
        description="Solana/Jito whale-flow and latency research portfolio.",
        ui_group="MEV Scout",
    ),
    "polymarket_quantum_fold": PortfolioRunnerSpec(
        portfolio_id="polymarket_quantum_fold",
        label="Polymarket Quantum-Fold",
        category="polymarket_paper",
        control_mode="local_managed",
        currency="USD",
        initial_balance=float(getattr(config, "POLYMARKET_QF_INITIAL_BALANCE_USD", 7500.0)),
        runner_path="portfolio.runners.polymarket_quantum_fold_runner:PolymarketQuantumFoldPortfolioRunner",
        autostart=False,
        enabled=bool(getattr(config, "POLYMARKET_QF_ENABLED", False)),
        description="Standalone Polymarket sports paper portfolio driven by coherence, folding, and online model-league learning.",
        ui_group="Polymarket",
    ),
    "contrarian_legacy": PortfolioRunnerSpec(
        portfolio_id="contrarian_legacy",
        label="Contrarian Legacy",
        category="crypto_contrarian",
        control_mode="local_managed",
        currency="USD",
        initial_balance=float(getattr(config, "CONTRARIAN_PORTFOLIO_INITIAL_BALANCE_USD", 5000.0)),
        runner_path="portfolio.runners.contrarian_runner:ContrarianLegacyPortfolioRunner",
        autostart=False,
        enabled=bool(getattr(config, "CONTRARIAN_LEGACY_ENABLED", False)),
        description="Directional contrarian portfolio isolated from hedge metrics and bankroll.",
        ui_group="Legacy",
    ),
}


def list_portfolios() -> List[PortfolioRunnerSpec]:
    return list(_REGISTRY.values())


def get_portfolio_spec(portfolio_id: str) -> PortfolioRunnerSpec:
    if portfolio_id not in _REGISTRY:
        raise KeyError(f"Unknown portfolio: {portfolio_id}")
    return _REGISTRY[portfolio_id]


def create_runner(portfolio_id: str):
    spec = get_portfolio_spec(portfolio_id)
    if spec.control_mode == "disabled" or not spec.enabled or not spec.runner_path:
        raise ValueError(f"Portfolio {portfolio_id} is disabled")
    start_blocker = research_factory_start_blocker(portfolio_id)
    if start_blocker is not None:
        raise ValueError(f"Portfolio {portfolio_id} is blocked: {start_blocker}")
    module_name, class_name = spec.runner_path.split(":", 1)
    module = importlib.import_module(module_name)
    runner_cls = getattr(module, class_name)
    return runner_cls(spec)
