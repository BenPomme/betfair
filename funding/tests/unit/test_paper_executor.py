import asyncio
from datetime import datetime, timezone
from decimal import Decimal

import config
from funding.core.schemas import FundingOpportunity
from funding.execution.paper_executor import FundingPaperExecutor
from funding.execution.position_manager import PositionManager


def _make_opportunity() -> FundingOpportunity:
    return FundingOpportunity(
        symbol="BTCUSDT",
        current_rate=Decimal("0.0005"),
        predicted_rate=Decimal("0.0005"),
        annualized_yield=Decimal("0.50"),
        entry_price_spot=Decimal("50000"),
        entry_price_perp=Decimal("50010"),
        position_size=Decimal("2000"),
        expected_funding_payment=Decimal("1.00"),
        timestamp=datetime.now(timezone.utc),
        next_funding_time=datetime.now(timezone.utc),
        spot_bid=Decimal("49995"),
        spot_ask=Decimal("50005"),
        perp_bid=Decimal("50008"),
        perp_ask=Decimal("50012"),
        basis_bps=Decimal("2"),
    )


def test_entry_fees_match_market_execution_with_bnb_discount():
    executor = FundingPaperExecutor()
    fees = executor._estimate_entry_fees(Decimal("2000"))
    assert fees == Decimal("2.40")


def test_exit_fees_match_market_execution_with_bnb_discount():
    executor = FundingPaperExecutor()
    fees = executor._estimate_exit_fees(Decimal("2000"))
    assert fees == Decimal("2.40")


def test_validation_mode_rejects_missing_testnet_credentials(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FUNDING_VALIDATION_MODE", True)
    monkeypatch.setattr(config, "FUNDING_PAPER_REQUIRE_TESTNET_FILLS", True)
    monkeypatch.setattr(config, "FUNDING_PAPER_ALLOW_SIM_FALLBACK", False)
    monkeypatch.setattr(config, "BINANCE_FUTURES_TESTNET_API_KEY", "")
    monkeypatch.setattr(config, "BINANCE_FUTURES_TESTNET_API_SECRET", "")
    monkeypatch.setattr(config, "BINANCE_SPOT_TESTNET_API_KEY", "")
    monkeypatch.setattr(config, "BINANCE_SPOT_TESTNET_API_SECRET", "")
    monkeypatch.setattr(config, "FUNDING_VALIDATION_ARCHIVE_DIR", str(tmp_path / "archive"))

    pm = PositionManager(state_path=str(tmp_path / "funding_positions.json"))
    pm.begin_validation_run("run_test", manifest={"scope": "hedge_only"})
    executor = FundingPaperExecutor(position_manager=pm)

    result = asyncio.run(
        executor.open_hedge(_make_opportunity(), exchange_filters={"LOT_SIZE": {"stepSize": "0.001"}})
    )
    assert result is None
    rejections = pm.get_recent_rejections(limit=10)
    assert rejections
    assert rejections[-1]["reason"] == "missing_testnet_credentials"


def test_split_fee_fields_are_consistent():
    executor = FundingPaperExecutor()
    spot_fee, perp_fee = executor._split_entry_fees(Decimal("2000"))
    assert spot_fee == Decimal("1.50")
    assert perp_fee == Decimal("0.90")
    assert spot_fee + perp_fee == executor._estimate_entry_fees(Decimal("2000"))


def test_slippage_bps_uses_reference_price():
    executor = FundingPaperExecutor()
    buy_slip = executor._slippage_bps(Decimal("100.50"), Decimal("100"), "BUY")
    sell_slip = executor._slippage_bps(Decimal("99.50"), Decimal("100"), "SELL")
    assert buy_slip == Decimal("50.00")
    assert sell_slip == Decimal("50.00")
