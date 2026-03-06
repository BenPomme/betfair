import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import config
from funding.core.schemas import FundingOpportunity
from funding.execution.paper_executor import FundingPaperExecutor
from funding.execution.position_manager import PositionManager


class _FakeFuturesClient:
    async def set_leverage(self, symbol, leverage):
        return {"symbol": symbol, "leverage": leverage}

    async def set_margin_type(self, symbol, margin_type):
        return {"symbol": symbol, "marginType": margin_type}

    async def place_order(self, **kwargs):
        side = kwargs["side"]
        if side == "SELL":
            return {"orderId": 1, "avgPrice": "50010", "executedQty": str(kwargs["quantity"])}
        return {"orderId": 2, "avgPrice": "50012", "executedQty": str(kwargs["quantity"])}

    async def get_order_book(self, symbol, limit=5):
        return {"bids": [["50008", "10"]], "asks": [["50012", "10"]]}


class _FakeSpotClient:
    async def place_order(self, **kwargs):
        side = kwargs["side"]
        if side == "BUY":
            return {"orderId": 3, "executedQty": str(kwargs["quantity"]), "cummulativeQuoteQty": "2000"}
        return {"orderId": 4, "executedQty": str(kwargs["quantity"]), "cummulativeQuoteQty": "1995"}

    async def get_order_book(self, symbol, limit=5):
        return {"bids": [["49995", "10"]], "asks": [["50005", "10"]]}


def test_exchange_backed_validation_trade_persists_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FUNDING_VALIDATION_MODE", True)
    monkeypatch.setattr(config, "FUNDING_PAPER_REQUIRE_TESTNET_FILLS", True)
    monkeypatch.setattr(config, "FUNDING_PAPER_ALLOW_SIM_FALLBACK", False)
    monkeypatch.setattr(config, "FUNDING_VALIDATION_ARCHIVE_DIR", str(tmp_path / "archive"))
    monkeypatch.setattr(config, "BINANCE_FUTURES_TESTNET_API_KEY", "x")
    monkeypatch.setattr(config, "BINANCE_FUTURES_TESTNET_API_SECRET", "x")
    monkeypatch.setattr(config, "BINANCE_SPOT_TESTNET_API_KEY", "x")
    monkeypatch.setattr(config, "BINANCE_SPOT_TESTNET_API_SECRET", "x")

    pm = PositionManager(state_path=str(tmp_path / "funding_positions.json"))
    pm.begin_validation_run("run_int", manifest={"scope": "hedge_only"})
    executor = FundingPaperExecutor(
        futures_client=_FakeFuturesClient(),
        spot_client=_FakeSpotClient(),
        position_manager=pm,
    )

    opp = FundingOpportunity(
        symbol="BTCUSDT",
        current_rate=Decimal("0.0005"),
        predicted_rate=Decimal("0.0005"),
        annualized_yield=Decimal("0.50"),
        entry_price_spot=Decimal("50000"),
        entry_price_perp=Decimal("50010"),
        position_size=Decimal("2000"),
        expected_funding_payment=Decimal("1.00"),
        timestamp=datetime.now(timezone.utc),
        next_funding_time=datetime.now(timezone.utc) + timedelta(minutes=8),
        spot_bid=Decimal("49995"),
        spot_ask=Decimal("50005"),
        perp_bid=Decimal("50008"),
        perp_ask=Decimal("50012"),
        basis_bps=Decimal("2"),
    )

    opened = asyncio.run(
        executor.open_hedge(opp, exchange_filters={"LOT_SIZE": {"stepSize": "0.001"}})
    )
    assert opened is not None
    assert opened.fill_source == "exchange_testnet"
    assert opened.validation_run_id == "run_int"
    assert opened.entry_order_id_spot == "3"
    assert opened.entry_order_id_perp == "1"
