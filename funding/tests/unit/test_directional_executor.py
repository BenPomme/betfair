from __future__ import annotations

from decimal import Decimal

from funding.execution.directional_executor import DirectionalExecutor


class _StubFuturesClient:
    async def get_symbol_info(self, symbol: str):
        assert symbol == "BTCUSDT"
        return {
            "symbol": symbol,
            "filters": {
                "MARKET_LOT_SIZE": {
                    "stepSize": "0.001",
                    "minQty": "0.01",
                    "maxQty": "1000",
                }
            },
        }


def test_directional_executor_normalizes_quantity_to_lot_step():
    executor = DirectionalExecutor(futures_client=_StubFuturesClient())
    normalized = __import__("asyncio").run(
        executor._normalize_quantity("BTCUSDT", Decimal("1.23456"))
    )

    assert normalized == Decimal("1.234")


def test_directional_executor_rejects_below_min_qty():
    executor = DirectionalExecutor(futures_client=_StubFuturesClient())
    normalized = __import__("asyncio").run(
        executor._normalize_quantity("BTCUSDT", Decimal("0.005"))
    )

    assert normalized == Decimal("0")
