from decimal import Decimal

from funding.execution.paper_executor import FundingPaperExecutor


def test_entry_fees_match_market_execution_with_bnb_discount():
    executor = FundingPaperExecutor()
    fees = executor._estimate_entry_fees(Decimal("2000"))
    assert fees == Decimal("2.40")


def test_exit_fees_match_market_execution_with_bnb_discount():
    executor = FundingPaperExecutor()
    fees = executor._estimate_exit_fees(Decimal("2000"))
    assert fees == Decimal("2.40")
