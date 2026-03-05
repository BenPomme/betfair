"""
Backtester for funding rate arbitrage strategy.
Replays historical funding rates and simulates the full strategy lifecycle:
  entry → hold → collect funding → exit → calculate P&L.

Compares simple threshold strategy vs ML-gated strategy.
"""
import logging
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from funding.core import fee_calculator
from funding.ml.feature_engineer import (
    build_features_all_symbols,
    get_feature_columns,
    load_funding_rates,
)
from funding.ml.funding_predictor import FundingPredictor

logger = logging.getLogger(__name__)


@dataclass
class BacktestPosition:
    """A simulated hedge position during backtesting."""
    symbol: str
    entry_time: pd.Timestamp
    entry_rate: float
    entry_price: float
    notional: float
    funding_collected: float = 0.0
    periods_held: int = 0
    exit_time: Optional[pd.Timestamp] = None
    exit_price: float = 0.0
    trading_fees: float = 0.0
    pnl: float = 0.0


@dataclass
class BacktestResult:
    """Summary of a backtest run."""
    strategy: str
    total_trades: int = 0
    winning_trades: int = 0
    total_funding_collected: float = 0.0
    total_trading_fees: float = 0.0
    total_slippage: float = 0.0
    net_pnl: float = 0.0
    avg_hold_periods: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    avg_pnl_per_trade: float = 0.0
    trades: List[BacktestPosition] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"=== {self.strategy} ===\n"
            f"Trades: {self.total_trades} (win rate: {self.win_rate:.1%})\n"
            f"Funding collected: ${self.total_funding_collected:.2f}\n"
            f"Trading fees: ${self.total_trading_fees:.2f}\n"
            f"Slippage: ${self.total_slippage:.2f}\n"
            f"Net P&L: ${self.net_pnl:.2f}\n"
            f"Avg hold: {self.avg_hold_periods:.1f} periods\n"
            f"Avg P&L/trade: ${self.avg_pnl_per_trade:.2f}\n"
            f"Max drawdown: ${self.max_drawdown:.2f}\n"
            f"Sharpe ratio: {self.sharpe_ratio:.3f}\n"
        )


class FundingBacktester:
    """Backtest funding rate arbitrage strategies on historical data."""

    def __init__(
        self,
        notional_per_trade: float = 500.0,
        max_positions: int = 5,
        slippage_pct: float = 0.0005,  # 0.05% per leg
        maker_orders: bool = False,
        bnb_discount: bool = False,
    ):
        self._notional = notional_per_trade
        self._max_positions = max_positions
        self._slippage_pct = slippage_pct
        self._maker = maker_orders
        self._bnb = bnb_discount

    def _calc_fees(self, notional: float) -> float:
        """Calculate round-trip trading fees."""
        fees = fee_calculator.trading_fees_round_trip(
            Decimal(str(notional)),
            maker=self._maker,
            bnb_discount=self._bnb,
        )
        return float(fees)

    def _calc_slippage(self, notional: float) -> float:
        """Estimate slippage for both legs entry + exit."""
        return notional * self._slippage_pct * 4  # 4 trades (2 legs × 2 sides)

    def run_simple_strategy(
        self,
        rates_df: pd.DataFrame,
        min_rate: float = 0.0002,
        max_hold_periods: int = 21,  # 7 days
        exit_on_negative: bool = True,
    ) -> BacktestResult:
        """Backtest simple threshold strategy.

        Entry: funding rate > min_rate for current + last 2 periods.
        Exit: rate turns negative, or max hold reached.
        """
        result = BacktestResult(strategy="simple_threshold")
        positions: Dict[str, BacktestPosition] = {}
        daily_pnl: List[float] = []
        cumulative_pnl = 0.0
        peak_pnl = 0.0

        symbols = rates_df["symbol"].unique() if "symbol" in rates_df.columns else [rates_df.index.name or "UNKNOWN"]

        for symbol in symbols:
            if "symbol" in rates_df.columns:
                sym_df = rates_df[rates_df["symbol"] == symbol].sort_index()
            else:
                sym_df = rates_df.sort_index()

            for i in range(3, len(sym_df)):
                row = sym_df.iloc[i]
                time_idx = sym_df.index[i]
                rate = row["funding_rate"]
                price = row.get("mark_price", 0)

                # Check exits first
                key = symbol
                if key in positions:
                    pos = positions[key]
                    pos.periods_held += 1

                    # Collect funding
                    funding = self._notional * rate
                    pos.funding_collected += funding

                    # Exit conditions
                    should_exit = False
                    if exit_on_negative and rate < 0:
                        should_exit = True
                    if pos.periods_held >= max_hold_periods:
                        should_exit = True

                    if should_exit:
                        pos.exit_time = time_idx
                        pos.exit_price = price
                        fees = self._calc_fees(self._notional)
                        slippage = self._calc_slippage(self._notional)
                        pos.trading_fees = fees
                        # Price P&L ≈ 0 (delta-neutral), so net = funding - fees - slippage
                        pos.pnl = pos.funding_collected - fees - slippage

                        result.trades.append(pos)
                        cumulative_pnl += pos.pnl
                        daily_pnl.append(pos.pnl)
                        peak_pnl = max(peak_pnl, cumulative_pnl)
                        result.max_drawdown = max(result.max_drawdown, peak_pnl - cumulative_pnl)

                        del positions[key]

                # Check entries
                if key not in positions and len(positions) < self._max_positions:
                    # Last 3 rates all positive and above threshold
                    last_3 = [sym_df.iloc[i - j]["funding_rate"] for j in range(3)]
                    if all(r >= min_rate for r in last_3):
                        positions[key] = BacktestPosition(
                            symbol=symbol,
                            entry_time=time_idx,
                            entry_rate=rate,
                            entry_price=price,
                            notional=self._notional,
                        )

        # Close remaining positions at last known price
        for pos in positions.values():
            fees = self._calc_fees(self._notional)
            slippage = self._calc_slippage(self._notional)
            pos.trading_fees = fees
            pos.pnl = pos.funding_collected - fees - slippage
            result.trades.append(pos)
            cumulative_pnl += pos.pnl
            daily_pnl.append(pos.pnl)

        self._compute_summary(result, daily_pnl, cumulative_pnl)
        return result

    def run_ml_strategy(
        self,
        features_df: pd.DataFrame,
        predictor: FundingPredictor,
        min_confidence: float = 0.6,
        min_predicted_rate: float = 0.0002,
        max_hold_periods: int = 21,
        dynamic_sizing: bool = False,
    ) -> BacktestResult:
        """Backtest ML-gated strategy.

        Entry: ML predicts positive rate with confidence > threshold.
        Exit: ML predicts negative, or rate turns negative, or max hold.

        If dynamic_sizing=True, scale position size by confidence:
          notional * (0.5 + confidence) — ranges from 0.9x to 1.5x base.
        """
        result = BacktestResult(strategy="ml_gated")
        positions: Dict[str, BacktestPosition] = {}
        daily_pnl: List[float] = []
        cumulative_pnl = 0.0
        peak_pnl = 0.0

        # Get predictions for all data
        predictions = predictor.predict(features_df)

        symbols = features_df["symbol"].unique() if "symbol" in features_df.columns else ["UNKNOWN"]

        for symbol in symbols:
            if "symbol" in features_df.columns:
                mask = features_df["symbol"] == symbol
                sym_feat = features_df[mask].sort_index()
                sym_pred = predictions[mask].sort_index()
            else:
                sym_feat = features_df.sort_index()
                sym_pred = predictions.sort_index()

            for i in range(len(sym_feat)):
                time_idx = sym_feat.index[i]
                row = sym_feat.iloc[i]
                pred = sym_pred.iloc[i]
                rate = row["funding_rate"]
                price = row.get("mark_price", 0)

                key = symbol

                # Dynamic sizing: scale by confidence
                if dynamic_sizing:
                    notional = self._notional * (0.5 + pred["confidence"])
                else:
                    notional = self._notional

                # Check exits
                if key in positions:
                    pos = positions[key]
                    pos.periods_held += 1
                    funding = pos.notional * rate
                    pos.funding_collected += funding

                    should_exit = False
                    if rate < 0:
                        should_exit = True
                    if pos.periods_held >= max_hold_periods:
                        should_exit = True
                    # ML exit: predicts negative with high confidence
                    if pred["predicted_positive"] == 0 and pred["confidence"] > min_confidence:
                        should_exit = True

                    if should_exit:
                        pos.exit_time = time_idx
                        pos.exit_price = price
                        fees = self._calc_fees(pos.notional)
                        slippage = self._calc_slippage(pos.notional)
                        pos.trading_fees = fees
                        pos.pnl = pos.funding_collected - fees - slippage

                        result.trades.append(pos)
                        cumulative_pnl += pos.pnl
                        daily_pnl.append(pos.pnl)
                        peak_pnl = max(peak_pnl, cumulative_pnl)
                        result.max_drawdown = max(result.max_drawdown, peak_pnl - cumulative_pnl)

                        del positions[key]

                # Check entries
                if key not in positions and len(positions) < self._max_positions:
                    if (
                        pred["predicted_positive"] == 1
                        and pred["confidence"] >= min_confidence
                        and pred["predicted_rate"] >= min_predicted_rate
                    ):
                        positions[key] = BacktestPosition(
                            symbol=symbol,
                            entry_time=time_idx,
                            entry_rate=rate,
                            entry_price=price,
                            notional=notional,
                        )

        # Close remaining
        for pos in positions.values():
            fees = self._calc_fees(self._notional)
            slippage = self._calc_slippage(self._notional)
            pos.trading_fees = fees
            pos.pnl = pos.funding_collected - fees - slippage
            result.trades.append(pos)
            cumulative_pnl += pos.pnl
            daily_pnl.append(pos.pnl)

        self._compute_summary(result, daily_pnl, cumulative_pnl)
        return result

    def _compute_summary(
        self, result: BacktestResult, daily_pnl: List[float], cumulative_pnl: float
    ) -> None:
        """Compute summary statistics."""
        result.total_trades = len(result.trades)
        result.winning_trades = sum(1 for t in result.trades if t.pnl > 0)
        result.total_funding_collected = sum(t.funding_collected for t in result.trades)
        result.total_trading_fees = sum(t.trading_fees for t in result.trades)
        result.net_pnl = cumulative_pnl

        if result.total_trades > 0:
            result.win_rate = result.winning_trades / result.total_trades
            result.avg_pnl_per_trade = cumulative_pnl / result.total_trades
            result.avg_hold_periods = np.mean([t.periods_held for t in result.trades])

        if len(daily_pnl) > 1:
            pnl_arr = np.array(daily_pnl)
            mean_ret = pnl_arr.mean()
            std_ret = pnl_arr.std()
            if std_ret > 0:
                result.sharpe_ratio = (mean_ret / std_ret) * np.sqrt(
                    3 * 365
                )  # Annualized (3 periods/day)
