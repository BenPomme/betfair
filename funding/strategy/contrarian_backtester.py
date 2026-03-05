"""
Backtester for the contrarian directional strategy.

Simulates entries on extreme funding rate events, applies stop-loss (SL) and
take-profit (TP) checks against a forward price series, and tracks the
resulting equity curve.

Supports three strategy modes:
  - "xgb"   — XGBoost model (ContrarianXGBoost.predict interface)
  - "tft"   — TFT model (TFTPredictor.predict interface)
  - "naive" — Baseline: always trade contrarian on extreme funding, no model gate

Both ML models must expose:
    predict(features: pd.DataFrame) -> pd.DataFrame
with output columns at minimum:
    direction_prob    -- float in [0, 1]
    confidence        -- float in [0, 1]
    predicted_direction -- int (1=long, 0=short)

Input DataFrame is produced by build_contrarian_features() and must contain
at minimum the following columns:
    funding_rate      -- float, the funding rate at each 8h period
    mark_price        -- float, the mark price used for entry and walk-forward SL/TP
Target columns (price_return_24h_target, direction_24h) are used for labelling
but not required at backtest time; they are used only when present for logging.
"""

import logging
from dataclasses import dataclass, field
from decimal import Decimal, getcontext
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

getcontext().prec = 10

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------

@dataclass
class ContrarianTrade:
    """Record of a single simulated contrarian trade."""

    symbol: str
    entry_idx: int                      # Row index in df at entry
    entry_price: float
    entry_funding_rate: float
    direction: str                      # "long" or "short"
    strategy: str                       # "xgb", "tft", or "naive"
    confidence: float                   # 0.0 for naive baseline

    stop_loss: float = 0.0
    take_profit: float = 0.0
    exit_price: float = 0.0
    exit_idx: int = -1
    hold_periods: int = 0
    outcome: str = "open"               # "win", "loss", or "timeout"
    pnl_pct: float = 0.0                # % return on capital deployed
    pnl_usd: float = 0.0               # USD P&L based on capital_pct of balance


# ---------------------------------------------------------------------------
# Main backtester
# ---------------------------------------------------------------------------

class ContrarianBacktester:
    """Backtester for the contrarian directional strategy.

    Simulates entries at mark_price whenever |funding_rate| > min_funding_rate,
    optionally gated by an ML model's confidence score.  Stop-loss and
    take-profit are checked by walking forward through the price series up to
    max_hold_periods.

    Args:
        initial_balance: Starting virtual balance in USD.
    """

    def __init__(self, initial_balance: float = 500.0) -> None:
        self.initial_balance = float(initial_balance)
        self.results: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def backtest(
        self,
        df: pd.DataFrame,
        strategy: str = "xgb",
        model: Optional[Any] = None,
        stop_loss_pct: float = 0.025,
        take_profit_ratio: float = 2.5,
        capital_pct: float = 0.025,
        max_hold_periods: int = 9,
        min_funding_rate: float = 0.0005,
        min_confidence: float = 0.65,
    ) -> dict:
        """Run a backtest over the provided feature DataFrame.

        The simulation iterates chronologically row by row.  For each row it:
          1. Checks whether |funding_rate| > min_funding_rate (extreme event).
          2. If a model is provided (strategy != "naive"): calls model.predict()
             on the single-row feature slice and gates on confidence.
          3. Determines trade direction:
               extreme positive rate -> SHORT (market over-long, contrarian)
               extreme negative rate -> LONG  (market over-short, contrarian)
          4. Computes SL and TP prices from entry mark_price.
          5. Walks forward up to max_hold_periods to see if SL or TP is hit.
          6. Records the trade outcome and updates the running balance.

        A row is skipped for entry when:
          - It is already occupied by an open position's hold window (to avoid
            overlapping trades in the same symbol/time slot).
          - The model returns confidence below min_confidence.

        Args:
            df:                  Feature DataFrame from build_contrarian_features.
                                 Must contain 'funding_rate' and 'mark_price'.
            strategy:            Label string: "xgb", "tft", or "naive".
            model:               Fitted model with .predict(DataFrame) -> DataFrame.
                                 Required unless strategy=="naive".
            stop_loss_pct:       Fraction of entry price as stop distance (e.g. 0.025 = 2.5%).
            take_profit_ratio:   TP distance = stop_loss_pct * take_profit_ratio.
            capital_pct:         Fraction of current balance deployed per trade.
            max_hold_periods:    Max number of 8h periods to hold before forced exit.
            min_funding_rate:    Minimum |funding_rate| to consider a row extreme.
            min_confidence:      Minimum model confidence to enter (ignored for "naive").

        Returns:
            dict with keys:
                total_trades     (int)
                wins             (int)
                losses           (int)
                timeouts         (int)
                win_rate         (float)
                total_pnl        (float)  — cumulative USD P&L
                final_balance    (float)
                max_drawdown     (float)  — peak-to-trough in USD
                sharpe           (float)  — annualised Sharpe on per-trade P&L
                avg_hold_periods (float)
                profit_factor    (float)  — gross wins / gross losses
                equity_curve     (list)   — balance at each completed trade
                trades_list      (list)   — list of ContrarianTrade objects
        """
        if df is None or df.empty:
            logger.warning("backtest(%s): empty DataFrame supplied", strategy)
            return self._empty_result(strategy)

        required = {"funding_rate", "mark_price"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"backtest: DataFrame missing required columns: {missing}"
            )

        df = df.copy().reset_index(drop=False)  # keep original index as column 'index'

        balance = self.initial_balance
        peak_balance = balance
        max_drawdown = 0.0
        trades: List[ContrarianTrade] = []
        equity_curve: List[float] = [balance]

        # Track which row indices are already inside a hold window to avoid
        # starting a new trade at the same time as an active walk-forward scan
        occupied_until: Dict[str, int] = {}  # symbol -> row index of last occupied row

        symbol_col = "symbol" if "symbol" in df.columns else None

        n = len(df)

        for i in range(n):
            row = df.iloc[i]
            symbol = str(row[symbol_col]) if symbol_col else "UNKNOWN"
            funding_rate = float(row["funding_rate"])
            mark_price = float(row["mark_price"])

            # Skip rows that are NaN in critical columns
            if np.isnan(funding_rate) or np.isnan(mark_price) or mark_price <= 0.0:
                continue

            # Skip if within hold window of a previous trade for this symbol
            if occupied_until.get(symbol, -1) >= i:
                continue

            # --- Extreme funding filter ---
            if abs(funding_rate) <= min_funding_rate:
                continue

            # --- Direction ---
            # Contrarian: positive rate -> market over-long -> we SHORT
            #             negative rate -> market over-short -> we LONG
            direction = "short" if funding_rate > 0.0 else "long"

            # --- Model gate ---
            confidence = 0.0
            if strategy == "naive" or model is None:
                # Naive baseline always enters
                confidence = 0.0
            else:
                feature_row = df.iloc[[i]].copy()
                try:
                    preds = model.predict(feature_row)
                except Exception as exc:
                    logger.warning(
                        "backtest(%s): model.predict raised %s at row %d — skipping",
                        strategy, exc, i,
                    )
                    continue

                if preds is None or (hasattr(preds, "empty") and preds.empty):
                    continue

                confidence = float(preds["confidence"].iloc[0])
                if confidence < min_confidence:
                    continue

                # Reconcile model direction vs funding direction
                # The model's predicted_direction (1=price up, 0=price down)
                # should agree with our contrarian bet; if it disagrees we skip.
                if "predicted_direction" in preds.columns:
                    pred_dir = int(preds["predicted_direction"].iloc[0])
                    model_direction = "long" if pred_dir == 1 else "short"
                    if model_direction != direction:
                        logger.debug(
                            "backtest(%s): row %d — model direction '%s' disagrees "
                            "with contrarian direction '%s', skipping",
                            strategy, i, model_direction, direction,
                        )
                        continue

            # --- Compute SL / TP levels ---
            stop_distance = mark_price * stop_loss_pct
            reward_distance = stop_distance * take_profit_ratio

            if direction == "long":
                sl_price = mark_price - stop_distance
                tp_price = mark_price + reward_distance
            else:  # short
                sl_price = mark_price + stop_distance
                tp_price = mark_price - reward_distance

            # --- Capital at risk ---
            capital_deployed = balance * capital_pct
            # P&L in USD is capital_deployed * pnl_pct
            # (no leverage modelled here — caller adjusts capital_pct to encode leverage)

            # --- Walk forward ---
            outcome = "timeout"
            exit_price = mark_price
            exit_idx = min(i + max_hold_periods, n - 1)
            hold_periods = 0

            for j in range(i + 1, min(i + max_hold_periods + 1, n)):
                forward_row = df.iloc[j]
                forward_sym = str(forward_row[symbol_col]) if symbol_col else "UNKNOWN"
                # Only look at the same symbol's prices
                if symbol_col and forward_sym != symbol:
                    continue

                fwd_price = float(forward_row["mark_price"])
                hold_periods = j - i

                if np.isnan(fwd_price) or fwd_price <= 0.0:
                    continue

                if direction == "long":
                    if fwd_price <= sl_price:
                        outcome = "loss"
                        exit_price = sl_price
                        exit_idx = j
                        break
                    if fwd_price >= tp_price:
                        outcome = "win"
                        exit_price = tp_price
                        exit_idx = j
                        break
                else:  # short
                    if fwd_price >= sl_price:
                        outcome = "loss"
                        exit_price = sl_price
                        exit_idx = j
                        break
                    if fwd_price <= tp_price:
                        outcome = "win"
                        exit_price = tp_price
                        exit_idx = j
                        break

            if hold_periods == 0:
                hold_periods = min(max_hold_periods, n - 1 - i)

            # If we never hit SL/TP, exit at the last forward price
            if outcome == "timeout":
                last_j = min(i + max_hold_periods, n - 1)
                last_forward = df.iloc[last_j]
                if symbol_col:
                    # Scan backwards to find the same symbol's last price
                    for k in range(last_j, i, -1):
                        if str(df.iloc[k][symbol_col]) == symbol:
                            last_forward = df.iloc[k]
                            break
                exit_price = float(last_forward["mark_price"])
                exit_idx = last_j

            # --- P&L calculation ---
            if direction == "long":
                pnl_pct = (exit_price - mark_price) / mark_price
            else:
                pnl_pct = (mark_price - exit_price) / mark_price

            pnl_usd = capital_deployed * pnl_pct
            balance += pnl_usd

            # Update drawdown tracking
            peak_balance = max(peak_balance, balance)
            drawdown = peak_balance - balance
            max_drawdown = max(max_drawdown, drawdown)

            equity_curve.append(balance)

            # --- Mark rows as occupied ---
            occupied_until[symbol] = exit_idx

            # --- Record trade ---
            trade = ContrarianTrade(
                symbol=symbol,
                entry_idx=i,
                entry_price=mark_price,
                entry_funding_rate=funding_rate,
                direction=direction,
                strategy=strategy,
                confidence=confidence,
                stop_loss=sl_price,
                take_profit=tp_price,
                exit_price=exit_price,
                exit_idx=exit_idx,
                hold_periods=hold_periods,
                outcome=outcome,
                pnl_pct=pnl_pct,
                pnl_usd=pnl_usd,
            )
            trades.append(trade)

            logger.debug(
                "backtest(%s): row=%d sym=%s dir=%s entry=%.4f sl=%.4f tp=%.4f "
                "exit=%.4f hold=%d outcome=%s pnl_usd=%.4f bal=%.2f",
                strategy, i, symbol, direction, mark_price, sl_price, tp_price,
                exit_price, hold_periods, outcome, pnl_usd, balance,
            )

        result = self._compute_stats(strategy, trades, equity_curve, balance)
        self.results[strategy] = result
        return result

    def compare_models(
        self,
        df: pd.DataFrame,
        xgb_model: Optional[Any] = None,
        tft_model: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, dict]:
        """Run backtest for xgb, tft, and naive strategies and return comparison.

        Args:
            df:        Feature DataFrame from build_contrarian_features.
            xgb_model: Fitted ContrarianXGBoost instance (or None to skip xgb).
            tft_model: Fitted TFTPredictor instance (or None to skip tft).
            **kwargs:  Additional keyword arguments forwarded to backtest()
                       (stop_loss_pct, take_profit_ratio, capital_pct, etc.)

        Returns:
            Dict with keys "xgb", "tft", "naive", each mapping to a backtest
            result dict as returned by backtest().
        """
        comparison: Dict[str, dict] = {}

        # Always run naive baseline
        logger.info("compare_models: running naive baseline")
        comparison["naive"] = self.backtest(df, strategy="naive", model=None, **kwargs)

        # XGBoost
        if xgb_model is not None:
            logger.info("compare_models: running xgb strategy")
            comparison["xgb"] = self.backtest(df, strategy="xgb", model=xgb_model, **kwargs)
        else:
            logger.info("compare_models: xgb_model not provided, skipping xgb")
            comparison["xgb"] = self._empty_result("xgb")

        # TFT
        if tft_model is not None:
            logger.info("compare_models: running tft strategy")
            comparison["tft"] = self.backtest(df, strategy="tft", model=tft_model, **kwargs)
        else:
            logger.info("compare_models: tft_model not provided, skipping tft")
            comparison["tft"] = self._empty_result("tft")

        self._log_comparison(comparison)
        return comparison

    def plot_equity_curves(self) -> None:
        """Plot equity curves for all strategies stored in self.results.

        Requires matplotlib.  If not installed, logs a warning and returns.
        Each strategy stored in self.results after a compare_models() or
        backtest() call is plotted as a separate line.
        """
        try:
            import matplotlib.pyplot as plt  # noqa: PLC0415
        except ImportError:
            logger.warning(
                "plot_equity_curves: matplotlib is not installed — cannot plot. "
                "Install with: pip install matplotlib"
            )
            return

        if not self.results:
            logger.warning("plot_equity_curves: no results available — run backtest() first")
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        colors = {"xgb": "#2196F3", "tft": "#4CAF50", "naive": "#FF9800"}
        linestyles = {"xgb": "-", "tft": "--", "naive": ":"}

        for strategy, result in self.results.items():
            equity = result.get("equity_curve", [])
            if not equity:
                continue
            color = colors.get(strategy, "#9E9E9E")
            ls = linestyles.get(strategy, "-")
            ax.plot(
                range(len(equity)),
                equity,
                label=f"{strategy} (trades: {result['total_trades']}, "
                      f"win rate: {result['win_rate']:.1%}, "
                      f"P&L: ${result['total_pnl']:.2f})",
                color=color,
                linestyle=ls,
                linewidth=1.8,
            )

        ax.axhline(
            y=self.initial_balance,
            color="#9E9E9E",
            linestyle="-",
            linewidth=0.8,
            label=f"Initial balance ${self.initial_balance:.0f}",
        )
        ax.set_title("Contrarian Strategy — Equity Curves", fontsize=14, fontweight="bold")
        ax.set_xlabel("Completed Trades")
        ax.set_ylabel("Balance (USD)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_stats(
        self,
        strategy: str,
        trades: List[ContrarianTrade],
        equity_curve: List[float],
        final_balance: float,
    ) -> dict:
        """Compute summary statistics from a completed list of trades."""
        total = len(trades)
        if total == 0:
            return self._empty_result(strategy)

        wins = sum(1 for t in trades if t.outcome == "win")
        losses = sum(1 for t in trades if t.outcome == "loss")
        timeouts = sum(1 for t in trades if t.outcome == "timeout")

        win_rate = wins / total
        total_pnl = final_balance - self.initial_balance

        avg_hold = float(np.mean([t.hold_periods for t in trades]))

        pnl_usd_list = np.array([t.pnl_usd for t in trades], dtype=float)
        gross_wins = float(pnl_usd_list[pnl_usd_list > 0].sum()) if (pnl_usd_list > 0).any() else 0.0
        gross_losses = abs(float(pnl_usd_list[pnl_usd_list < 0].sum())) if (pnl_usd_list < 0).any() else 0.0
        profit_factor = gross_wins / gross_losses if gross_losses > 0.0 else float("inf")

        # Sharpe: annualise assuming 3 periods per day (8h each) * 365 days
        # = 1095 periods per year; we compute per-trade Sharpe
        sharpe = 0.0
        if len(pnl_usd_list) > 1:
            mean_pnl = pnl_usd_list.mean()
            std_pnl = pnl_usd_list.std()
            if std_pnl > 0.0:
                # annualisation factor: sqrt(trades per year)
                # Approximate: 3 periods/day * 365 days = 1095; divide by avg hold
                periods_per_year = (3 * 365) / max(avg_hold, 1)
                sharpe = (mean_pnl / std_pnl) * float(np.sqrt(periods_per_year))

        # Max drawdown (already tracked during simulation; re-derive from equity curve)
        eq = np.array(equity_curve, dtype=float)
        running_max = np.maximum.accumulate(eq)
        drawdowns = running_max - eq
        max_drawdown = float(drawdowns.max()) if len(drawdowns) > 0 else 0.0

        logger.info(
            "backtest(%s): %d trades | wins=%d losses=%d timeouts=%d | "
            "win_rate=%.2f | pnl=%.2f | max_dd=%.2f | sharpe=%.3f | pf=%.3f",
            strategy, total, wins, losses, timeouts,
            win_rate, total_pnl, max_drawdown, sharpe, profit_factor,
        )

        return {
            "strategy": strategy,
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "timeouts": timeouts,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "final_balance": final_balance,
            "max_drawdown": max_drawdown,
            "sharpe": sharpe,
            "avg_hold_periods": avg_hold,
            "profit_factor": profit_factor,
            "equity_curve": equity_curve,
            "trades_list": trades,
        }

    @staticmethod
    def _empty_result(strategy: str) -> dict:
        """Return an empty result dict for a strategy that produced no trades."""
        return {
            "strategy": strategy,
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "timeouts": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "final_balance": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "avg_hold_periods": 0.0,
            "profit_factor": 0.0,
            "equity_curve": [],
            "trades_list": [],
        }

    @staticmethod
    def _log_comparison(comparison: Dict[str, dict]) -> None:
        """Log a side-by-side summary of compare_models results."""
        header = f"{'Strategy':<10} {'Trades':>7} {'Win%':>7} {'P&L':>10} {'MaxDD':>10} {'Sharpe':>8} {'PF':>7}"
        logger.info("compare_models results:\n%s", header)
        for name, res in comparison.items():
            row = (
                f"{name:<10} "
                f"{res['total_trades']:>7} "
                f"{res['win_rate']:>7.1%} "
                f"{res['total_pnl']:>10.2f} "
                f"{res['max_drawdown']:>10.2f} "
                f"{res['sharpe']:>8.3f} "
                f"{res['profit_factor']:>7.2f}"
            )
            logger.info("  %s", row)
