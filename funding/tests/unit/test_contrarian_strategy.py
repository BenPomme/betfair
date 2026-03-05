"""
Unit tests for ContrarianStrategy.

Tests cover:
  1. Signal generation: extreme funding produces signals
  2. No signals for moderate (below-threshold) funding
  3. Direction mapping: positive extreme -> SHORT, negative extreme -> LONG
  4. Confidence filter: signals below CONTRARIAN_MIN_CONFIDENCE are dropped
  5. Position sizing: correct capital per trade and notional
  6. SL/TP calculation: correct stop-loss and take-profit prices for LONG and SHORT

The ML model dependency is mocked so tests run without trained weights.
"""

import os
import unittest
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from unittest.mock import MagicMock, patch

import pandas as pd

# ---------------------------------------------------------------------------
# Patch config constants before importing the module under test so that tests
# are not sensitive to whatever is in .env at test runtime.
# ---------------------------------------------------------------------------
_CONFIG_DEFAULTS = {
    "CONTRARIAN_MIN_FUNDING_RATE": Decimal("0.0005"),
    "CONTRARIAN_MIN_CONFIDENCE": 0.65,
    "CONTRARIAN_CAPITAL_PER_TRADE_PCT": Decimal("0.025"),
    "CONTRARIAN_LEVERAGE": 2,
    "CONTRARIAN_STOP_LOSS_PCT": Decimal("0.025"),
    "CONTRARIAN_TAKE_PROFIT_RATIO": Decimal("2.5"),
    "FUNDING_MARGIN_TYPE": "ISOLATED",
}


def _make_snapshot(symbol: str, funding_rate: str, mark_price: str = "50000.0"):
    """Create a minimal FundingSnapshot for a given symbol and funding rate."""
    from funding.core.schemas import FundingSnapshot
    return FundingSnapshot(
        symbol=symbol,
        funding_rate=Decimal(funding_rate),
        next_funding_time=datetime.now(timezone.utc),
        mark_price=Decimal(mark_price),
        index_price=Decimal(mark_price),
        open_interest=Decimal("1000"),
        timestamp=datetime.now(timezone.utc),
    )


def _make_history(n: int = 10, rate: str = "0.001") -> list:
    """Produce a list of n funding-rate history dicts (newest first)."""
    return [{"funding_rate": rate}] * n


def _make_model_prediction(
    direction_prob: float = 0.80,
    predicted_return_24h: float = 0.02,
    confidence: float = 0.70,
    predicted_return_72h: float = 0.04,
) -> pd.DataFrame:
    """Return a single-row prediction DataFrame as the real model would."""
    return pd.DataFrame([{
        "direction_prob": direction_prob,
        "predicted_return_24h": predicted_return_24h,
        "confidence": confidence,
        "predicted_return_72h": predicted_return_72h,
    }])


class TestContrarianStrategySignals(unittest.TestCase):
    """Tests for ContrarianStrategy.evaluate_signals()."""

    def _make_strategy(self, model_prediction: pd.DataFrame):
        """Build a ContrarianStrategy with a mocked ML model."""
        with patch.dict("sys.modules", {}):
            import config
        mock_model = MagicMock()
        mock_model.predict.return_value = model_prediction
        from funding.strategy.contrarian_strategy import ContrarianStrategy
        return ContrarianStrategy(model=mock_model), mock_model

    # ------------------------------------------------------------------
    # 1. Signal generated for extreme funding
    # ------------------------------------------------------------------

    @patch.dict(os.environ, {
        "CONTRARIAN_MIN_FUNDING_RATE": "0.0005",
        "CONTRARIAN_MIN_CONFIDENCE": "0.65",
    })
    def test_extreme_positive_funding_produces_signal(self):
        """A rate well above the threshold must produce exactly one signal."""
        strategy, mock_model = self._make_strategy(
            _make_model_prediction(direction_prob=0.20, confidence=0.70)
        )
        mock_model.__class__.__name__ = "MockModel"

        snapshots = {"BTCUSDT": _make_snapshot("BTCUSDT", "0.0015")}
        histories = {"BTCUSDT": _make_history(rate="0.0015")}

        with patch("config.CONTRARIAN_MIN_FUNDING_RATE", Decimal("0.0005")), \
             patch("config.CONTRARIAN_MIN_CONFIDENCE", 0.65):
            signals = strategy.evaluate_signals(snapshots, histories)

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].symbol, "BTCUSDT")

    @patch.dict(os.environ, {
        "CONTRARIAN_MIN_FUNDING_RATE": "0.0005",
        "CONTRARIAN_MIN_CONFIDENCE": "0.65",
    })
    def test_extreme_negative_funding_produces_signal(self):
        """A strongly negative rate must also produce a signal."""
        strategy, mock_model = self._make_strategy(
            _make_model_prediction(direction_prob=0.80, confidence=0.70)
        )

        snapshots = {"ETHUSDT": _make_snapshot("ETHUSDT", "-0.0012")}
        histories = {"ETHUSDT": _make_history(rate="-0.0012")}

        with patch("config.CONTRARIAN_MIN_FUNDING_RATE", Decimal("0.0005")), \
             patch("config.CONTRARIAN_MIN_CONFIDENCE", 0.65):
            signals = strategy.evaluate_signals(snapshots, histories)

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].symbol, "ETHUSDT")

    # ------------------------------------------------------------------
    # 2. No signals for moderate (below-threshold) funding
    # ------------------------------------------------------------------

    @patch.dict(os.environ, {
        "CONTRARIAN_MIN_FUNDING_RATE": "0.0005",
        "CONTRARIAN_MIN_CONFIDENCE": "0.65",
    })
    def test_moderate_funding_produces_no_signal(self):
        """Funding rate below the threshold must be skipped entirely."""
        strategy, mock_model = self._make_strategy(
            _make_model_prediction(confidence=0.80)
        )

        snapshots = {"SOLUSDT": _make_snapshot("SOLUSDT", "0.0001")}  # below 0.0005
        histories = {"SOLUSDT": _make_history(rate="0.0001")}

        with patch("config.CONTRARIAN_MIN_FUNDING_RATE", Decimal("0.0005")), \
             patch("config.CONTRARIAN_MIN_CONFIDENCE", 0.65):
            signals = strategy.evaluate_signals(snapshots, histories)

        self.assertEqual(len(signals), 0)
        mock_model.predict.assert_not_called()

    @patch.dict(os.environ, {
        "CONTRARIAN_MIN_FUNDING_RATE": "0.0005",
        "CONTRARIAN_MIN_CONFIDENCE": "0.65",
    })
    def test_zero_funding_produces_no_signal(self):
        """A zero funding rate is not extreme and must be skipped."""
        strategy, mock_model = self._make_strategy(
            _make_model_prediction(confidence=0.90)
        )

        snapshots = {"BNBUSDT": _make_snapshot("BNBUSDT", "0.0000")}
        histories = {"BNBUSDT": _make_history(rate="0.0000")}

        with patch("config.CONTRARIAN_MIN_FUNDING_RATE", Decimal("0.0005")), \
             patch("config.CONTRARIAN_MIN_CONFIDENCE", 0.65):
            signals = strategy.evaluate_signals(snapshots, histories)

        self.assertEqual(len(signals), 0)

    # ------------------------------------------------------------------
    # 3. Direction mapping
    # ------------------------------------------------------------------

    @patch.dict(os.environ, {
        "CONTRARIAN_MIN_FUNDING_RATE": "0.0005",
        "CONTRARIAN_MIN_CONFIDENCE": "0.65",
    })
    def test_positive_extreme_rate_produces_short_signal(self):
        """Extreme positive funding -> contrarian SHORT (bet on reversal down)."""
        from funding.core.schemas import DirectionalSide
        strategy, _ = self._make_strategy(
            _make_model_prediction(direction_prob=0.20, confidence=0.70)
        )

        snapshots = {"BTCUSDT": _make_snapshot("BTCUSDT", "0.0010")}
        histories = {"BTCUSDT": _make_history(rate="0.0010")}

        with patch("config.CONTRARIAN_MIN_FUNDING_RATE", Decimal("0.0005")), \
             patch("config.CONTRARIAN_MIN_CONFIDENCE", 0.65):
            signals = strategy.evaluate_signals(snapshots, histories)

        self.assertEqual(len(signals), 1)
        self.assertIs(signals[0].direction, DirectionalSide.SHORT)

    @patch.dict(os.environ, {
        "CONTRARIAN_MIN_FUNDING_RATE": "0.0005",
        "CONTRARIAN_MIN_CONFIDENCE": "0.65",
    })
    def test_negative_extreme_rate_produces_long_signal(self):
        """Extreme negative funding -> contrarian LONG (bet on reversal up)."""
        from funding.core.schemas import DirectionalSide
        strategy, _ = self._make_strategy(
            _make_model_prediction(direction_prob=0.80, confidence=0.70)
        )

        snapshots = {"ETHUSDT": _make_snapshot("ETHUSDT", "-0.0008")}
        histories = {"ETHUSDT": _make_history(rate="-0.0008")}

        with patch("config.CONTRARIAN_MIN_FUNDING_RATE", Decimal("0.0005")), \
             patch("config.CONTRARIAN_MIN_CONFIDENCE", 0.65):
            signals = strategy.evaluate_signals(snapshots, histories)

        self.assertEqual(len(signals), 1)
        self.assertIs(signals[0].direction, DirectionalSide.LONG)

    # ------------------------------------------------------------------
    # 4. Confidence filter
    # ------------------------------------------------------------------

    @patch.dict(os.environ, {
        "CONTRARIAN_MIN_FUNDING_RATE": "0.0005",
        "CONTRARIAN_MIN_CONFIDENCE": "0.65",
    })
    def test_low_confidence_signal_is_filtered(self):
        """Model confidence below CONTRARIAN_MIN_CONFIDENCE must drop the signal."""
        strategy, _ = self._make_strategy(
            _make_model_prediction(confidence=0.55)  # below 0.65 threshold
        )

        snapshots = {"BTCUSDT": _make_snapshot("BTCUSDT", "0.0020")}
        histories = {"BTCUSDT": _make_history(rate="0.0020")}

        with patch("config.CONTRARIAN_MIN_FUNDING_RATE", Decimal("0.0005")), \
             patch("config.CONTRARIAN_MIN_CONFIDENCE", 0.65):
            signals = strategy.evaluate_signals(snapshots, histories)

        self.assertEqual(len(signals), 0)

    @patch.dict(os.environ, {
        "CONTRARIAN_MIN_FUNDING_RATE": "0.0005",
        "CONTRARIAN_MIN_CONFIDENCE": "0.65",
    })
    def test_confidence_exactly_at_threshold_passes(self):
        """Confidence exactly equal to the threshold must pass the filter."""
        strategy, _ = self._make_strategy(
            _make_model_prediction(confidence=0.65)
        )

        snapshots = {"BTCUSDT": _make_snapshot("BTCUSDT", "0.0010")}
        histories = {"BTCUSDT": _make_history(rate="0.0010")}

        with patch("config.CONTRARIAN_MIN_FUNDING_RATE", Decimal("0.0005")), \
             patch("config.CONTRARIAN_MIN_CONFIDENCE", 0.65):
            signals = strategy.evaluate_signals(snapshots, histories)

        self.assertEqual(len(signals), 1)

    @patch.dict(os.environ, {
        "CONTRARIAN_MIN_FUNDING_RATE": "0.0005",
        "CONTRARIAN_MIN_CONFIDENCE": "0.65",
    })
    def test_regime_adapter_reduces_confidence_below_threshold(self):
        """Regime multiplier < 1.0 can push adjusted confidence below threshold."""
        strategy, mock_model = self._make_strategy(
            _make_model_prediction(confidence=0.70)
        )

        # Attach a regime adapter that applies the "crisis" multiplier (0.70):
        # adjusted = 0.70 * 0.70 = 0.49 < 0.65 -> should be filtered
        mock_adapter = MagicMock()
        mock_adapter.get_multiplier.return_value = 0.70  # crisis multiplier

        from funding.strategy.contrarian_strategy import ContrarianStrategy
        strategy_with_regime = ContrarianStrategy(
            model=mock_model, regime_adapter=mock_adapter
        )

        snapshots = {"BTCUSDT": _make_snapshot("BTCUSDT", "0.0015")}
        histories = {"BTCUSDT": _make_history(rate="0.0015")}

        with patch("config.CONTRARIAN_MIN_FUNDING_RATE", Decimal("0.0005")), \
             patch("config.CONTRARIAN_MIN_CONFIDENCE", 0.65):
            signals = strategy_with_regime.evaluate_signals(snapshots, histories)

        self.assertEqual(len(signals), 0)

    @patch.dict(os.environ, {
        "CONTRARIAN_MIN_FUNDING_RATE": "0.0005",
        "CONTRARIAN_MIN_CONFIDENCE": "0.65",
    })
    def test_regime_adapter_multiplies_passing_confidence(self):
        """Regime multiplier > 1.0 is applied to confidence that already passes the
        pre-filter, and the adjusted (boosted) value is stored on the signal.

        Note: the pre-filter (Step 4) runs before regime adjustment (Step 6), so the
        regime adapter can only scale a signal that already passes. A raw confidence
        of 0.70 with a 1.10 "high" multiplier produces adjusted confidence 0.77.
        """
        strategy, mock_model = self._make_strategy(
            _make_model_prediction(confidence=0.70)  # passes pre-filter at 0.65
        )

        # "high" multiplier 1.10: adjusted = 0.70 * 1.10 = 0.77
        mock_adapter = MagicMock()
        mock_adapter.get_multiplier.return_value = 1.10

        from funding.strategy.contrarian_strategy import ContrarianStrategy
        strategy_with_regime = ContrarianStrategy(
            model=mock_model, regime_adapter=mock_adapter
        )

        snapshots = {"BTCUSDT": _make_snapshot("BTCUSDT", "0.0015")}
        histories = {"BTCUSDT": _make_history(rate="0.0015")}

        with patch("config.CONTRARIAN_MIN_FUNDING_RATE", Decimal("0.0005")), \
             patch("config.CONTRARIAN_MIN_CONFIDENCE", 0.65):
            signals = strategy_with_regime.evaluate_signals(snapshots, histories)

        self.assertEqual(len(signals), 1)
        self.assertAlmostEqual(signals[0].confidence, 0.70 * 1.10, places=6)

    # ------------------------------------------------------------------
    # 5. No model -> no signals
    # ------------------------------------------------------------------

    def test_no_model_returns_empty_list(self):
        """Strategy with model=None must return an empty list and not crash."""
        from funding.strategy.contrarian_strategy import ContrarianStrategy
        strategy = ContrarianStrategy(model=None)

        snapshots = {"BTCUSDT": _make_snapshot("BTCUSDT", "0.0020")}
        histories = {"BTCUSDT": _make_history(rate="0.0020")}

        with patch("config.CONTRARIAN_MIN_FUNDING_RATE", Decimal("0.0005")), \
             patch("config.CONTRARIAN_MIN_CONFIDENCE", 0.65):
            signals = strategy.evaluate_signals(snapshots, histories)

        self.assertEqual(signals, [])

    # ------------------------------------------------------------------
    # 6. Signals sorted by confidence descending
    # ------------------------------------------------------------------

    @patch.dict(os.environ, {
        "CONTRARIAN_MIN_FUNDING_RATE": "0.0005",
        "CONTRARIAN_MIN_CONFIDENCE": "0.65",
    })
    def test_signals_sorted_by_confidence_descending(self):
        """Multiple signals must be returned strongest-first."""
        from funding.strategy.contrarian_strategy import ContrarianStrategy
        mock_model = MagicMock()

        # Return different confidence values per symbol call
        confidences = {"BTCUSDT": 0.70, "ETHUSDT": 0.90, "SOLUSDT": 0.75}

        def side_effect(df):
            # Identify symbol from the call count to vary output
            idx = mock_model.predict.call_count - 1
            sym = list(confidences.keys())[idx]
            conf = confidences[sym]
            return pd.DataFrame([{
                "direction_prob": 0.20,
                "predicted_return_24h": 0.01,
                "confidence": conf,
                "predicted_return_72h": 0.02,
            }])

        mock_model.predict.side_effect = side_effect

        strategy = ContrarianStrategy(model=mock_model)

        snapshots = {
            "BTCUSDT": _make_snapshot("BTCUSDT", "0.0010"),
            "ETHUSDT": _make_snapshot("ETHUSDT", "0.0010"),
            "SOLUSDT": _make_snapshot("SOLUSDT", "0.0010"),
        }
        histories = {
            sym: _make_history(rate="0.0010") for sym in snapshots
        }

        with patch("config.CONTRARIAN_MIN_FUNDING_RATE", Decimal("0.0005")), \
             patch("config.CONTRARIAN_MIN_CONFIDENCE", 0.65):
            signals = strategy.evaluate_signals(snapshots, histories)

        self.assertEqual(len(signals), 3)
        confs = [s.confidence for s in signals]
        self.assertEqual(confs, sorted(confs, reverse=True))

    # ------------------------------------------------------------------
    # 7. Missing rate history skips symbol
    # ------------------------------------------------------------------

    @patch.dict(os.environ, {
        "CONTRARIAN_MIN_FUNDING_RATE": "0.0005",
        "CONTRARIAN_MIN_CONFIDENCE": "0.65",
    })
    def test_missing_rate_history_skips_symbol(self):
        """Symbol without rate history must be silently skipped."""
        strategy, mock_model = self._make_strategy(
            _make_model_prediction(confidence=0.80)
        )

        snapshots = {"BTCUSDT": _make_snapshot("BTCUSDT", "0.0015")}
        histories = {}  # empty — no history for BTCUSDT

        with patch("config.CONTRARIAN_MIN_FUNDING_RATE", Decimal("0.0005")), \
             patch("config.CONTRARIAN_MIN_CONFIDENCE", 0.65):
            signals = strategy.evaluate_signals(snapshots, histories)

        self.assertEqual(len(signals), 0)
        mock_model.predict.assert_not_called()

    # ------------------------------------------------------------------
    # 8. Prediction metadata attached to signal
    # ------------------------------------------------------------------

    @patch.dict(os.environ, {
        "CONTRARIAN_MIN_FUNDING_RATE": "0.0005",
        "CONTRARIAN_MIN_CONFIDENCE": "0.65",
    })
    def test_signal_carries_prediction_metadata(self):
        """ContrarianSignal must carry predicted_return_24h and funding_rate."""
        strategy, _ = self._make_strategy(
            _make_model_prediction(
                direction_prob=0.20,
                predicted_return_24h=0.035,
                confidence=0.72,
                predicted_return_72h=0.08,
            )
        )

        snapshots = {"BTCUSDT": _make_snapshot("BTCUSDT", "0.0010")}
        histories = {"BTCUSDT": _make_history(rate="0.0010")}

        with patch("config.CONTRARIAN_MIN_FUNDING_RATE", Decimal("0.0005")), \
             patch("config.CONTRARIAN_MIN_CONFIDENCE", 0.65):
            signals = strategy.evaluate_signals(snapshots, histories)

        self.assertEqual(len(signals), 1)
        sig = signals[0]
        self.assertAlmostEqual(sig.predicted_return_24h, 0.035, places=6)
        self.assertAlmostEqual(sig.predicted_return_72h, 0.08, places=6)
        self.assertEqual(sig.funding_rate, Decimal("0.0010"))


class TestCalculatePositionParams(unittest.TestCase):
    """Tests for ContrarianStrategy.calculate_position_params().

    Mark prices are kept small (e.g. "10.00") throughout this class so that
    intermediate Decimal values remain within the module-level decimal precision
    of 10 significant figures set by contrarian_strategy.py at import time.
    Specifically: entry * stop_loss_pct = 10.00 * 0.025 = 0.25, which
    quantizes to "0.25000000" (8 significant figures) without overflow.
    """

    def _make_strategy(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = _make_model_prediction()
        from funding.strategy.contrarian_strategy import ContrarianStrategy
        return ContrarianStrategy(model=mock_model)

    def _make_signal(
        self,
        symbol: str = "BTCUSDT",
        direction_str: str = "LONG",
        funding_rate: str = "0.0010",
        mark_price: str = "10.00",
    ):
        """Build a ContrarianSignal with an optional mark_price duck-type attribute."""
        from funding.core.schemas import ContrarianSignal, DirectionalSide
        direction = DirectionalSide(direction_str)
        sig = ContrarianSignal(
            symbol=symbol,
            direction=direction,
            confidence=0.75,
            predicted_return_24h=0.02,
            predicted_return_72h=0.05,
            model_name="MockModel",
            funding_rate=Decimal(funding_rate),
        )
        # Duck-type extension used by calculate_position_params
        object.__setattr__(sig, "mark_price", Decimal(mark_price))
        return sig

    # ------------------------------------------------------------------
    # 5. Position sizing
    # ------------------------------------------------------------------

    @patch("config.CONTRARIAN_CAPITAL_PER_TRADE_PCT", Decimal("0.025"))
    @patch("config.CONTRARIAN_LEVERAGE", 2)
    @patch("config.CONTRARIAN_STOP_LOSS_PCT", Decimal("0.025"))
    @patch("config.CONTRARIAN_TAKE_PROFIT_RATIO", Decimal("2.5"))
    @patch("config.FUNDING_MARGIN_TYPE", "ISOLATED")
    def test_capital_per_trade_correct(self):
        """capital_used = balance * CONTRARIAN_CAPITAL_PER_TRADE_PCT (2.5%)."""
        strategy = self._make_strategy()
        balance = Decimal("10000.00")
        signal = self._make_signal(direction_str="LONG", mark_price="10.00")

        params = strategy.calculate_position_params(signal, balance)

        expected_capital = (balance * Decimal("0.025")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        self.assertEqual(params["capital_used"], expected_capital)

    @patch("config.CONTRARIAN_CAPITAL_PER_TRADE_PCT", Decimal("0.025"))
    @patch("config.CONTRARIAN_LEVERAGE", 2)
    @patch("config.CONTRARIAN_STOP_LOSS_PCT", Decimal("0.025"))
    @patch("config.CONTRARIAN_TAKE_PROFIT_RATIO", Decimal("2.5"))
    @patch("config.FUNDING_MARGIN_TYPE", "ISOLATED")
    def test_notional_equals_capital_times_leverage(self):
        """notional = capital_used * leverage."""
        strategy = self._make_strategy()
        balance = Decimal("10000.00")
        signal = self._make_signal(direction_str="SHORT", mark_price="10.00")

        params = strategy.calculate_position_params(signal, balance)

        expected_capital = (balance * Decimal("0.025")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        expected_notional = (expected_capital * Decimal("2")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        self.assertEqual(params["notional"], expected_notional)

    @patch("config.CONTRARIAN_CAPITAL_PER_TRADE_PCT", Decimal("0.025"))
    @patch("config.CONTRARIAN_LEVERAGE", 2)
    @patch("config.CONTRARIAN_STOP_LOSS_PCT", Decimal("0.025"))
    @patch("config.CONTRARIAN_TAKE_PROFIT_RATIO", Decimal("2.5"))
    @patch("config.FUNDING_MARGIN_TYPE", "ISOLATED")
    def test_quantity_equals_notional_over_mark_price(self):
        """quantity = notional / mark_price, rounded to 8 dp."""
        strategy = self._make_strategy()
        balance = Decimal("10000.00")
        mark_price = Decimal("10.00")
        signal = self._make_signal(direction_str="LONG", mark_price=str(mark_price))

        params = strategy.calculate_position_params(signal, balance)

        expected_capital = (balance * Decimal("0.025")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        expected_notional = (expected_capital * Decimal("2")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        expected_qty = (expected_notional / mark_price).quantize(
            Decimal("0.00000001"), rounding=ROUND_HALF_UP
        )
        self.assertEqual(params["quantity"], expected_qty)

    @patch("config.CONTRARIAN_CAPITAL_PER_TRADE_PCT", Decimal("0.025"))
    @patch("config.CONTRARIAN_LEVERAGE", 2)
    @patch("config.CONTRARIAN_STOP_LOSS_PCT", Decimal("0.025"))
    @patch("config.CONTRARIAN_TAKE_PROFIT_RATIO", Decimal("2.5"))
    @patch("config.FUNDING_MARGIN_TYPE", "ISOLATED")
    def test_leverage_and_margin_type_returned(self):
        """leverage and margin_type must match config values."""
        strategy = self._make_strategy()
        signal = self._make_signal(direction_str="LONG", mark_price="10.00")

        params = strategy.calculate_position_params(signal, Decimal("5000.00"))

        self.assertEqual(params["leverage"], 2)
        self.assertEqual(params["margin_type"], "ISOLATED")

    # ------------------------------------------------------------------
    # 6. SL/TP calculation
    # ------------------------------------------------------------------

    @patch("config.CONTRARIAN_CAPITAL_PER_TRADE_PCT", Decimal("0.025"))
    @patch("config.CONTRARIAN_LEVERAGE", 2)
    @patch("config.CONTRARIAN_STOP_LOSS_PCT", Decimal("0.025"))
    @patch("config.CONTRARIAN_TAKE_PROFIT_RATIO", Decimal("2.5"))
    @patch("config.FUNDING_MARGIN_TYPE", "ISOLATED")
    def test_long_stop_loss_below_entry(self):
        """LONG stop-loss must be strictly below the entry price."""
        strategy = self._make_strategy()
        signal = self._make_signal(direction_str="LONG", mark_price="10.00")

        params = strategy.calculate_position_params(signal, Decimal("10000.00"))

        self.assertLess(params["stop_loss"], params["entry_price"])

    @patch("config.CONTRARIAN_CAPITAL_PER_TRADE_PCT", Decimal("0.025"))
    @patch("config.CONTRARIAN_LEVERAGE", 2)
    @patch("config.CONTRARIAN_STOP_LOSS_PCT", Decimal("0.025"))
    @patch("config.CONTRARIAN_TAKE_PROFIT_RATIO", Decimal("2.5"))
    @patch("config.FUNDING_MARGIN_TYPE", "ISOLATED")
    def test_long_take_profit_above_entry(self):
        """LONG take-profit must be strictly above the entry price."""
        strategy = self._make_strategy()
        signal = self._make_signal(direction_str="LONG", mark_price="10.00")

        params = strategy.calculate_position_params(signal, Decimal("10000.00"))

        self.assertGreater(params["take_profit"], params["entry_price"])

    @patch("config.CONTRARIAN_CAPITAL_PER_TRADE_PCT", Decimal("0.025"))
    @patch("config.CONTRARIAN_LEVERAGE", 2)
    @patch("config.CONTRARIAN_STOP_LOSS_PCT", Decimal("0.025"))
    @patch("config.CONTRARIAN_TAKE_PROFIT_RATIO", Decimal("2.5"))
    @patch("config.FUNDING_MARGIN_TYPE", "ISOLATED")
    def test_short_stop_loss_above_entry(self):
        """SHORT stop-loss must be strictly above the entry price."""
        strategy = self._make_strategy()
        signal = self._make_signal(direction_str="SHORT", mark_price="10.00")

        params = strategy.calculate_position_params(signal, Decimal("10000.00"))

        self.assertGreater(params["stop_loss"], params["entry_price"])

    @patch("config.CONTRARIAN_CAPITAL_PER_TRADE_PCT", Decimal("0.025"))
    @patch("config.CONTRARIAN_LEVERAGE", 2)
    @patch("config.CONTRARIAN_STOP_LOSS_PCT", Decimal("0.025"))
    @patch("config.CONTRARIAN_TAKE_PROFIT_RATIO", Decimal("2.5"))
    @patch("config.FUNDING_MARGIN_TYPE", "ISOLATED")
    def test_short_take_profit_below_entry(self):
        """SHORT take-profit must be strictly below the entry price."""
        strategy = self._make_strategy()
        signal = self._make_signal(direction_str="SHORT", mark_price="10.00")

        params = strategy.calculate_position_params(signal, Decimal("10000.00"))

        self.assertLess(params["take_profit"], params["entry_price"])

    @patch("config.CONTRARIAN_CAPITAL_PER_TRADE_PCT", Decimal("0.025"))
    @patch("config.CONTRARIAN_LEVERAGE", 2)
    @patch("config.CONTRARIAN_STOP_LOSS_PCT", Decimal("0.025"))
    @patch("config.CONTRARIAN_TAKE_PROFIT_RATIO", Decimal("2.5"))
    @patch("config.FUNDING_MARGIN_TYPE", "ISOLATED")
    def test_sl_tp_distances_respect_ratio(self):
        """TP distance must be CONTRARIAN_TAKE_PROFIT_RATIO (2.5) * SL distance."""
        strategy = self._make_strategy()
        mark_price = Decimal("10.00")
        signal = self._make_signal(direction_str="LONG", mark_price=str(mark_price))

        params = strategy.calculate_position_params(signal, Decimal("10000.00"))

        entry = params["entry_price"]
        sl_dist = entry - params["stop_loss"]
        tp_dist = params["take_profit"] - entry

        ratio = tp_dist / sl_dist
        self.assertAlmostEqual(float(ratio), 2.5, places=4)

    @patch("config.CONTRARIAN_CAPITAL_PER_TRADE_PCT", Decimal("0.025"))
    @patch("config.CONTRARIAN_LEVERAGE", 2)
    @patch("config.CONTRARIAN_STOP_LOSS_PCT", Decimal("0.025"))
    @patch("config.CONTRARIAN_TAKE_PROFIT_RATIO", Decimal("2.5"))
    @patch("config.FUNDING_MARGIN_TYPE", "ISOLATED")
    def test_sl_distance_equals_entry_times_stop_loss_pct(self):
        """SL distance from entry = entry * CONTRARIAN_STOP_LOSS_PCT (2.5%)."""
        strategy = self._make_strategy()
        mark_price = Decimal("10.00")
        signal = self._make_signal(direction_str="LONG", mark_price=str(mark_price))

        params = strategy.calculate_position_params(signal, Decimal("10000.00"))

        entry = params["entry_price"]
        sl_distance = entry - params["stop_loss"]
        expected_distance = (entry * Decimal("0.025")).quantize(
            Decimal("0.00000001"), rounding=ROUND_HALF_UP
        )
        self.assertEqual(sl_distance, expected_distance)

    @patch("config.CONTRARIAN_CAPITAL_PER_TRADE_PCT", Decimal("0.025"))
    @patch("config.CONTRARIAN_LEVERAGE", 2)
    @patch("config.CONTRARIAN_STOP_LOSS_PCT", Decimal("0.025"))
    @patch("config.CONTRARIAN_TAKE_PROFIT_RATIO", Decimal("2.5"))
    @patch("config.FUNDING_MARGIN_TYPE", "ISOLATED")
    def test_no_mark_price_returns_sentinel_zeros(self):
        """When mark_price is absent on the signal, sentinels (Decimal 0) are returned."""
        from funding.core.schemas import ContrarianSignal, DirectionalSide
        strategy = self._make_strategy()

        # Build signal WITHOUT duck-typed mark_price
        signal = ContrarianSignal(
            symbol="BTCUSDT",
            direction=DirectionalSide.LONG,
            confidence=0.75,
            predicted_return_24h=0.02,
            predicted_return_72h=0.05,
            model_name="MockModel",
            funding_rate=Decimal("0.0010"),
        )

        params = strategy.calculate_position_params(signal, Decimal("10000.00"))

        self.assertEqual(params["entry_price"], Decimal("0"))
        self.assertEqual(params["stop_loss"], Decimal("0"))
        self.assertEqual(params["take_profit"], Decimal("0"))
        self.assertEqual(params["quantity"], Decimal("0"))

    @patch("config.CONTRARIAN_CAPITAL_PER_TRADE_PCT", Decimal("0.025"))
    @patch("config.CONTRARIAN_LEVERAGE", 2)
    @patch("config.CONTRARIAN_STOP_LOSS_PCT", Decimal("0.025"))
    @patch("config.CONTRARIAN_TAKE_PROFIT_RATIO", Decimal("2.5"))
    @patch("config.FUNDING_MARGIN_TYPE", "ISOLATED")
    def test_long_side_is_buy(self):
        """LONG signal must map to exchange side 'BUY'."""
        strategy = self._make_strategy()
        signal = self._make_signal(direction_str="LONG", mark_price="10.00")

        params = strategy.calculate_position_params(signal, Decimal("10000.00"))

        self.assertEqual(params["side"], "BUY")

    @patch("config.CONTRARIAN_CAPITAL_PER_TRADE_PCT", Decimal("0.025"))
    @patch("config.CONTRARIAN_LEVERAGE", 2)
    @patch("config.CONTRARIAN_STOP_LOSS_PCT", Decimal("0.025"))
    @patch("config.CONTRARIAN_TAKE_PROFIT_RATIO", Decimal("2.5"))
    @patch("config.FUNDING_MARGIN_TYPE", "ISOLATED")
    def test_short_side_is_sell(self):
        """SHORT signal must map to exchange side 'SELL'."""
        strategy = self._make_strategy()
        signal = self._make_signal(direction_str="SHORT", mark_price="10.00")

        params = strategy.calculate_position_params(signal, Decimal("10000.00"))

        self.assertEqual(params["side"], "SELL")


if __name__ == "__main__":
    unittest.main()
