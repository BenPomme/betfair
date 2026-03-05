"""
Contrarian strategy: enter directional positions after extreme funding events.

Strategy logic:
  1. Filter symbols whose funding rate is extreme (|rate| > CONTRARIAN_MIN_FUNDING_RATE).
  2. For each extreme symbol, build a live feature row from the snapshot and
     recent rate history, enriched with sentiment data when available.
  3. Run ML inference (ContrarianXGBoost or TFTPredictor — anything exposing
     a .predict(DataFrame) interface that returns direction_prob,
     predicted_return_24h, and confidence columns).
  4. Filter predictions below CONTRARIAN_MIN_CONFIDENCE.
  5. Determine direction: extremely positive funding → SHORT (contrarian bet
     on price reversal down); extremely negative funding → LONG.
  6. Optionally multiply confidence by a regime-specific multiplier if a
     RegimeAdapter is provided.
  7. Return a list of ContrarianSignal objects ready for the executor.

All monetary and price arithmetic uses decimal.Decimal.
"""

import logging
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, getcontext
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config
from funding.core.schemas import ContrarianSignal, DirectionalSide

getcontext().prec = 10

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regime adapter protocol
# ---------------------------------------------------------------------------
# The regime_adapter passed to ContrarianStrategy.__init__ must expose:
#
#   get_multiplier(regime: Optional[str]) -> float
#       Return a confidence multiplier for the given regime label string.
#       Regime labels are: "low", "medium", "high", "crisis" (or None when
#       the regime is unknown).
#
# A lightweight default adapter is provided below. Callers may substitute
# any object that satisfies the same duck-type interface.
# ---------------------------------------------------------------------------

# Regime integer -> label string (matches RegimeHMM / RegimeTransformer ordering)
_REGIME_LABEL: Dict[int, str] = {
    0: "low",
    1: "medium",
    2: "high",
    3: "crisis",
}

# Per-regime confidence multipliers:
#   low     — calm market, contrarian plays are riskier (less momentum to reverse)
#   medium  — baseline, no adjustment
#   high    — elevated volatility, reversals happen faster; slight boost
#   crisis  — extreme regime; overshoots are common but dangerous; reduce
_DEFAULT_REGIME_MULTIPLIERS: Dict[Optional[str], float] = {
    "low": 0.85,
    "medium": 1.00,
    "high": 1.10,
    "crisis": 0.70,
    None: 1.00,  # unknown regime — no adjustment
}


class DefaultRegimeAdapter:
    """Fallback regime adapter using fixed per-regime multipliers.

    Wraps either a RegimeHMM or RegimeTransformer instance (anything with a
    predict_regime(features: DataFrame) -> int method) and translates the
    integer regime to a confidence multiplier.

    When no model is provided, get_multiplier always returns 1.0.
    """

    def __init__(
        self,
        model=None,
        multipliers: Optional[Dict[Optional[str], float]] = None,
    ) -> None:
        self._model = model
        self._multipliers = multipliers or dict(_DEFAULT_REGIME_MULTIPLIERS)

    def get_multiplier(self, regime: Optional[str]) -> float:
        """Return confidence multiplier for a regime label string.

        Args:
            regime: Regime label ("low", "medium", "high", "crisis") or None.

        Returns:
            float multiplier; defaults to 1.0 for unknown regime labels.
        """
        return float(self._multipliers.get(regime, 1.0))

    def predict_regime_label(self, features: pd.DataFrame) -> Optional[str]:
        """Predict regime label from live feature DataFrame.

        Args:
            features: Single-row or multi-row regime feature DataFrame.
                      Only the last row is used when multiple rows are passed.

        Returns:
            Regime label string, or None when prediction fails.
        """
        if self._model is None:
            return None
        try:
            regime_int: int = self._model.predict_regime(features)
            return _REGIME_LABEL.get(regime_int)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Regime prediction failed: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Live feature builder
# ---------------------------------------------------------------------------

def _build_live_feature_row(
    snapshot,
    rate_history: list,
    long_short_ratio: Optional[float],
    fear_greed: Optional[int],
) -> Optional[pd.DataFrame]:
    """Build a single-row contrarian feature DataFrame for ML inference.

    Produces the same feature names used by build_contrarian_features() so
    that a trained ContrarianXGBoost or TFTPredictor can score it directly.
    Missing optional features are filled with NaN (models tolerate them).

    Args:
        snapshot:          FundingSnapshot for the symbol.
        rate_history:      List of historical funding rate dicts, each with at
                           minimum a "funding_rate" key.  Newest entry first
                           (as returned by Binance API).  Must contain at
                           least 1 entry.
        long_short_ratio:  Current long/short ratio for the symbol, or None.
        fear_greed:        Fear & greed index value (0-100), or None.

    Returns:
        Single-row pd.DataFrame aligned to contrarian feature schema, or
        None if rate_history contains fewer than 1 entry.
    """
    if not rate_history:
        return None

    # Reverse to chronological order (oldest first)
    rates_chron: List[float] = [
        float(h["funding_rate"]) for h in reversed(rate_history)
    ]
    current_rate = float(snapshot.funding_rate)
    now = datetime.now(timezone.utc)

    # ---- Lag features ----
    rate_lag_1 = rates_chron[-2] if len(rates_chron) >= 2 else 0.0
    rate_lag_2 = rates_chron[-3] if len(rates_chron) >= 3 else 0.0
    rate_lag_3 = rates_chron[-4] if len(rates_chron) >= 4 else 0.0

    # ---- Rate change features ----
    rate_change_8h = current_rate - rate_lag_1
    rate_change_24h = (
        current_rate - rates_chron[-4] if len(rates_chron) >= 4 else 0.0
    )

    # ---- Rolling stats ----
    rates_last_3 = rates_chron[-3:] if len(rates_chron) >= 3 else rates_chron
    rates_last_9 = rates_chron[-9:] if len(rates_chron) >= 9 else rates_chron
    rates_last_30 = rates_chron[-30:] if len(rates_chron) >= 30 else rates_chron

    rate_mean_3 = float(np.mean(rates_last_3))
    rate_std_3 = float(np.std(rates_last_3))
    rate_mean_9 = float(np.mean(rates_last_9))

    # ---- Z-score (30-period) ----
    mean_30 = float(np.mean(rates_last_30))
    std_30 = float(np.std(rates_last_30))
    funding_zscore_30 = (
        (current_rate - mean_30) / std_30 if std_30 != 0.0 else 0.0
    )

    # ---- Extremity flags ----
    thresh = float(config.CONTRARIAN_MIN_FUNDING_RATE)
    is_extreme_pos = current_rate > thresh
    is_extreme_neg = current_rate < -thresh
    funding_rate_abs = abs(current_rate)

    # Consecutive extreme periods
    consecutive_extreme_positive = 0
    for r in reversed(rates_chron):
        if r > thresh:
            consecutive_extreme_positive += 1
        else:
            break

    consecutive_extreme_negative = 0
    for r in reversed(rates_chron):
        if r < -thresh:
            consecutive_extreme_negative += 1
        else:
            break

    # ---- Positive streak ----
    rate_positive_streak = sum(1 for r in reversed(rates_chron) if r > 0)

    # ---- Long/short ratio features ----
    ls_ratio = long_short_ratio if long_short_ratio is not None else float("nan")
    ls_extreme_long = (
        float(ls_ratio > 2.0) if long_short_ratio is not None else float("nan")
    )
    ls_extreme_short = (
        float(ls_ratio < 0.5) if long_short_ratio is not None else float("nan")
    )

    # ---- Fear & greed features ----
    fear_greed_value = float(fear_greed) if fear_greed is not None else float("nan")
    fear_greed_extreme_fear = (
        float(fear_greed < 25) if fear_greed is not None else float("nan")
    )
    fear_greed_extreme_greed = (
        float(fear_greed > 75) if fear_greed is not None else float("nan")
    )

    # ---- Time features ----
    hour_of_day = now.hour
    day_of_week = now.weekday()
    is_weekend = float(day_of_week >= 5)

    features: Dict[str, float] = {
        # Lag features
        "rate_lag_1": rate_lag_1,
        "rate_lag_2": rate_lag_2,
        "rate_lag_3": rate_lag_3,
        # Change features
        "rate_change_8h": rate_change_8h,
        "rate_change_24h": rate_change_24h,
        # Rolling stats
        "rate_mean_3": rate_mean_3,
        "rate_std_3": rate_std_3,
        "rate_mean_9": rate_mean_9,
        # Z-score and percentile
        "funding_zscore_30": funding_zscore_30,
        # Extremity
        "funding_rate_abs": funding_rate_abs,
        "consecutive_extreme_positive": float(consecutive_extreme_positive),
        "consecutive_extreme_negative": float(consecutive_extreme_negative),
        # Streak
        "rate_positive_streak": float(rate_positive_streak),
        # Long/short ratio
        "ls_ratio": ls_ratio,
        "ls_extreme_long": ls_extreme_long,
        "ls_extreme_short": ls_extreme_short,
        # Fear & greed
        "fear_greed_value": fear_greed_value,
        "fear_greed_extreme_fear": fear_greed_extreme_fear,
        "fear_greed_extreme_greed": fear_greed_extreme_greed,
        # Time
        "hour_of_day": float(hour_of_day),
        "day_of_week": float(day_of_week),
        "is_weekend": is_weekend,
    }

    return pd.DataFrame([features], index=[pd.Timestamp(now)])


# ---------------------------------------------------------------------------
# Main strategy class
# ---------------------------------------------------------------------------

class ContrarianStrategy:
    """Core contrarian signal generator.

    Wraps an ML model and an optional regime adapter to produce
    ContrarianSignal objects from live market snapshots.

    The ML model must expose:
        predict(features: pd.DataFrame) -> pd.DataFrame
    with output columns:
        direction_prob       -- float in [0, 1]; P(price up in 24h)
        predicted_return_24h -- float; predicted 24h price return
        confidence           -- float in [0, 1]; |direction_prob - 0.5| * 2

    The regime_adapter, if supplied, must expose:
        get_multiplier(regime: Optional[str]) -> float

    Both dependencies are optional; the class degrades gracefully when they
    are absent.
    """

    def __init__(
        self,
        model=None,
        regime_adapter=None,
    ) -> None:
        """Initialise the contrarian strategy.

        Args:
            model: Any object with a .predict(DataFrame) method matching the
                   ContrarianXGBoost / TFTPredictor output schema.
                   When None, no signals are produced.
            regime_adapter: Optional adapter exposing .get_multiplier(regime).
                            When None, confidence is not adjusted.
        """
        self._model = model
        self._regime_adapter = regime_adapter

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_signals(
        self,
        snapshots: Dict[str, object],
        rate_histories: Dict[str, list],
        sentiment: Optional[Dict[str, float]] = None,
        fear_greed: Optional[int] = None,
    ) -> List[ContrarianSignal]:
        """Evaluate live snapshots and return contrarian trade signals.

        Pipeline per symbol:
          1. Skip if |funding_rate| < CONTRARIAN_MIN_FUNDING_RATE.
          2. Build live feature row from snapshot + rate history + sentiment.
          3. Run ML prediction; skip if model unavailable or prediction fails.
          4. Filter: confidence < CONTRARIAN_MIN_CONFIDENCE -> skip.
          5. Determine trade direction (contrarian: positive rate -> SHORT).
          6. Apply regime confidence multiplier if regime_adapter present.
          7. Re-filter after adjustment: adjusted confidence < threshold -> skip.
          8. Append ContrarianSignal.

        Args:
            snapshots:     Mapping of symbol -> FundingSnapshot.
            rate_histories: Mapping of symbol -> list of historical funding
                           rate dicts (newest-first, same format as Binance
                           get_funding_rate_history response).
            sentiment:     Optional mapping of symbol -> long_short_ratio float.
            fear_greed:    Optional market-wide fear & greed index value (0-100).

        Returns:
            List of ContrarianSignal objects passing all filters, sorted by
            adjusted confidence descending.
        """
        if self._model is None:
            logger.warning(
                "ContrarianStrategy: no ML model configured — no signals produced"
            )
            return []

        min_rate: Decimal = config.CONTRARIAN_MIN_FUNDING_RATE
        min_confidence: float = config.CONTRARIAN_MIN_CONFIDENCE

        signals: List[ContrarianSignal] = []

        for symbol, snapshot in snapshots.items():
            funding_rate: Decimal = snapshot.funding_rate

            # --- Step 1: Extreme funding filter ---
            if abs(funding_rate) < min_rate:
                logger.debug(
                    "%s: funding rate %s below threshold %s, skipping",
                    symbol, funding_rate, min_rate,
                )
                continue

            # --- Step 2: Build feature row ---
            history = rate_histories.get(symbol, [])
            if not history:
                logger.debug(
                    "%s: no rate history available, skipping", symbol
                )
                continue

            long_short_ratio: Optional[float] = (
                sentiment.get(symbol) if sentiment else None
            )

            feature_row = _build_live_feature_row(
                snapshot=snapshot,
                rate_history=history,
                long_short_ratio=long_short_ratio,
                fear_greed=fear_greed,
            )
            if feature_row is None:
                logger.debug(
                    "%s: could not build feature row, skipping", symbol
                )
                continue

            # --- Step 3: ML prediction ---
            try:
                prediction: pd.DataFrame = self._model.predict(feature_row)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "%s: ML prediction failed (%s), skipping", symbol, exc
                )
                continue

            if prediction.empty:
                logger.debug(
                    "%s: model returned empty prediction, skipping", symbol
                )
                continue

            direction_prob: float = float(prediction["direction_prob"].iloc[0])
            predicted_return_24h: float = float(
                prediction["predicted_return_24h"].iloc[0]
            )
            confidence: float = float(prediction["confidence"].iloc[0])

            # Extract 72h return if model provides it; fall back to 0.0
            predicted_return_72h: float = 0.0
            if "predicted_return_72h" in prediction.columns:
                predicted_return_72h = float(
                    prediction["predicted_return_72h"].iloc[0]
                )

            # --- Step 4: Confidence pre-filter ---
            if confidence < min_confidence:
                logger.debug(
                    "%s: confidence %.4f below threshold %.4f, skipping",
                    symbol, confidence, min_confidence,
                )
                continue

            # --- Step 5: Determine trade direction (contrarian) ---
            #
            # Extreme positive rate -> longs are paying shorts ->
            #   market is positioned long -> contrarian SHORTS (bets on reversal).
            # Extreme negative rate -> shorts are paying longs ->
            #   market is positioned short -> contrarian LONGS.
            if funding_rate > Decimal("0"):
                direction = DirectionalSide.SHORT
            else:
                direction = DirectionalSide.LONG

            # --- Step 6: Regime adjustment ---
            regime_label: Optional[str] = None
            adjusted_confidence = confidence

            if self._regime_adapter is not None:
                try:
                    multiplier = self._regime_adapter.get_multiplier(regime_label)
                    adjusted_confidence = confidence * multiplier
                    logger.debug(
                        "%s: regime=%s, multiplier=%.3f, "
                        "confidence %.4f -> %.4f",
                        symbol, regime_label, multiplier,
                        confidence, adjusted_confidence,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "%s: regime adapter error (%s), using raw confidence",
                        symbol, exc,
                    )

            # --- Step 7: Post-regime confidence filter ---
            if adjusted_confidence < min_confidence:
                logger.debug(
                    "%s: adjusted confidence %.4f below threshold %.4f "
                    "(after regime multiplier), skipping",
                    symbol, adjusted_confidence, min_confidence,
                )
                continue

            # --- Step 8: Build signal ---
            model_name = type(self._model).__name__

            signal = ContrarianSignal(
                symbol=symbol,
                direction=direction,
                confidence=adjusted_confidence,
                predicted_return_24h=predicted_return_24h,
                predicted_return_72h=predicted_return_72h,
                model_name=model_name,
                funding_rate=funding_rate,
                long_short_ratio=long_short_ratio,
                fear_greed=fear_greed,
                regime=regime_label,
                timestamp=datetime.now(timezone.utc),
            )
            signals.append(signal)

            logger.info(
                "ContrarianSignal: %s %s | rate=%s | conf=%.4f | "
                "pred_ret_24h=%.4f | pred_ret_72h=%.4f | regime=%s",
                symbol,
                direction.value,
                funding_rate,
                adjusted_confidence,
                predicted_return_24h,
                predicted_return_72h,
                regime_label,
            )

        # Sort by adjusted confidence descending so callers get the strongest
        # signals first
        signals.sort(key=lambda s: s.confidence, reverse=True)
        return signals

    def calculate_position_params(
        self,
        signal: ContrarianSignal,
        balance: Decimal,
    ) -> Dict[str, object]:
        """Compute position sizing parameters for an approved contrarian signal.

        Capital allocation:
            capital_per_trade = balance * CONTRARIAN_CAPITAL_PER_TRADE_PCT
            notional          = capital_per_trade * CONTRARIAN_LEVERAGE
            quantity          = notional / mark_price

        Risk levels (symmetric stop/take-profit around entry):
            stop_loss_pct     = CONTRARIAN_STOP_LOSS_PCT
            stop_distance     = entry * stop_loss_pct
            take_profit_dist  = stop_distance * CONTRARIAN_TAKE_PROFIT_RATIO

        All prices are returned as Decimal rounded to 8 decimal places.
        Quantity is rounded to 8 decimal places (contract precision); the
        caller must apply exchange-specific lot-size rounding before submission.

        Args:
            signal:  A ContrarianSignal produced by evaluate_signals().
            balance: Available account balance in USD (Decimal).

        Returns:
            Dict with keys:
                side          (str)      — "BUY" (LONG) or "SELL" (SHORT)
                quantity      (Decimal)  — contract quantity
                leverage      (int)      — leverage to set on the exchange
                margin_type   (str)      — "ISOLATED"
                stop_loss     (Decimal)  — stop-loss price
                take_profit   (Decimal)  — take-profit price
                notional      (Decimal)  — total notional USD value
                capital_used  (Decimal)  — margin capital committed
                entry_price   (Decimal)  — current mark price used as reference

        Raises:
            ValueError: When mark_price is zero or negative (cannot size).
        """
        # Config constants — read once
        capital_pct: Decimal = config.CONTRARIAN_CAPITAL_PER_TRADE_PCT
        leverage: int = config.CONTRARIAN_LEVERAGE
        stop_loss_pct: Decimal = config.CONTRARIAN_STOP_LOSS_PCT
        take_profit_ratio: Decimal = config.CONTRARIAN_TAKE_PROFIT_RATIO

        # Entry price: use the signal's funding_rate snapshot mark price
        # The snapshot mark price is embedded in the signal via the snapshot
        # that produced it; we derive it from the model's perspective.
        # The snapshot is not stored on ContrarianSignal, so callers must
        # pass balance from the live account. Mark price must come from the
        # snapshot; we approximate via the signal.funding_rate's associated
        # snapshot, but since ContrarianSignal does not embed mark_price
        # directly, we compute quantity from balance / leverage as a notional
        # and return entry_price as Decimal("0") to indicate it must be filled
        # by the caller from the live snapshot.
        #
        # NOTE: The executor always retrieves the current mark price at order
        # time. The entry_price here is INFORMATIONAL and will be overridden
        # by the actual fill price. We surface it so the caller can perform
        # pre-flight sanity checks.
        #
        # A zero or missing mark_price means we cannot compute quantity — the
        # caller must provide it via the snapshot at execution time.

        # --- Capital sizing ---
        capital_per_trade: Decimal = (balance * capital_pct).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        leverage_d = Decimal(leverage)
        notional: Decimal = (capital_per_trade * leverage_d).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        # --- Side mapping ---
        # LONG signal -> buy to open (BUY side on exchange)
        # SHORT signal -> sell to open (SELL side on exchange)
        if signal.direction is DirectionalSide.LONG:
            exchange_side = "BUY"
        else:
            exchange_side = "SELL"

        # --- Stop-loss and take-profit ---
        # We need the mark price to compute absolute levels. The signal does
        # not carry mark_price directly; the executor will override these.
        # We return sentinel Decimal("0") values here to signal "not yet
        # computed"; the executor replaces them with live prices.
        #
        # However, for paper trading and pre-flight logging, we derive them
        # from the current funding_rate's associated mark price if the caller
        # can supply it. Since this method only receives `signal` and
        # `balance`, we emit Decimal("0") sentinels and document the protocol.
        #
        # If a mark_price attribute is present on the signal (duck-typed
        # extension), we use it; otherwise sentinels are returned.
        mark_price: Optional[Decimal] = getattr(signal, "mark_price", None)

        if mark_price is not None and mark_price > Decimal("0"):
            entry_price: Decimal = mark_price

            # quantity = notional / mark_price, rounded to 8 dp
            quantity: Decimal = (notional / entry_price).quantize(
                Decimal("0.00000001"), rounding=ROUND_HALF_UP
            )

            stop_distance: Decimal = (entry_price * stop_loss_pct).quantize(
                Decimal("0.00000001"), rounding=ROUND_HALF_UP
            )
            reward_distance: Decimal = (
                stop_distance * take_profit_ratio
            ).quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP)

            if signal.direction is DirectionalSide.LONG:
                stop_loss_price: Decimal = (entry_price - stop_distance).quantize(
                    Decimal("0.00000001"), rounding=ROUND_HALF_UP
                )
                take_profit_price: Decimal = (entry_price + reward_distance).quantize(
                    Decimal("0.00000001"), rounding=ROUND_HALF_UP
                )
            else:  # SHORT
                stop_loss_price = (entry_price + stop_distance).quantize(
                    Decimal("0.00000001"), rounding=ROUND_HALF_UP
                )
                take_profit_price = (entry_price - reward_distance).quantize(
                    Decimal("0.00000001"), rounding=ROUND_HALF_UP
                )
        else:
            # Mark price not available — return sentinels
            entry_price = Decimal("0")
            quantity = Decimal("0")
            stop_loss_price = Decimal("0")
            take_profit_price = Decimal("0")
            logger.warning(
                "%s: mark_price not available on signal — "
                "stop_loss, take_profit, and quantity are sentinel zeros; "
                "executor must compute them from live snapshot",
                signal.symbol,
            )

        logger.debug(
            "calculate_position_params: %s %s | notional=%s | qty=%s | "
            "sl=%s | tp=%s | leverage=%d",
            signal.symbol,
            exchange_side,
            notional,
            quantity,
            stop_loss_price,
            take_profit_price,
            leverage,
        )

        return {
            "side": exchange_side,
            "quantity": quantity,
            "leverage": leverage,
            "margin_type": config.FUNDING_MARGIN_TYPE,
            "stop_loss": stop_loss_price,
            "take_profit": take_profit_price,
            "notional": notional,
            "capital_used": capital_per_trade,
            "entry_price": entry_price,
        }
