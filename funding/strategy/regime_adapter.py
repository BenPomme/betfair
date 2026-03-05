"""
RegimeAdapter: translates the current volatility regime into parameter adjustments
for the contrarian strategy.

TIER 3 — FINANCIALLY CRITICAL.

A bug here modifies position sizing or stop-loss thresholds in the wrong direction,
leading to outsized losses (oversized positions in crisis) or missed trades (too-tight
stops in low vol). Every multiplier must be applied correctly.

Regime definitions (shared with RegimeHMM / RegimeTransformer):
    0 — low volatility  : scale up size, tighten stops, relax threshold
    1 — medium (normal) : no adjustment (1.0 multipliers on all params)
    2 — high volatility : reduce size, widen stops, raise threshold
    3 — crisis          : halt trading entirely (size_mult = 0.0)

The regime model (RegimeHMM or RegimeTransformer) is injected at construction time
and called via the shared predict_regime(features) interface.

Usage:
    adapter = RegimeAdapter(regime_model=selector.get_model())
    regime = adapter.update_regime(feature_df)
    if adapter.should_halt_trading():
        return
    adjusted = adapter.adjust_params({
        "size": Decimal("1000"),
        "stop_loss": Decimal("0.025"),
        "threshold": Decimal("0.0005"),
    })
"""
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

import pandas as pd

import config

logger = logging.getLogger(__name__)

# Regime labels (must match RegimeHMM / RegimeTransformer convention)
_REGIME_LOW = 0
_REGIME_MEDIUM = 1
_REGIME_HIGH = 2
_REGIME_CRISIS = 3

# Parameter keys that receive multiplicative adjustments.
# "size"      -> position notional / quantity
# "stop_loss" -> stop-loss distance as a fraction of entry price
# "threshold" -> minimum signal threshold (funding rate, confidence, etc.)
_SIZE_KEY = "size"
_STOP_KEY = "stop_loss"
_THRESHOLD_KEY = "threshold"


class RegimeAdapter:
    """
    Translate the current volatility regime into parameter adjustments.

    Adjustments are multiplicative: adjusted_value = base_value * multiplier.

    Regime 3 (crisis) sets size_mult = 0.0, which reduces any position size to
    zero, effectively halting new entries.  should_halt_trading() additionally
    checks config.REGIME_CRISIS_HALT_TRADING so operators can override via env.

    Attributes:
        _model          : Fitted RegimeHMM or RegimeTransformer (or None for default).
        _current_regime : Most recently predicted regime label (int, 0-3).
        _last_update    : UTC datetime of the last update_regime() call.
        _adjustments    : Dict mapping regime int -> {"size_mult", "stop_mult", "threshold_mult"}.
    """

    def __init__(self, regime_model=None) -> None:
        self._model = regime_model  # RegimeHMM or RegimeTransformer
        self._current_regime: int = _REGIME_MEDIUM  # default: medium
        self._last_update: Optional[datetime] = None

        # Regime adjustment multipliers — Tier 3 reviewed values.
        # Rationale:
        #   Regime 0 (low vol): markets are calm, slightly enlarge positions
        #     and shrink stop distance; accept slightly lower signal threshold.
        #   Regime 1 (medium): baseline, no adjustments.
        #   Regime 2 (high vol): reduce exposure, give positions more room,
        #     require a stronger signal to enter.
        #   Regime 3 (crisis): do not open new positions (size_mult=0.0);
        #     widen stops on existing ones and require extreme signal strength.
        self._adjustments = {
            _REGIME_LOW: {
                "size_mult":      Decimal("1.2"),
                "stop_mult":      Decimal("0.8"),
                "threshold_mult": Decimal("0.9"),
            },
            _REGIME_MEDIUM: {
                "size_mult":      Decimal("1.0"),
                "stop_mult":      Decimal("1.0"),
                "threshold_mult": Decimal("1.0"),
            },
            _REGIME_HIGH: {
                "size_mult":      Decimal("0.6"),
                "stop_mult":      Decimal("1.5"),
                "threshold_mult": Decimal("1.3"),
            },
            _REGIME_CRISIS: {
                "size_mult":      Decimal("0.0"),   # no new positions in crisis
                "stop_mult":      Decimal("2.0"),
                "threshold_mult": Decimal("2.0"),
            },
        }

    # ------------------------------------------------------------------
    # Regime update
    # ------------------------------------------------------------------

    def update_regime(self, features: pd.DataFrame) -> int:
        """
        Predict the current volatility regime from the latest feature data.

        Calls model.predict_regime(features) (shared interface for both
        RegimeHMM and RegimeTransformer), updates _current_regime and
        _last_update.

        Args:
            features: DataFrame with columns matching those used during model
                      training (from build_regime_features()).  For the
                      Transformer, at least window_size rows must be provided.

        Returns:
            Current regime label (int 0-3).
        """
        if self._model is None:
            logger.debug(
                "update_regime: no model provided; keeping default regime %d",
                self._current_regime,
            )
            return self._current_regime

        try:
            regime = int(self._model.predict_regime(features))
        except Exception as exc:
            logger.warning(
                "update_regime: predict_regime() raised %s — "
                "keeping previous regime %d",
                exc,
                self._current_regime,
            )
            return self._current_regime

        if regime not in self._adjustments:
            logger.warning(
                "update_regime: unknown regime label %d returned by model; "
                "clamping to MEDIUM (%d)",
                regime,
                _REGIME_MEDIUM,
            )
            regime = _REGIME_MEDIUM

        prev = self._current_regime
        self._current_regime = regime
        self._last_update = datetime.now(timezone.utc)

        if regime != prev:
            logger.info(
                "Regime changed: %d -> %d (low=0, medium=1, high=2, crisis=3)",
                prev,
                regime,
            )
        else:
            logger.debug("Regime unchanged: %d", regime)

        return self._current_regime

    # ------------------------------------------------------------------
    # Parameter adjustment
    # ------------------------------------------------------------------

    def adjust_params(self, base_params: dict) -> dict:
        """
        Apply regime multipliers to the base parameter dict.

        Recognized keys (adjusted multiplicatively):
            "size"       : position size / notional (Decimal or numeric)
            "stop_loss"  : stop-loss distance as a fraction (Decimal or numeric)
            "threshold"  : signal threshold (Decimal or numeric)

        All other keys are passed through unchanged.

        Args:
            base_params: Dict of strategy parameters (values may be Decimal,
                         float, or int).  A copy is returned; the original
                         is never mutated.

        Returns:
            New dict with adjusted values.  Adjusted values are returned as
            Decimal to preserve precision.

        Example:
            base = {"size": Decimal("1000"), "stop_loss": Decimal("0.025"),
                    "threshold": Decimal("0.0005"), "symbol": "BTCUSDT"}
            adj  = adapter.adjust_params(base)
            # adj["size"] == Decimal("600") in regime 2
        """
        mults = self._adjustments.get(self._current_regime, self._adjustments[_REGIME_MEDIUM])
        adjusted = dict(base_params)

        # Apply size multiplier
        if _SIZE_KEY in adjusted:
            adjusted[_SIZE_KEY] = Decimal(str(adjusted[_SIZE_KEY])) * mults["size_mult"]

        # Apply stop-loss multiplier
        if _STOP_KEY in adjusted:
            adjusted[_STOP_KEY] = Decimal(str(adjusted[_STOP_KEY])) * mults["stop_mult"]

        # Apply threshold multiplier
        if _THRESHOLD_KEY in adjusted:
            adjusted[_THRESHOLD_KEY] = Decimal(str(adjusted[_THRESHOLD_KEY])) * mults["threshold_mult"]

        logger.debug(
            "adjust_params: regime=%d, size_mult=%s, stop_mult=%s, threshold_mult=%s",
            self._current_regime,
            mults["size_mult"],
            mults["stop_mult"],
            mults["threshold_mult"],
        )
        return adjusted

    # ------------------------------------------------------------------
    # Halt flag
    # ------------------------------------------------------------------

    def should_halt_trading(self) -> bool:
        """
        Return True if trading should be halted due to a crisis regime.

        Halt condition:
            current_regime == 3 (crisis) AND config.REGIME_CRISIS_HALT_TRADING is True.

        The config flag allows operators to disable the automatic halt by
        setting REGIME_CRISIS_HALT_TRADING=false in the environment.

        Returns:
            bool — True means do not open new positions.
        """
        if self._current_regime == _REGIME_CRISIS and config.REGIME_CRISIS_HALT_TRADING:
            logger.warning(
                "should_halt_trading: CRISIS regime detected and "
                "REGIME_CRISIS_HALT_TRADING=true — halting new entries"
            )
            return True
        return False

    # ------------------------------------------------------------------
    # State inspection
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        """
        Return the current adapter state for logging and dashboard display.

        Returns:
            Dict with keys:
                current_regime (int)      : 0-3
                regime_label  (str)       : "low" | "medium" | "high" | "crisis"
                last_update   (str|None)  : ISO 8601 UTC string, or None
                regime_proba  (list|None) : Softmax probabilities [p0, p1, p2, p3]
                                            if the model supports predict_regime_proba()
                                            and at least one feature row is available.
                halt_trading  (bool)      : current value of should_halt_trading()
                adjustments   (dict)      : current multiplier set for current_regime
        """
        _labels = {
            _REGIME_LOW:    "low",
            _REGIME_MEDIUM: "medium",
            _REGIME_HIGH:   "high",
            _REGIME_CRISIS: "crisis",
        }
        mults = self._adjustments.get(self._current_regime, self._adjustments[_REGIME_MEDIUM])
        state: dict = {
            "current_regime": self._current_regime,
            "regime_label":   _labels.get(self._current_regime, "unknown"),
            "last_update":    self._last_update.isoformat() if self._last_update else None,
            "regime_proba":   None,
            "halt_trading":   self.should_halt_trading(),
            "adjustments": {
                "size_mult":      str(mults["size_mult"]),
                "stop_mult":      str(mults["stop_mult"]),
                "threshold_mult": str(mults["threshold_mult"]),
            },
        }

        # Attempt to include softmax probabilities if the model supports it
        if self._model is not None and hasattr(self._model, "predict_regime_proba"):
            # Probabilities require features — only include if already cached
            # by the caller (we cannot call predict here without features).
            # Callers that want proba should call model.predict_regime_proba(features)
            # directly and merge into this dict.
            pass

        return state
