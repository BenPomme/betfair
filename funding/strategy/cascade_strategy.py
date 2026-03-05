"""
CascadeStrategy: defensive actions during fragile market states.

TIER 3 — FINANCIALLY CRITICAL.

A cascade event is a rapid, self-reinforcing liquidation spiral where forced
selling drives prices lower, triggering further liquidations. During fragile
market states (high OI concentration, extreme funding, leverage saturation),
a single large liquidation can cascade across symbols.

This strategy module:
  1. Consumes features from the cascade feature pipeline.
  2. Calls a CascadePredictor model to estimate fragility probability and
     severity.
  3. Maps model output to defensive actions (alert-only vs. size reduction).
  4. Identifies the most vulnerable symbols (highest OI concentration or most
     extreme funding rate) to prioritise reduction.

Defensive action thresholds (from config):
  CASCADE_ALERT_THRESHOLD  (default 0.6) — alert operator, no size change
  CASCADE_ACTION_THRESHOLD (default 0.8) — alert + reduce open positions by 50%

The CascadePredictor model must expose:
    predict_fragility(features: pd.DataFrame) -> dict
with keys:
    probability        -- float in [0, 1]; P(cascade within monitoring window)
    severity           -- float in [0, 1]; expected market impact if cascade occurs
    vulnerable_symbols -- List[str]; symbols identified as most at risk
                          (may be empty if the model does not provide them)

When vulnerable_symbols is absent from the model output, the strategy falls
back to identifying them from a 'symbol' column in the features DataFrame
ranked by 'oi_concentration' descending (then 'funding_rate_abs' descending).
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd

import config

logger = logging.getLogger(__name__)


class CascadeStrategy:
    """Evaluate market fragility and emit defensive actions.

    The class is intentionally stateless between evaluations except for the
    last-seen fragility/severity scalars and update timestamp stored for
    dashboard and logging purposes.

    Attributes:
        _model:           CascadePredictor instance (or None when disabled).
        _last_fragility:  Probability score from the most recent evaluation.
        _last_severity:   Severity score from the most recent evaluation.
        _last_update:     UTC datetime of the most recent evaluation, or None.
    """

    def __init__(self, model=None) -> None:
        """Initialise the cascade strategy.

        Args:
            model: Any object with a predict_fragility(pd.DataFrame) -> dict
                   interface.  When None, evaluate_fragility() returns a
                   zero-probability result and no defensive actions are taken.
        """
        self._model = model
        self._last_fragility: float = 0.0
        self._last_severity: float = 0.0
        self._last_update: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Primary evaluation pipeline
    # ------------------------------------------------------------------

    def evaluate_fragility(self, features: pd.DataFrame) -> dict:
        """Evaluate current market fragility from a feature snapshot.

        Calls model.predict_fragility(features) and enriches the result with
        a timestamp and a fallback vulnerable-symbols derivation when the
        model does not provide one.

        The internal state (_last_fragility, _last_severity, _last_update) is
        updated on every successful call, including when the model is absent
        (in which case zeros are stored).

        Args:
            features: DataFrame produced by the cascade feature pipeline.
                      Expected columns (model-dependent; minimum set for
                      fallback vulnerable-symbol ranking):
                        - symbol              (str)
                        - oi_concentration    (float, higher = more concentrated)
                        - funding_rate_abs    (float, absolute funding rate)
                      Additional columns are passed transparently to the model.

        Returns:
            Dict with keys:
                probability          (float)      -- P(cascade) in [0, 1]
                severity             (float)      -- expected impact in [0, 1]
                vulnerable_symbols   (List[str])  -- symbols most at risk
                timestamp            (str)        -- ISO 8601 UTC
        """
        now = datetime.now(timezone.utc)
        iso_now = now.isoformat()

        # --- No model: return safe zero result ---
        if self._model is None:
            logger.warning(
                "CascadeStrategy: no model configured — returning zero fragility"
            )
            self._last_fragility = 0.0
            self._last_severity = 0.0
            self._last_update = now
            return {
                "probability": 0.0,
                "severity": 0.0,
                "vulnerable_symbols": [],
                "timestamp": iso_now,
            }

        # --- Call the model ---
        try:
            raw: dict = self._model.predict_fragility(features)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "CascadeStrategy.evaluate_fragility: model raised %s — "
                "returning zero fragility",
                exc,
            )
            self._last_fragility = 0.0
            self._last_severity = 0.0
            self._last_update = now
            return {
                "probability": 0.0,
                "severity": 0.0,
                "vulnerable_symbols": [],
                "timestamp": iso_now,
            }

        probability: float = float(raw.get("probability", 0.0))
        severity: float = float(raw.get("severity", 0.0))

        # Clamp to [0, 1] — defensive against misbehaving models
        probability = max(0.0, min(1.0, probability))
        severity = max(0.0, min(1.0, severity))

        # --- Vulnerable symbols ---
        # Prefer symbols provided directly by the model; fall back to deriving
        # them from the feature DataFrame ranked by OI concentration (desc)
        # then absolute funding rate (desc).
        model_symbols: Optional[List[str]] = raw.get("vulnerable_symbols")
        if model_symbols is not None:
            vulnerable_symbols: List[str] = list(model_symbols)
        else:
            vulnerable_symbols = self._derive_vulnerable_symbols(features)

        # --- Update state ---
        self._last_fragility = probability
        self._last_severity = severity
        self._last_update = now

        logger.info(
            "CascadeStrategy: probability=%.4f severity=%.4f "
            "vulnerable_symbols=%s",
            probability,
            severity,
            vulnerable_symbols,
        )

        return {
            "probability": probability,
            "severity": severity,
            "vulnerable_symbols": vulnerable_symbols,
            "timestamp": iso_now,
        }

    # ------------------------------------------------------------------
    # Defensive action mapping
    # ------------------------------------------------------------------

    def defensive_actions(
        self,
        fragility: dict,
        open_positions: list,
    ) -> dict:
        """Map a fragility assessment to concrete defensive actions.

        Threshold logic (values read from config at call time):
          probability < CASCADE_ALERT_THRESHOLD:
            No action required.  Return {"alert": False}.

          CASCADE_ALERT_THRESHOLD <= probability < CASCADE_ACTION_THRESHOLD:
            Alert the operator.  No position changes.
            Return {"alert": True, "message": "...", "reduce_sizes": False}.

          probability >= CASCADE_ACTION_THRESHOLD:
            Alert + reduce all open positions by 50%.  Symbols that appear in
            fragility["vulnerable_symbols"] are prioritised in the reduction
            list; remaining open positions follow after.
            Return {"alert": True, "reduce_sizes": True,
                    "symbols_to_reduce": [...], "reduction_pct": 0.5}.

        Position objects in open_positions must expose a .symbol attribute
        (or be dicts with a "symbol" key) to allow symbol extraction.  Objects
        that do not expose a symbol are included in the reduction list via
        their string representation.

        Args:
            fragility:       Output of evaluate_fragility().
            open_positions:  List of live position objects or dicts.

        Returns:
            Dict describing the required defensive action (see above).
        """
        probability: float = float(fragility.get("probability", 0.0))
        vulnerable_symbols: List[str] = list(
            fragility.get("vulnerable_symbols", [])
        )

        alert_threshold: float = config.CASCADE_ALERT_THRESHOLD
        action_threshold: float = config.CASCADE_ACTION_THRESHOLD

        if probability < alert_threshold:
            logger.debug(
                "CascadeStrategy.defensive_actions: probability=%.4f below "
                "alert threshold %.2f — no action",
                probability,
                alert_threshold,
            )
            return {"alert": False}

        if probability < action_threshold:
            message = (
                f"Cascade fragility alert: probability={probability:.4f} "
                f"(threshold={alert_threshold:.2f}). "
                f"Vulnerable symbols: {vulnerable_symbols}. "
                "Monitoring only — no position changes at this level."
            )
            logger.warning("CascadeStrategy: %s", message)
            return {
                "alert": True,
                "message": message,
                "reduce_sizes": False,
            }

        # --- Action threshold reached: build reduction list ---
        open_syms: List[str] = self._extract_symbols(open_positions)

        # Prioritise vulnerable symbols that are actually open, then the rest
        vulnerable_open = [s for s in vulnerable_symbols if s in open_syms]
        remaining_open = [s for s in open_syms if s not in vulnerable_symbols]
        symbols_to_reduce = vulnerable_open + remaining_open

        message = (
            f"Cascade fragility ACTION: probability={probability:.4f} "
            f"(action_threshold={action_threshold:.2f}). "
            f"Severity={fragility.get('severity', 0.0):.4f}. "
            f"Reducing {len(symbols_to_reduce)} position(s) by 50%: "
            f"{symbols_to_reduce}. "
            f"Vulnerable symbols identified: {vulnerable_symbols}."
        )
        logger.warning("CascadeStrategy: %s", message)

        return {
            "alert": True,
            "message": message,
            "reduce_sizes": True,
            "symbols_to_reduce": symbols_to_reduce,
            "reduction_pct": 0.5,
        }

    # ------------------------------------------------------------------
    # Alert gate
    # ------------------------------------------------------------------

    def should_send_alert(self, fragility: dict) -> bool:
        """Return True if the fragility probability warrants an operator alert.

        An alert should be sent whenever probability >= CASCADE_ALERT_THRESHOLD
        (covers both the alert-only and the action level).

        Args:
            fragility: Output of evaluate_fragility().

        Returns:
            bool — True means send an alert (Telegram, PagerDuty, etc.).
        """
        probability: float = float(fragility.get("probability", 0.0))
        return probability >= config.CASCADE_ALERT_THRESHOLD

    # ------------------------------------------------------------------
    # State inspection
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        """Return the current strategy state for logging and dashboard display.

        Returns:
            Dict with keys:
                last_fragility  (float)      -- most recent probability score
                last_severity   (float)      -- most recent severity score
                last_update     (str|None)   -- ISO 8601 UTC, or None if never run
        """
        return {
            "last_fragility": self._last_fragility,
            "last_severity": self._last_severity,
            "last_update": (
                self._last_update.isoformat() if self._last_update else None
            ),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _derive_vulnerable_symbols(self, features: pd.DataFrame) -> List[str]:
        """Derive vulnerable symbols from the feature DataFrame.

        Ranks symbols by:
          1. oi_concentration descending (higher = more concentrated risk)
          2. funding_rate_abs descending (higher = more extreme positioning)

        Only symbols present in a "symbol" column are considered.  If neither
        ranking column is available, returns an empty list.

        Args:
            features: DataFrame from the cascade feature pipeline.

        Returns:
            Ordered list of symbol strings (most vulnerable first).
        """
        if features.empty or "symbol" not in features.columns:
            return []

        rank_cols = [
            col for col in ("oi_concentration", "funding_rate_abs")
            if col in features.columns
        ]

        if not rank_cols:
            # No ranking columns — return symbols in original order
            return list(features["symbol"].dropna().unique())

        try:
            ranked = features.sort_values(
                by=rank_cols,
                ascending=[False] * len(rank_cols),
            )
            return list(ranked["symbol"].dropna().tolist())
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "CascadeStrategy._derive_vulnerable_symbols: sort failed (%s); "
                "returning unsorted symbols",
                exc,
            )
            return list(features["symbol"].dropna().tolist())

    @staticmethod
    def _extract_symbols(open_positions: list) -> List[str]:
        """Extract symbol strings from a heterogeneous list of position objects.

        Supports:
          - Objects with a .symbol attribute  (HedgePosition, DirectionalPosition)
          - Dicts with a "symbol" key
          - Anything else: str(item) used as a fallback

        Args:
            open_positions: List of position objects or dicts.

        Returns:
            List of symbol strings in the same order as open_positions.
        """
        symbols: List[str] = []
        for pos in open_positions:
            if hasattr(pos, "symbol"):
                symbols.append(str(pos.symbol))
            elif isinstance(pos, dict) and "symbol" in pos:
                symbols.append(str(pos["symbol"]))
            else:
                symbols.append(str(pos))
        return symbols
