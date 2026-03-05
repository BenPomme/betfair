"""
Model comparison framework for contrarian prediction models.

Trains both ContrarianXGBoost and TFTPredictor on the same walk-forward
data split and selects the best performer based on Sharpe ratio (primary)
and AUC (tiebreak).

Usage:
    from funding.ml.model_selector import ModelSelector
    from funding.ml.contrarian_features import build_contrarian_features_all

    df = build_contrarian_features_all(["ETHUSDT", "BTCUSDT"])
    selector = ModelSelector()
    comparison = selector.compare(df, n_trials=30, tft_epochs=50)
    best = selector.select_best()         # "xgboost" or "tft"
    model = selector.get_model()          # ready-to-use model instance
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from funding.ml.tft_predictor import TFTPredictor, _DEEP_LEARNING_AVAILABLE

logger = logging.getLogger(__name__)

# Optional: XGBoost can fail to load on macOS if libomp is missing (XGBoostError)
ContrarianXGBoost: Optional[type] = None
try:
    from funding.ml.contrarian_xgb import ContrarianXGBoost as _ContrarianXGBoost
    ContrarianXGBoost = _ContrarianXGBoost
except ImportError:
    pass
except Exception as e:
    if type(e).__name__ != "XGBoostError":
        raise
    logger.warning("ContrarianXGBoost unavailable (e.g. XGBoost/libomp): %s", e)

# Minimum AUC / Sharpe values used when a model fails to train
_FALLBACK_SHARPE = -999.0
_FALLBACK_AUC = 0.0

# Key used when storing the comparison result on disk
_COMPARISON_FILENAME = "contrarian_comparison.json"


def _compute_sharpe_from_metrics(metrics: Dict[str, Any]) -> float:
    """Extract sharpe_simulated from a metrics dict, falling back to 0.0."""
    val = metrics.get("sharpe_simulated", _FALLBACK_SHARPE)
    try:
        return float(val)
    except (TypeError, ValueError):
        return _FALLBACK_SHARPE


def _compute_auc_from_metrics(metrics: Dict[str, Any]) -> float:
    """Extract AUC from a metrics dict, falling back to 0.0.

    ContrarianXGBoost uses the key 'auc'.
    TFTPredictor uses 'direction_accuracy' as its closest analogue — we map
    that to a [0, 1] value and use it as a proxy AUC when the real AUC is
    absent.
    """
    if "auc" in metrics:
        try:
            return float(metrics["auc"])
        except (TypeError, ValueError):
            pass
    # TFT fallback: direction_accuracy is already in [0, 1]
    if "direction_accuracy" in metrics:
        try:
            return float(metrics["direction_accuracy"])
        except (TypeError, ValueError):
            pass
    return _FALLBACK_AUC


def _compute_profit_factor(pnl_array: np.ndarray) -> float:
    """Compute profit factor (sum of gains / |sum of losses|) from a PnL array.

    Returns 0.0 when there are no losses, or np.inf when losses are exactly 0.
    """
    if len(pnl_array) == 0:
        return 0.0
    gains = float(pnl_array[pnl_array > 0].sum())
    losses = float(np.abs(pnl_array[pnl_array < 0].sum()))
    if losses == 0.0:
        return np.inf if gains > 0 else 0.0
    return gains / losses


def _simulate_pnl(
    model: Any,
    X_test: pd.DataFrame,
    y_ret_test: np.ndarray,
) -> np.ndarray:
    """Run model.predict on the test features and compute per-period PnL.

    Signal convention:
        predicted_direction = True  → long (earn actual return)
        predicted_direction = False → short (earn negative actual return)

    Returns:
        1-D numpy array of PnL values, one per test row.
        Returns an empty array if prediction fails.
    """
    try:
        preds = model.predict(X_test)
        direction = preds["predicted_direction"].values.astype(bool)
        signal = np.where(direction, 1.0, -1.0)
        return signal * y_ret_test
    except Exception as exc:
        logger.warning("PnL simulation failed: %s", exc)
        return np.array([], dtype=float)


def _annualised_sharpe(pnl: np.ndarray) -> float:
    """Annualised Sharpe from an 8h-period PnL array (1095 periods/year)."""
    if len(pnl) < 2:
        return 0.0
    std = float(np.std(pnl, ddof=1))
    if std == 0.0:
        return 0.0
    periods_per_year = 365 * 3
    return float((np.mean(pnl) / std) * np.sqrt(periods_per_year))


class ModelSelector:
    """Compare ContrarianXGBoost and TFTPredictor; select the best performer.

    Attributes (read via properties):
        selected_model -- "xgboost", "tft", or None (before select_best() or when both unavailable)
    """

    def __init__(self, model_dir: Optional[str] = None) -> None:
        self._model_dir = Path(model_dir or "data/funding_models")
        self._comparison: Dict[str, Any] = {}   # stored comparison results
        self._selected_model: Optional[str] = None  # "xgboost", "tft", or None
        self._xgb: Optional[Any] = None
        self._tft: Optional[TFTPredictor] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compare(
        self,
        df: pd.DataFrame,
        n_trials: int = 30,
        tft_epochs: int = 50,
    ) -> Dict[str, Any]:
        """Train both models on the same data split and compare performance.

        Both models receive the identical DataFrame produced by
        build_contrarian_features_all.  They each perform their own
        internal walk-forward split (last 2 months held out), so the
        comparison is on an equivalent held-out window.

        After training, the comparison dict is written to:
            <model_dir>/contrarian_comparison.json

        Args:
            df:         DataFrame from build_contrarian_features_all.
            n_trials:   Optuna trials for ContrarianXGBoost hyperparameter
                        search (passed through to ContrarianXGBoost.train).
            tft_epochs: Maximum training epochs for TFTPredictor.

        Returns:
            Comparison dict with structure::

                {
                    "xgboost": {
                        "trained": True,
                        "direction_accuracy": float,
                        "auc": float,
                        "sharpe_simulated": float,
                        "profit_factor": float,
                        "metrics": { ... raw metrics from ContrarianXGBoost ... },
                        "error": None,
                    },
                    "tft": {
                        "trained": bool,
                        "direction_accuracy": float,
                        "auc": float,          # proxy via direction_accuracy
                        "sharpe_simulated": float,
                        "profit_factor": float,
                        "metrics": { ... raw metrics from TFTPredictor ... },
                        "error": str | None,   # e.g. "torch not installed"
                    },
                    "selected": "xgboost" | "tft" | None,
                }
        """
        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty.")

        logger.info(
            "ModelSelector.compare — rows: %d, columns: %d, n_trials: %d, "
            "tft_epochs: %d",
            len(df),
            len(df.columns),
            n_trials,
            tft_epochs,
        )

        if ContrarianXGBoost is not None:
            xgb_result = self._train_xgboost(df, n_trials)
        else:
            xgb_result = {
                "trained": False,
                "direction_accuracy": _FALLBACK_AUC,
                "auc": _FALLBACK_AUC,
                "sharpe_simulated": _FALLBACK_SHARPE,
                "profit_factor": 0.0,
                "metrics": {},
                "error": "XGBoost not available (e.g. libomp missing on macOS).",
            }

        if _DEEP_LEARNING_AVAILABLE:
            tft_result = self._train_tft(df, tft_epochs)
        else:
            tft_result = {
                "trained": False,
                "direction_accuracy": _FALLBACK_AUC,
                "auc": _FALLBACK_AUC,
                "sharpe_simulated": _FALLBACK_SHARPE,
                "profit_factor": 0.0,
                "metrics": {},
                "error": (
                    "torch / pytorch_lightning / pytorch_forecasting not installed — "
                    "skipping TFT training."
                ),
            }

        self._comparison = {
            "xgboost": xgb_result,
            "tft": tft_result,
        }

        # Run select_best to populate _selected_model and annotate comparison
        selected = self.select_best()
        self._comparison["selected"] = selected

        # Persist to disk
        self._save_comparison()

        logger.info(
            "Model comparison complete — xgb_sharpe: %.3f, tft_sharpe: %.3f, "
            "xgb_auc: %.3f, tft_auc: %.3f, selected: %s",
            xgb_result.get("sharpe_simulated", _FALLBACK_SHARPE),
            tft_result.get("sharpe_simulated", _FALLBACK_SHARPE),
            xgb_result.get("auc", _FALLBACK_AUC),
            tft_result.get("auc", _FALLBACK_AUC),
            selected,
        )

        return self._comparison

    def select_best(self) -> Optional[str]:
        """Determine the best model based on Sharpe (primary) and AUC (tiebreak).

        Selection rules:
        1. If only one model trained successfully, select it.
        2. Among successfully trained models, choose highest Sharpe.
        3. Tiebreak (|diff| < 0.01): choose highest AUC.
        4. If neither model trained, return None.
        5. When selection would be "xgboost" but ContrarianXGBoost is None,
           fall back to "tft" if TFT trained, else None.

        Returns:
            "xgboost", "tft", or None

        Side effects:
            Sets self._selected_model.
        """
        xgb_entry = self._comparison.get("xgboost", {})
        tft_entry = self._comparison.get("tft", {})

        xgb_trained = xgb_entry.get("trained", False)
        tft_trained = tft_entry.get("trained", False)

        if not xgb_trained and not tft_trained:
            logger.warning(
                "Neither model trained successfully. No model selected."
            )
            self._selected_model = None
            return None

        if xgb_trained and not tft_trained:
            if ContrarianXGBoost is None:
                self._selected_model = None
                return None
            logger.info("TFT unavailable — selecting 'xgboost' by default.")
            self._selected_model = "xgboost"
            return self._selected_model

        if tft_trained and not xgb_trained:
            logger.info("XGBoost failed — selecting 'tft' by default.")
            self._selected_model = "tft"
            return self._selected_model

        # Both trained — compare
        xgb_sharpe = xgb_entry.get("sharpe_simulated", _FALLBACK_SHARPE)
        tft_sharpe = tft_entry.get("sharpe_simulated", _FALLBACK_SHARPE)

        sharpe_diff = abs(xgb_sharpe - tft_sharpe)

        if sharpe_diff < 0.01:
            # Tiebreak by AUC
            xgb_auc = xgb_entry.get("auc", _FALLBACK_AUC)
            tft_auc = tft_entry.get("auc", _FALLBACK_AUC)
            if tft_auc > xgb_auc:
                logger.info(
                    "Sharpe tied (diff=%.4f); TFT wins on AUC (%.3f vs %.3f).",
                    sharpe_diff,
                    tft_auc,
                    xgb_auc,
                )
                self._selected_model = "tft"
            else:
                if ContrarianXGBoost is None:
                    self._selected_model = "tft"
                    logger.info(
                        "Sharpe tied (diff=%.4f); XGBoost would win on AUC but "
                        "unavailable — falling back to TFT (AUC %.3f).",
                        sharpe_diff,
                        tft_auc,
                    )
                else:
                    self._selected_model = "xgboost"
                    logger.info(
                        "Sharpe tied (diff=%.4f); XGBoost wins on AUC (%.3f vs %.3f).",
                        sharpe_diff,
                        xgb_auc,
                        tft_auc,
                    )
        elif xgb_sharpe >= tft_sharpe:
            if ContrarianXGBoost is None:
                self._selected_model = "tft"
                logger.info(
                    "XGBoost would win but unavailable — falling back to TFT "
                    "(Sharpe: %.3f).",
                    tft_sharpe,
                )
            else:
                logger.info(
                    "XGBoost selected — Sharpe: %.3f > TFT Sharpe: %.3f",
                    xgb_sharpe,
                    tft_sharpe,
                )
                self._selected_model = "xgboost"
        else:
            logger.info(
                "TFT selected — Sharpe: %.3f > XGBoost Sharpe: %.3f",
                tft_sharpe,
                xgb_sharpe,
            )
            self._selected_model = "tft"

        return self._selected_model

    def get_model(self):
        """Return the selected model instance.

        If no explicit selection has been made (or selection is "auto"),
        calls select_best() first.

        When the selected model is "xgboost" but ContrarianXGBoost is
        unavailable (e.g. libomp missing), falls back to TFT if available,
        else returns None.

        Returns:
            ContrarianXGBoost, TFTPredictor, or None if no model is available.

        Raises:
            RuntimeError: If no comparison has been run yet (compare() not called).
        """
        if not self._comparison:
            raise RuntimeError(
                "No comparison data available. Call compare() first."
            )

        if self._selected_model is None or self._selected_model == "auto":
            self.select_best()

        if self._selected_model is None:
            return None

        if self._selected_model == "tft":
            if self._tft is None:
                raise RuntimeError(
                    "TFT model selected but instance is None. "
                    "Ensure compare() completed successfully for TFT."
                )
            return self._tft

        # Selected is "xgboost"
        if ContrarianXGBoost is None:
            if self._tft is not None:
                return self._tft
            return None
        if self._xgb is None:
            raise RuntimeError(
                "XGBoost model selected but instance is None. "
                "Ensure compare() completed successfully for XGBoost."
            )
        return self._xgb

    def get_comparison(self) -> Dict[str, Any]:
        """Return the stored comparison results from the last compare() call.

        Returns:
            Dict with keys "xgboost", "tft", and "selected". Empty dict if
            compare() has not been called yet.
        """
        return dict(self._comparison)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def selected_model(self) -> Optional[str]:
        """The currently selected model name ("xgboost", "tft", or None)."""
        return self._selected_model

    # ------------------------------------------------------------------
    # Internal: training helpers
    # ------------------------------------------------------------------

    def _train_xgboost(
        self,
        df: pd.DataFrame,
        n_trials: int,
    ) -> Dict[str, Any]:
        """Train ContrarianXGBoost and return a standardised result dict.

        Never raises — exceptions are caught and recorded in the result so
        the TFT branch can still proceed. If ContrarianXGBoost is unavailable,
        returns a minimal untrained result (caller may skip calling this).
        """
        if ContrarianXGBoost is None:
            return {
                "trained": False,
                "direction_accuracy": _FALLBACK_AUC,
                "auc": _FALLBACK_AUC,
                "sharpe_simulated": _FALLBACK_SHARPE,
                "profit_factor": 0.0,
                "metrics": {},
                "error": "XGBoost not available (e.g. libomp missing on macOS).",
            }
        logger.info("Training ContrarianXGBoost (n_trials=%d)...", n_trials)
        result: Dict[str, Any] = {
            "trained": False,
            "direction_accuracy": _FALLBACK_AUC,
            "auc": _FALLBACK_AUC,
            "sharpe_simulated": _FALLBACK_SHARPE,
            "profit_factor": 0.0,
            "metrics": {},
            "error": None,
        }

        try:
            xgb_model = ContrarianXGBoost(model_dir=str(self._model_dir / "contrarian_xgb"))
            metrics = xgb_model.train(df, tune=True, n_trials=n_trials, test_months=2)

            # Compute profit factor from a fresh PnL simulation on the held-out slice
            pnl = self._xgb_pnl_simulation(xgb_model, df)
            profit_factor = _compute_profit_factor(pnl)

            result.update(
                {
                    "trained": True,
                    "direction_accuracy": float(metrics.get("accuracy", 0.0)),
                    "auc": _compute_auc_from_metrics(metrics),
                    "sharpe_simulated": _compute_sharpe_from_metrics(metrics),
                    "profit_factor": profit_factor,
                    "metrics": {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                                for k, v in metrics.items()},
                    "error": None,
                }
            )
            self._xgb = xgb_model
            logger.info(
                "ContrarianXGBoost trained — auc: %.3f, sharpe: %.3f, "
                "profit_factor: %.3f",
                result["auc"],
                result["sharpe_simulated"],
                result["profit_factor"],
            )

        except Exception as exc:
            logger.error("ContrarianXGBoost training failed: %s", exc, exc_info=True)
            result["error"] = str(exc)

        return result

    def _train_tft(
        self,
        df: pd.DataFrame,
        tft_epochs: int,
    ) -> Dict[str, Any]:
        """Try to train TFTPredictor and return a standardised result dict.

        Skips gracefully with a warning when pytorch / pytorch-forecasting
        are not installed. Never raises — exceptions are caught and recorded.
        """
        result: Dict[str, Any] = {
            "trained": False,
            "direction_accuracy": _FALLBACK_AUC,
            "auc": _FALLBACK_AUC,
            "sharpe_simulated": _FALLBACK_SHARPE,
            "profit_factor": 0.0,
            "metrics": {},
            "error": None,
        }

        if not _DEEP_LEARNING_AVAILABLE:
            msg = (
                "torch / pytorch_lightning / pytorch_forecasting not installed — "
                "skipping TFT training. Install with: "
                "pip install torch pytorch-lightning pytorch-forecasting"
            )
            logger.warning(msg)
            result["error"] = msg
            return result

        logger.info("Training TFTPredictor (max_epochs=%d)...", tft_epochs)
        try:
            tft_model = TFTPredictor(model_dir=str(self._model_dir / "tft"))
            metrics = tft_model.train(df, max_epochs=tft_epochs)

            # TFT does not produce an AUC or Sharpe natively; derive them
            # from direction_accuracy as a proxy AUC and simulate PnL for Sharpe.
            pnl = self._tft_pnl_simulation(tft_model, df)
            sharpe = _annualised_sharpe(pnl)
            profit_factor = _compute_profit_factor(pnl)
            proxy_auc = _compute_auc_from_metrics(metrics)

            result.update(
                {
                    "trained": True,
                    "direction_accuracy": float(metrics.get("direction_accuracy", 0.0)),
                    "auc": proxy_auc,
                    "sharpe_simulated": sharpe,
                    "profit_factor": profit_factor,
                    "metrics": {
                        k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                        for k, v in metrics.items()
                    },
                    "error": None,
                }
            )
            self._tft = tft_model
            logger.info(
                "TFTPredictor trained — auc(proxy): %.3f, sharpe: %.3f, "
                "profit_factor: %.3f",
                result["auc"],
                result["sharpe_simulated"],
                result["profit_factor"],
            )

        except Exception as exc:
            logger.error("TFTPredictor training failed: %s", exc, exc_info=True)
            result["error"] = str(exc)

        return result

    # ------------------------------------------------------------------
    # Internal: PnL simulation helpers
    # ------------------------------------------------------------------

    def _xgb_pnl_simulation(
        self,
        model: Any,
        df: pd.DataFrame,
    ) -> np.ndarray:
        """Simulate PnL on the XGBoost held-out test slice.

        Reproduces the same walk-forward cut used by ContrarianXGBoost.train
        (last 2 months), then calls model.predict on the test features.

        Returns:
            1-D float array of per-period PnL values. Empty on failure.
        """
        try:
            split_date = df.index.max() - pd.DateOffset(months=2)
            test_df = df[df.index > split_date]

            if test_df.empty or "price_return_24h_target" not in test_df.columns:
                return np.array([], dtype=float)

            y_ret = test_df["price_return_24h_target"].astype(float).values
            feature_cols = model.feature_columns
            available = [c for c in feature_cols if c in test_df.columns]
            X_test = (
                test_df[available]
                .reindex(columns=feature_cols, fill_value=0.0)
                .fillna(0)
                .replace([np.inf, -np.inf], 0)
            )
            return _simulate_pnl(model, X_test, y_ret)
        except Exception as exc:
            logger.warning("XGBoost PnL simulation failed: %s", exc)
            return np.array([], dtype=float)

    def _tft_pnl_simulation(
        self,
        model: TFTPredictor,
        df: pd.DataFrame,
    ) -> np.ndarray:
        """Simulate PnL on the TFT held-out validation slice.

        Uses the last 20% of the sorted data (mirroring TFTPredictor's
        internal _VALIDATION_FRACTION split) to keep the comparison fair.

        Returns:
            1-D float array of per-period PnL values. Empty on failure.
        """
        try:
            sorted_df = df.sort_index()
            n = len(sorted_df)
            cutoff = int(n * 0.80)
            test_df = sorted_df.iloc[cutoff:]

            if test_df.empty or "price_return_24h_target" not in test_df.columns:
                return np.array([], dtype=float)

            y_ret = test_df["price_return_24h_target"].astype(float).values
            preds = model.predict(test_df)
            if preds.empty:
                return np.array([], dtype=float)

            direction = preds["predicted_direction"].values.astype(bool) if "predicted_direction" in preds.columns else (preds["direction_prob"].values >= 0.5)
            signal = np.where(direction, 1.0, -1.0)
            n_min = min(len(signal), len(y_ret))
            return signal[:n_min] * y_ret[:n_min]

        except Exception as exc:
            logger.warning("TFT PnL simulation failed: %s", exc)
            return np.array([], dtype=float)

    # ------------------------------------------------------------------
    # Internal: persistence
    # ------------------------------------------------------------------

    def _save_comparison(self) -> None:
        """Serialise _comparison to <model_dir>/contrarian_comparison.json."""
        self._model_dir.mkdir(parents=True, exist_ok=True)
        output_path = self._model_dir / _COMPARISON_FILENAME

        # Make the dict JSON-safe (numpy scalars → Python scalars)
        safe = _make_json_safe(self._comparison)

        try:
            with open(output_path, "w") as fh:
                json.dump(safe, fh, indent=2, default=str)
            logger.info("Saved model comparison to %s", output_path)
        except Exception as exc:
            logger.warning("Could not save comparison JSON: %s", exc)

    def load_comparison(self) -> Dict[str, Any]:
        """Load a previously saved comparison from disk.

        Returns:
            The comparison dict loaded from contrarian_comparison.json,
            or an empty dict if the file does not exist.
        """
        path = self._model_dir / _COMPARISON_FILENAME
        if not path.exists():
            logger.info("No comparison file found at %s", path)
            return {}

        with open(path) as fh:
            self._comparison = json.load(fh)

        self._selected_model = self._comparison.get("selected")
        logger.info(
            "Loaded comparison from %s — selected: %s",
            path,
            self._selected_model,
        )
        return dict(self._comparison)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _make_json_safe(obj: Any) -> Any:
    """Recursively convert numpy / non-JSON types to Python builtins."""
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None  # JSON does not support NaN / Inf
    return obj
