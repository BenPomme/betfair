"""
RegimeSelector: compare HMM vs Transformer on validation data and select best model.

Trains both models on the provided DataFrame, computes accuracy and stability
metrics, selects the winner, and persists the comparison to JSON for auditing.

Regime ordering (shared by both models):
    0 — low vol
    1 — medium
    2 — high
    3 — crisis

Usage:
    from funding.ml.regime_selector import RegimeSelector
    from funding.ml.regime_features import build_regime_features, assign_regime_labels

    df = build_regime_features(["BTCUSDT", "ETHUSDT"])
    df["regime_label"] = assign_regime_labels(df)

    selector = RegimeSelector()
    comparison = selector.compare(df)
    best = selector.select_best()
    model = selector.get_model()
    selector.save_comparison()
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

import config
from funding.ml.regime_hmm import RegimeHMM
from funding.ml.regime_transformer import RegimeTransformer
from funding.ml.regime_features import (
    get_regime_feature_columns,
    assign_regime_labels,
)

logger = logging.getLogger(__name__)

_COMPARISON_FILENAME = "regime_comparison.json"


class RegimeSelector:
    """
    Compare RegimeHMM and RegimeTransformer and select the best model.

    Selection criterion: highest validation accuracy.
    Tiebreak: highest stability (longer average same-regime runs are better).

    Attributes:
        _model_dir    : Path to directory for model artifacts and comparison JSON.
        _hmm          : RegimeHMM instance (populated after compare()).
        _transformer  : RegimeTransformer instance (populated after compare()).
        _comparison   : Dict with per-model metrics from the last compare() call.
        _selected     : "hmm" or "transformer" (set after select_best()).
    """

    def __init__(self, model_dir: Optional[str] = None) -> None:
        self._model_dir = Path(model_dir or config.REGIME_MODEL_PATH)
        self._hmm: Optional[RegimeHMM] = None
        self._transformer: Optional[RegimeTransformer] = None
        self._comparison: Dict = {}
        self._selected: Optional[str] = None  # "hmm" or "transformer"

    # ------------------------------------------------------------------
    # Core comparison logic
    # ------------------------------------------------------------------

    def compare(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
    ) -> Dict:
        """
        Train both HMM and Transformer on *df*, compute comparison metrics.

        Metrics computed per model:
            accuracy     : Val accuracy (Transformer) or held-out regime match rate (HMM).
            stability    : Average duration of consecutive same-regime runs (in 8h periods).
            crisis_corr  : Pearson correlation between crisis-regime flag and realized_vol_24h
                           drawdown proxy (higher = better crisis detection).
            metrics      : Raw training metrics dict from the model's fit() call.

        The DataFrame must contain a "regime_label" column (from assign_regime_labels())
        for Transformer training and for accuracy evaluation of the HMM.

        Args:
            df             : Feature DataFrame from build_regime_features().
                             Must have a "regime_label" column.
            feature_columns: Optional explicit feature column list. If None,
                             get_regime_feature_columns(df) is called.

        Returns:
            Comparison dict with keys "hmm" and "transformer", each containing
            accuracy, stability, crisis_corr, and model-specific metrics.
        """
        if "regime_label" not in df.columns:
            logger.info(
                "compare(): 'regime_label' column missing; assigning labels from realized_vol_24h"
            )
            df = df.copy()
            df["regime_label"] = assign_regime_labels(df)

        cols = feature_columns or get_regime_feature_columns(df)

        # Temporal train/val split — last 20% held out for evaluation
        clean = df[cols + ["regime_label"]].dropna()
        n = len(clean)
        n_val = max(1, int(n * 0.2))
        n_train = n - n_val
        train_df = clean.iloc[:n_train]
        val_df = clean.iloc[n_train:]

        logger.info(
            "RegimeSelector.compare(): train=%d, val=%d rows, features=%d",
            n_train, n_val, len(cols),
        )

        self._comparison = {}

        # --- HMM ---
        self._comparison["hmm"] = self._evaluate_hmm(
            train_df, val_df, cols
        )

        # --- Transformer ---
        self._comparison["transformer"] = self._evaluate_transformer(
            clean, val_df, cols
        )

        logger.info(
            "Comparison: HMM accuracy=%.4f stability=%.2f | "
            "Transformer accuracy=%.4f stability=%.2f",
            self._comparison["hmm"]["accuracy"],
            self._comparison["hmm"]["stability"],
            self._comparison["transformer"]["accuracy"],
            self._comparison["transformer"]["stability"],
        )
        return dict(self._comparison)

    def _evaluate_hmm(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_columns: List[str],
    ) -> Dict:
        """
        Fit RegimeHMM on train_df, evaluate on val_df.

        Returns dict with accuracy, stability, crisis_corr, metrics.
        """
        n_states = config.REGIME_N_STATES
        try:
            self._hmm = RegimeHMM(
                n_states=n_states,
                model_dir=str(self._model_dir),
            )
            fit_metrics = self._hmm.fit(train_df, feature_columns=feature_columns)

            # Predict on val set
            val_features = val_df[feature_columns]
            val_labels = val_df["regime_label"].values.astype(int)

            # Predict row by row isn't feasible for HMM — predict all val at once
            # (HMM uses sequence — predict the whole val block, compare last labels)
            hmm_raw = self._hmm._model.predict(
                self._hmm._prepare_X(val_features)
            )
            predicted = np.array(
                [self._hmm._regime_map.get(int(s), int(s)) for s in hmm_raw]
            )

            accuracy = float((predicted == val_labels).mean()) if len(val_labels) > 0 else 0.0
            stability = float(self._hmm._compute_stability(predicted))
            crisis_corr = _compute_crisis_drawdown_corr(predicted, val_df)

            return {
                "accuracy": accuracy,
                "stability": stability,
                "crisis_corr": crisis_corr,
                "metrics": fit_metrics,
            }
        except Exception as exc:
            logger.warning("HMM evaluation failed: %s", exc, exc_info=True)
            self._hmm = None
            return {
                "accuracy": 0.0,
                "stability": 0.0,
                "crisis_corr": 0.0,
                "metrics": {},
                "error": str(exc),
            }

    def _evaluate_transformer(
        self,
        full_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_columns: List[str],
    ) -> Dict:
        """
        Fit RegimeTransformer on full_df (uses its own internal 80/20 split),
        then evaluate on val_df.

        Returns dict with accuracy, stability, crisis_corr, metrics.
        """
        n_states = config.REGIME_N_STATES
        try:
            self._transformer = RegimeTransformer(
                n_states=n_states,
                model_dir=str(self._model_dir),
            )
            fit_metrics = self._transformer.fit(
                full_df, feature_columns=feature_columns
            )

            # Accuracy on val_df: need at least window_size rows for prediction
            window = self._transformer._window_size
            if len(val_df) < window:
                logger.warning(
                    "val_df has only %d rows; Transformer requires window_size=%d. "
                    "Using fit val_accuracy as proxy.",
                    len(val_df),
                    window,
                )
                accuracy = fit_metrics.get("val_accuracy", 0.0)
                predicted = np.array([])
            else:
                # Slide a window over val_df, predict at each step
                val_features = val_df[feature_columns]
                val_labels = val_df["regime_label"].values.astype(int)
                predicted_list = []
                label_list = []
                for i in range(window - 1, len(val_df)):
                    window_slice = val_features.iloc[i - window + 1 : i + 1]
                    pred = self._transformer.predict_regime(window_slice)
                    predicted_list.append(pred)
                    label_list.append(val_labels[i])

                predicted = np.array(predicted_list, dtype=int)
                labels_arr = np.array(label_list, dtype=int)
                accuracy = float((predicted == labels_arr).mean()) if len(labels_arr) > 0 else 0.0

            stability = _compute_stability(predicted) if len(predicted) > 0 else 0.0
            crisis_corr = _compute_crisis_drawdown_corr(predicted, val_df.iloc[window - 1:])

            return {
                "accuracy": accuracy,
                "stability": stability,
                "crisis_corr": crisis_corr,
                "metrics": fit_metrics,
            }
        except Exception as exc:
            logger.warning("Transformer evaluation failed: %s", exc, exc_info=True)
            self._transformer = None
            return {
                "accuracy": 0.0,
                "stability": 0.0,
                "crisis_corr": 0.0,
                "metrics": {},
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select_best(self) -> str:
        """
        Select the best model by accuracy; break ties by stability.

        Must be called after compare().

        Returns:
            "hmm" or "transformer" (also stored in self._selected).

        Raises:
            RuntimeError: If compare() has not been called yet.
        """
        if not self._comparison:
            raise RuntimeError(
                "select_best() called before compare(). "
                "Call compare(df) first."
            )

        hmm_acc = self._comparison.get("hmm", {}).get("accuracy", 0.0)
        tf_acc = self._comparison.get("transformer", {}).get("accuracy", 0.0)
        hmm_stab = self._comparison.get("hmm", {}).get("stability", 0.0)
        tf_stab = self._comparison.get("transformer", {}).get("stability", 0.0)

        if hmm_acc > tf_acc:
            self._selected = "hmm"
        elif tf_acc > hmm_acc:
            self._selected = "transformer"
        else:
            # Tiebreak: higher stability wins
            self._selected = "hmm" if hmm_stab >= tf_stab else "transformer"

        logger.info(
            "RegimeSelector: selected '%s' "
            "(HMM acc=%.4f stab=%.2f | Transformer acc=%.4f stab=%.2f)",
            self._selected,
            hmm_acc, hmm_stab,
            tf_acc, tf_stab,
        )
        return self._selected

    def get_model(self) -> Union[RegimeHMM, RegimeTransformer]:
        """
        Return the selected model instance.

        Must be called after select_best().

        Returns:
            RegimeHMM or RegimeTransformer instance.

        Raises:
            RuntimeError: If select_best() has not been called or the selected
                          model instance is unavailable.
        """
        if self._selected is None:
            raise RuntimeError(
                "get_model() called before select_best(). "
                "Call compare(df) then select_best() first."
            )
        if self._selected == "hmm":
            if self._hmm is None:
                raise RuntimeError(
                    "HMM was selected but the model instance is unavailable "
                    "(it may have failed during evaluation)."
                )
            return self._hmm
        else:
            if self._transformer is None:
                raise RuntimeError(
                    "Transformer was selected but the model instance is unavailable "
                    "(it may have failed during evaluation)."
                )
            return self._transformer

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_comparison(self) -> None:
        """
        Persist the comparison dict to JSON at model_dir/regime_comparison.json.

        Raises:
            RuntimeError: If compare() has not been called yet.
        """
        if not self._comparison:
            raise RuntimeError(
                "save_comparison() called before compare(). "
                "Call compare(df) first."
            )
        self._model_dir.mkdir(parents=True, exist_ok=True)
        path = self._model_dir / _COMPARISON_FILENAME
        payload = {
            "selected": self._selected,
            "comparison": self._comparison,
        }
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        logger.info("RegimeSelector comparison saved to %s", path)

    def load_comparison(self) -> Dict:
        """
        Load the comparison dict from JSON at model_dir/regime_comparison.json.

        Returns:
            The loaded comparison dict (also stored in self._comparison).

        Raises:
            FileNotFoundError: If the comparison file does not exist.
        """
        path = self._model_dir / _COMPARISON_FILENAME
        if not path.exists():
            raise FileNotFoundError(
                f"Regime comparison file not found: {path}. "
                "Call compare() and save_comparison() first."
            )
        raw = json.loads(path.read_text(encoding="utf-8"))
        self._comparison = raw.get("comparison", {})
        self._selected = raw.get("selected")
        logger.info(
            "RegimeSelector comparison loaded from %s (selected=%s)",
            path,
            self._selected,
        )
        return dict(self._comparison)


# ------------------------------------------------------------------
# Module-level helpers (not exported as class methods)
# ------------------------------------------------------------------

def _compute_stability(predictions: np.ndarray) -> float:
    """
    Compute stability as the mean length of consecutive same-regime runs.

    Args:
        predictions: 1-D integer array of regime predictions.

    Returns:
        Mean run length (float). Returns 0.0 for empty input.
    """
    if len(predictions) == 0:
        return 0.0
    runs = []
    current = 1
    for i in range(1, len(predictions)):
        if predictions[i] == predictions[i - 1]:
            current += 1
        else:
            runs.append(current)
            current = 1
    runs.append(current)
    return float(np.mean(runs))


def _compute_crisis_drawdown_corr(
    predictions: np.ndarray,
    val_df: pd.DataFrame,
) -> float:
    """
    Compute Pearson correlation between the crisis-regime indicator (regime==3)
    and realized_vol_24h as a drawdown proxy.

    Higher correlation indicates the model correctly flags high-vol crisis periods.

    Args:
        predictions: 1-D int array of regime labels aligned to val_df rows.
        val_df:      Validation DataFrame containing realized_vol_24h (optional).

    Returns:
        Pearson correlation coefficient (float in [-1, 1]), or 0.0 if unavailable.
    """
    if len(predictions) == 0 or "realized_vol_24h" not in val_df.columns:
        return 0.0
    try:
        vol = val_df["realized_vol_24h"].values
        # Align lengths (prediction window may differ from val_df length)
        min_len = min(len(predictions), len(vol))
        if min_len < 2:
            return 0.0
        crisis_flag = (predictions[:min_len] == 3).astype(float)
        vol_slice = vol[:min_len].astype(float)
        if np.std(crisis_flag) == 0 or np.std(vol_slice) == 0:
            return 0.0
        corr = float(np.corrcoef(crisis_flag, vol_slice)[0, 1])
        return corr if np.isfinite(corr) else 0.0
    except Exception as exc:
        logger.debug("crisis_drawdown_corr computation failed: %s", exc)
        return 0.0
