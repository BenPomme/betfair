"""
XGBoost contrarian model for predicting price direction after extreme funding events.

Two sub-models:
  1. Classifier (XGBClassifier): predicts direction_24h (binary — will price be
     higher in 24h than now?)
  2. Regressor (XGBRegressor): predicts price_return_24h_target (forward 24h
     return magnitude)

Training uses walk-forward validation to prevent lookahead bias.
Hyperparameter tuning via Optuna (TimeSeriesSplit cross-validation).
A calibrated probability threshold is selected so the model fires at
approximately 2–4 actionable signals per week.

Save format:
  <model_dir>/<name>_clf.pkl   — serialised XGBClassifier
  <model_dir>/<name>_reg.pkl   — serialised XGBRegressor
  <model_dir>/<name>_meta.json — feature_columns, metrics, threshold

Usage:
    from funding.ml.contrarian_xgb import ContrarianXGBoost
    from funding.ml.contrarian_features import build_contrarian_features

    df = build_contrarian_features("ETHUSDT")
    model = ContrarianXGBoost()
    metrics = model.train(df, tune=True)
    model.save()

    preds = model.predict(features_df)
    # preds columns: direction_prob, predicted_return_24h, confidence,
    #                predicted_direction
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

from funding.ml.contrarian_features import get_contrarian_feature_columns

logger = logging.getLogger(__name__)

# Suppress Optuna's per-trial INFO spam
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Minimum rows in training split before we attempt to train at all
_MIN_TRAIN_ROWS = 50
# Minimum rows in the test split to compute meaningful metrics
_MIN_TEST_ROWS = 10
# Number of 8h periods per week (3 per day × 7 days)
_PERIODS_PER_WEEK = 21


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class ContrarianXGBoost:
    """XGBoost contrarian model: predicts price direction after extreme funding.

    Attributes (read via properties):
        metrics         -- dict of evaluation metrics from the last train() call
        feature_columns -- list of feature column names used during training
        threshold       -- calibrated probability threshold for direction signal
    """

    def __init__(self, model_dir: Optional[str] = None):
        self._model_dir = Path(model_dir or "data/funding_models/contrarian_xgb")
        self._clf: Optional[xgb.XGBClassifier] = None
        self._reg: Optional[xgb.XGBRegressor] = None
        self._feature_columns: List[str] = []
        self._metrics: Dict[str, Any] = {}
        self._threshold: float = 0.5  # calibrated probability threshold

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        df: pd.DataFrame,
        tune: bool = True,
        n_trials: int = 30,
        test_months: int = 2,
    ) -> Dict[str, Any]:
        """Train classifier + regressor on contrarian features.

        Performs a single walk-forward split: all data except the last
        `test_months` months is used for training; the held-out tail is
        used for evaluation and threshold calibration.

        If `tune=True`, Optuna searches the XGBoost hyperparameter space
        using 3-fold TimeSeriesSplit cross-validation *within the training
        portion only* (no test leakage).

        Args:
            df:          Output of build_contrarian_features / build_contrarian_features_all.
                         Must contain columns returned by get_contrarian_feature_columns
                         plus targets 'direction_24h' and 'price_return_24h_target'.
            tune:        Whether to run Optuna hyperparameter search.
            n_trials:    Number of Optuna trials per model.
            test_months: Months held out for evaluation / threshold calibration.

        Returns:
            Dict with keys: accuracy, auc, precision, recall, f1,
            sharpe_simulated, threshold, n_trades_per_week, test_size,
            feature_count, top_features_classifier, top_features_regressor.

        Raises:
            ValueError: If df is empty, lacks required columns, or has
                        insufficient rows.
        """
        df = self._sanitize_training_frame(df)
        self._validate_input(df)

        self._feature_columns = get_contrarian_feature_columns(df)
        if not self._feature_columns:
            raise ValueError(
                "get_contrarian_feature_columns returned no columns. "
                "Check that the DataFrame was built with build_contrarian_features."
            )

        # Sanitise feature matrix
        X = (
            df[self._feature_columns]
            .copy()
            .fillna(0)
            .replace([np.inf, -np.inf], 0)
        )
        y_dir = df["direction_24h"].astype(int).values
        y_ret = df["price_return_24h_target"].astype(float).values

        # Walk-forward split on time
        X_train, X_test, y_dir_train, y_dir_test, y_ret_train, y_ret_test = (
            self._walk_forward_split(df, X, y_dir, y_ret, test_months)
        )

        logger.info(
            "Walk-forward split — train: %d rows, test: %d rows, features: %d",
            len(X_train), len(X_test), len(self._feature_columns),
        )

        # Class balance weight for the classifier
        n_pos = int(y_dir_train.sum())
        n_neg = len(y_dir_train) - n_pos
        scale_pos_weight = float(n_neg) / float(n_pos) if n_pos > 0 else 1.0
        logger.info(
            "Class balance — neg: %d, pos: %d, scale_pos_weight: %.3f",
            n_neg, n_pos, scale_pos_weight,
        )

        # Hyperparameter search or defaults
        if tune:
            clf_params = self._tune_classifier(
                X_train, y_dir_train, n_trials, scale_pos_weight
            )
            reg_params = self._tune_regressor(X_train, y_ret_train, n_trials)
        else:
            clf_params = self._default_params()
            reg_params = self._default_params()

        # Train classifier
        self._clf = xgb.XGBClassifier(
            eval_metric="auc",
            use_label_encoder=False,
            verbosity=0,
            scale_pos_weight=scale_pos_weight,
            **clf_params,
        )
        self._clf.fit(X_train, y_dir_train)

        # Train regressor
        self._reg = xgb.XGBRegressor(
            eval_metric="mae",
            verbosity=0,
            **reg_params,
        )
        self._reg.fit(X_train, y_ret_train)

        # Evaluate on test set and calibrate threshold
        self._metrics = self._evaluate(
            X_test, y_dir_test, y_ret_test, len(X_test)
        )
        logger.info(
            "Metrics — acc: %.3f, auc: %.3f, precision: %.3f, recall: %.3f, "
            "f1: %.3f, sharpe: %.3f, threshold: %.3f, trades/wk: %.1f",
            self._metrics["accuracy"],
            self._metrics["auc"],
            self._metrics["precision"],
            self._metrics["recall"],
            self._metrics["f1"],
            self._metrics["sharpe_simulated"],
            self._metrics["threshold"],
            self._metrics["n_trades_per_week"],
        )
        return self._metrics

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Predict price direction for new observations.

        Args:
            features: DataFrame with the same feature columns used during
                      training (extra columns are silently ignored; missing
                      feature columns are filled with 0).

        Returns:
            DataFrame (same index as `features`) with columns:
              direction_prob      -- P(price higher in 24h)
              predicted_return_24h -- regressor's predicted 24h return
              confidence          -- |prob - 0.5| * 2  (0 = uncertain, 1 = certain)
              predicted_direction -- bool; True when prob >= self._threshold

        Raises:
            RuntimeError: If models have not been trained or loaded.
        """
        if self._clf is None or self._reg is None:
            raise RuntimeError(
                "Models not initialised. Call train() or load() first."
            )
        if features.empty:
            return pd.DataFrame(
                columns=[
                    "direction_prob",
                    "predicted_return_24h",
                    "confidence",
                    "predicted_direction",
                ],
                index=features.index,
            )

        X = self._prepare_predict_features(features)

        dir_probs: np.ndarray = self._clf.predict_proba(X)[:, 1]
        ret_preds: np.ndarray = self._reg.predict(X)

        result = pd.DataFrame(index=features.index)
        result["direction_prob"] = dir_probs
        result["predicted_return_24h"] = ret_preds
        result["confidence"] = np.abs(dir_probs - 0.5) * 2.0
        result["predicted_direction"] = dir_probs >= self._threshold
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, name: str = "contrarian_xgb") -> None:
        """Persist both models and metadata to self._model_dir.

        Files written:
          <name>_clf.pkl   — XGBClassifier (pickle)
          <name>_reg.pkl   — XGBRegressor (pickle)
          <name>_meta.json — feature_columns, metrics, threshold

        Args:
            name: Base filename stem (no extension).

        Raises:
            RuntimeError: If models have not been trained.
        """
        if self._clf is None or self._reg is None:
            raise RuntimeError(
                "Nothing to save — models have not been trained. "
                "Call train() first."
            )

        self._model_dir.mkdir(parents=True, exist_ok=True)

        clf_path = self._model_dir / f"{name}_clf.pkl"
        reg_path = self._model_dir / f"{name}_reg.pkl"
        meta_path = self._model_dir / f"{name}_meta.json"

        with open(clf_path, "wb") as fh:
            pickle.dump(self._clf, fh)
        with open(reg_path, "wb") as fh:
            pickle.dump(self._reg, fh)

        meta = {
            "name": name,
            "feature_columns": self._feature_columns,
            "threshold": self._threshold,
            "metrics": self._metrics,
        }
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2, default=str)

        logger.info(
            "Saved ContrarianXGBoost '%s' to %s (%d features, threshold=%.3f)",
            name,
            self._model_dir,
            len(self._feature_columns),
            self._threshold,
        )

    def load(self, name: str = "contrarian_xgb") -> None:
        """Load both models and metadata from self._model_dir.

        Args:
            name: Base filename stem (no extension).

        Raises:
            FileNotFoundError: If any expected file is missing.
        """
        clf_path = self._model_dir / f"{name}_clf.pkl"
        reg_path = self._model_dir / f"{name}_reg.pkl"
        meta_path = self._model_dir / f"{name}_meta.json"

        for p in (clf_path, reg_path, meta_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"Model file not found: {p}. "
                    "Run train() and save() before calling load()."
                )

        with open(clf_path, "rb") as fh:
            self._clf = pickle.load(fh)
        with open(reg_path, "rb") as fh:
            self._reg = pickle.load(fh)

        with open(meta_path) as fh:
            meta = json.load(fh)

        self._feature_columns = meta["feature_columns"]
        self._threshold = float(meta.get("threshold", 0.5))
        self._metrics = meta.get("metrics", {})

        logger.info(
            "Loaded ContrarianXGBoost '%s' from %s (%d features, threshold=%.3f)",
            name,
            self._model_dir,
            len(self._feature_columns),
            self._threshold,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def metrics(self) -> Dict[str, Any]:
        """Evaluation metrics from the last train() call."""
        return dict(self._metrics)

    @property
    def feature_columns(self) -> List[str]:
        """Feature column names used during training."""
        return list(self._feature_columns)

    @property
    def threshold(self) -> float:
        """Calibrated probability threshold for direction signal."""
        return self._threshold

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_input(self, df: pd.DataFrame) -> None:
        """Raise ValueError if df is unsuitable for training."""
        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty.")

        required_targets = {"direction_24h", "price_return_24h_target"}
        missing = required_targets - set(df.columns)
        if missing:
            raise ValueError(
                f"Input DataFrame missing target columns: {missing}. "
                "Ensure df was built with build_contrarian_features."
            )

        if len(df) < _MIN_TRAIN_ROWS + _MIN_TEST_ROWS:
            raise ValueError(
                f"Input DataFrame has only {len(df)} rows; need at least "
                f"{_MIN_TRAIN_ROWS + _MIN_TEST_ROWS} for a meaningful train/test split."
            )

        # Check that the index is a DatetimeIndex (required for time-based split)
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                "Input DataFrame index must be a DatetimeIndex. "
                "Ensure df was built with build_contrarian_features (indexed by funding_time_dt)."
            )

        if not np.isfinite(df["price_return_24h_target"].astype(float).values).all():
            raise ValueError("price_return_24h_target contains non-finite values after sanitization.")

        dir_values = set(pd.to_numeric(df["direction_24h"], errors="coerce").dropna().astype(int).tolist())
        if not dir_values.issubset({0, 1}):
            raise ValueError("direction_24h contains values outside {0,1} after sanitization.")

    def _sanitize_training_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows with non-finite targets before model fitting."""
        if df is None:
            return pd.DataFrame()
        clean = df.copy()
        clean["price_return_24h_target"] = pd.to_numeric(
            clean["price_return_24h_target"], errors="coerce"
        )
        clean["direction_24h"] = pd.to_numeric(clean["direction_24h"], errors="coerce")
        clean = clean.replace([np.inf, -np.inf], np.nan)
        clean = clean.dropna(subset=["price_return_24h_target", "direction_24h"])
        clean = clean[np.isfinite(clean["price_return_24h_target"])]
        clean = clean[clean["direction_24h"].isin([0, 1])]
        return clean

    def _walk_forward_split(
        self,
        df: pd.DataFrame,
        X: pd.DataFrame,
        y_dir: np.ndarray,
        y_ret: np.ndarray,
        test_months: int,
    ):
        """Split data into train/test using a time-based walk-forward cut.

        The split point is (max_date - test_months). If the resulting test
        set has fewer than _MIN_TEST_ROWS, the split is adjusted to guarantee
        at least _MIN_TEST_ROWS in the test set regardless of the calendar cut.

        Returns:
            (X_train, X_test, y_dir_train, y_dir_test, y_ret_train, y_ret_test)
        """
        split_date = df.index.max() - pd.DateOffset(months=test_months)
        train_mask = df.index <= split_date
        test_mask = df.index > split_date

        # Guarantee minimum test size
        if test_mask.sum() < _MIN_TEST_ROWS:
            n_test = max(_MIN_TEST_ROWS, len(df) // 5)
            train_mask = np.ones(len(df), dtype=bool)
            test_mask = np.zeros(len(df), dtype=bool)
            train_mask[-n_test:] = False
            test_mask[-n_test:] = True
            logger.warning(
                "Calendar-based split produced < %d test rows; "
                "falling back to last %d rows as test set.",
                _MIN_TEST_ROWS,
                n_test,
            )

        if train_mask.sum() < _MIN_TRAIN_ROWS:
            raise ValueError(
                f"Training set has only {train_mask.sum()} rows after split; "
                f"need at least {_MIN_TRAIN_ROWS}. Provide more historical data."
            )

        return (
            X[train_mask],
            X[test_mask],
            y_dir[train_mask],
            y_dir[test_mask],
            y_ret[train_mask],
            y_ret[test_mask],
        )

    @staticmethod
    def _default_params() -> dict:
        """Conservative default XGBoost parameters when tuning is disabled."""
        return {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "gamma": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        }

    def _tune_classifier(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        n_trials: int,
        scale_pos_weight: float,
    ) -> dict:
        """Search XGBClassifier hyperparameters with Optuna.

        Uses 3-fold TimeSeriesSplit within the training data.
        Objective: maximise mean AUC across folds.

        Args:
            X_train:          Feature matrix (training portion only).
            y_train:          Binary target labels.
            n_trials:         Number of Optuna trials.
            scale_pos_weight: Class imbalance weight for positive class.

        Returns:
            Best parameter dict (without scale_pos_weight — applied at
            XGBClassifier construction time in train()).
        """
        tscv = TimeSeriesSplit(n_splits=3)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 0.5),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            }
            aucs = []
            for train_idx, val_idx in tscv.split(X_train):
                if len(val_idx) < 5:
                    # Too few validation samples to compute AUC reliably
                    continue
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]

                # Skip fold if only one class in train or validation
                if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
                    continue

                clf = xgb.XGBClassifier(
                    eval_metric="auc",
                    use_label_encoder=False,
                    verbosity=0,
                    scale_pos_weight=scale_pos_weight,
                    **params,
                )
                clf.fit(X_tr, y_tr)
                probs = clf.predict_proba(X_val)[:, 1]
                try:
                    aucs.append(roc_auc_score(y_val, probs))
                except ValueError:
                    # Only one class in y_val — skip silently
                    pass

            return float(np.mean(aucs)) if aucs else 0.5

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        best_auc = study.best_value
        logger.info("Classifier Optuna best AUC = %.4f", best_auc)
        return study.best_params

    def _tune_regressor(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        n_trials: int,
    ) -> dict:
        """Search XGBRegressor hyperparameters with Optuna.

        Uses 3-fold TimeSeriesSplit within the training data.
        Objective: minimise mean MAE across folds (Optuna maximises negative MAE).

        Args:
            X_train: Feature matrix (training portion only).
            y_train: Continuous target (price_return_24h_target).
            n_trials: Number of Optuna trials.

        Returns:
            Best parameter dict.
        """
        tscv = TimeSeriesSplit(n_splits=3)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 0.5),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            }
            maes = []
            for train_idx, val_idx in tscv.split(X_train):
                if len(val_idx) < 5:
                    continue
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]

                reg = xgb.XGBRegressor(eval_metric="mae", verbosity=0, **params)
                reg.fit(X_tr, y_tr)
                preds = reg.predict(X_val)
                mae = float(np.mean(np.abs(y_val - preds)))
                maes.append(mae)

            return -float(np.mean(maes)) if maes else -999.0

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        best_mae = -study.best_value
        logger.info("Regressor Optuna best MAE = %.6f", best_mae)
        return study.best_params

    def _evaluate(
        self,
        X_test: pd.DataFrame,
        y_dir_test: np.ndarray,
        y_ret_test: np.ndarray,
        n_test_rows: int,
    ) -> Dict[str, Any]:
        """Evaluate both models on the held-out test set and calibrate threshold.

        Threshold calibration selects the probability cutoff that produces
        roughly 2–4 signals per week (i.e. 2–4 out of _PERIODS_PER_WEEK
        periods). We search over a fine grid of candidate thresholds and pick
        the one closest to 3 signals/week while remaining >= 2 and <= 4.
        If no threshold satisfies those bounds, we fall back to the one
        minimising |rate - 3|.

        Simulated Sharpe is computed as the annualised Sharpe of a strategy
        that goes long when predicted_direction=True (using predicted_return_24h
        as the PnL proxy, with the sign of the prediction applied).

        Args:
            X_test:      Feature matrix for the test set.
            y_dir_test:  True binary direction labels for the test set.
            y_ret_test:  True 24h returns for the test set.
            n_test_rows: Total number of test rows (for reference).

        Returns:
            Dict of evaluation metrics.
        """
        dir_probs: np.ndarray = self._clf.predict_proba(X_test)[:, 1]
        ret_preds: np.ndarray = self._reg.predict(X_test)

        # --- Threshold calibration ---
        target_min, target_max, target_ideal = 2.0, 4.0, 3.0
        best_threshold = self._calibrate_threshold(
            dir_probs, target_min, target_max, target_ideal, n_test_rows
        )
        self._threshold = best_threshold
        dir_preds = (dir_probs >= best_threshold).astype(int)

        # --- Classification metrics ---
        n_unique = len(np.unique(y_dir_test))
        if n_unique < 2:
            logger.warning(
                "Test set contains only one class; AUC and some metrics will be 0."
            )
            auc = 0.5
        else:
            auc = float(roc_auc_score(y_dir_test, dir_probs))

        # Handle edge cases where precision/recall/f1 are ill-defined
        _kw = dict(zero_division=0)
        acc = float(accuracy_score(y_dir_test, dir_preds))
        prec = float(precision_score(y_dir_test, dir_preds, **_kw))
        rec = float(recall_score(y_dir_test, dir_preds, **_kw))
        f1 = float(f1_score(y_dir_test, dir_preds, **_kw))

        # --- Simulated Sharpe ---
        # When we predict "up", go long; when we predict "down", go short.
        # PnL proxy per period: sign(prediction) * actual_return
        signal = np.where(dir_preds == 1, 1.0, -1.0)
        pnl = signal * y_ret_test
        # Only count periods where a trade was actually signalled (optional:
        # always trade for a clean Sharpe; using all periods is more conservative)
        sharpe = self._annualised_sharpe(pnl)

        # Trades per week on the test set
        n_signals = int(dir_preds.sum() + (len(dir_preds) - dir_preds.sum()))  # all periods traded
        # Recompute using only periods above threshold (the actual active signals)
        n_active_signals = int((dir_probs >= best_threshold).sum())
        n_trades_per_week = (
            n_active_signals / len(dir_preds) * _PERIODS_PER_WEEK
            if len(dir_preds) > 0
            else 0.0
        )

        metrics: Dict[str, Any] = {
            "accuracy": acc,
            "auc": auc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "sharpe_simulated": sharpe,
            "threshold": best_threshold,
            "n_trades_per_week": round(n_trades_per_week, 2),
            "test_size": n_test_rows,
            "feature_count": len(self._feature_columns),
            "positive_rate_test": float(y_dir_test.mean()),
        }

        # Feature importance (top 10)
        if hasattr(self._clf, "feature_importances_"):
            clf_imp = sorted(
                zip(self._feature_columns, self._clf.feature_importances_),
                key=lambda x: x[1],
                reverse=True,
            )[:10]
            metrics["top_features_classifier"] = {
                k: float(v) for k, v in clf_imp
            }

        if hasattr(self._reg, "feature_importances_"):
            reg_imp = sorted(
                zip(self._feature_columns, self._reg.feature_importances_),
                key=lambda x: x[1],
                reverse=True,
            )[:10]
            metrics["top_features_regressor"] = {
                k: float(v) for k, v in reg_imp
            }

        return metrics

    @staticmethod
    def _calibrate_threshold(
        probs: np.ndarray,
        target_min: float,
        target_max: float,
        target_ideal: float,
        n_test_rows: int,
    ) -> float:
        """Find probability threshold giving ~2-4 signals per week.

        Searches a grid of 200 candidate thresholds in (0.5, 0.99].
        Selects the candidate whose implied signals/week is closest to
        `target_ideal` while falling in [target_min, target_max].
        Falls back to the global closest if none satisfy the bounds.

        Args:
            probs:        Array of predicted positive-class probabilities.
            target_min:   Minimum acceptable signals per week.
            target_max:   Maximum acceptable signals per week.
            target_ideal: Ideal signals per week (used as tie-breaker).
            n_test_rows:  Number of test rows (denominator for frequency).

        Returns:
            Calibrated threshold in [0.5, 0.99].
        """
        if n_test_rows == 0:
            return 0.5

        candidates = np.linspace(0.50, 0.99, 200)
        best_thresh = 0.5
        best_dist = float("inf")
        in_range_best_thresh = None
        in_range_best_dist = float("inf")

        for thresh in candidates:
            n_signals = int((probs >= thresh).sum())
            rate = n_signals / n_test_rows * _PERIODS_PER_WEEK
            dist = abs(rate - target_ideal)

            if dist < best_dist:
                best_dist = dist
                best_thresh = float(thresh)

            if target_min <= rate <= target_max and dist < in_range_best_dist:
                in_range_best_dist = dist
                in_range_best_thresh = float(thresh)

        chosen = in_range_best_thresh if in_range_best_thresh is not None else best_thresh
        logger.info(
            "Threshold calibration: chosen=%.3f (in-range=%s, fallback=%.3f)",
            chosen,
            in_range_best_thresh is not None,
            best_thresh,
        )
        return chosen

    @staticmethod
    def _annualised_sharpe(pnl: np.ndarray) -> float:
        """Compute annualised Sharpe from a per-period PnL array.

        Annualisation factor: sqrt(periods_per_year) where one period = 8h,
        so periods_per_year = 365 * 3 = 1095.

        Returns 0.0 when std is zero or pnl has fewer than 2 elements.
        """
        if len(pnl) < 2:
            return 0.0
        std = float(np.std(pnl, ddof=1))
        if std == 0.0:
            return 0.0
        mean = float(np.mean(pnl))
        periods_per_year = 365 * 3  # 8h periods
        return float((mean / std) * np.sqrt(periods_per_year))

    def _prepare_predict_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Align predict-time features to the training feature set.

        Missing columns are filled with 0; extra columns are dropped.
        Inf / NaN values are replaced with 0.

        Args:
            features: Raw feature DataFrame at inference time.

        Returns:
            DataFrame with exactly self._feature_columns as columns,
            in the same order as during training.
        """
        available = set(features.columns)
        missing = set(self._feature_columns) - available
        if missing:
            logger.warning(
                "predict(): %d feature column(s) missing from input — filling with 0: %s",
                len(missing),
                sorted(missing),
            )

        X = features.reindex(columns=self._feature_columns, fill_value=0.0)
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        return X
