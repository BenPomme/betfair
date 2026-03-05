"""
Funding rate predictor using gradient boosted trees.
Uses XGBoost if available, falls back to sklearn GradientBoosting.

Two sub-models:
  1. Direction classifier: will next funding rate be positive? (binary)
  2. Magnitude regressor: how large will the rate be? (regression)

Training uses walk-forward validation to prevent lookahead bias.
Hyperparameter tuning via Optuna on validation set.
"""
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    roc_auc_score,
)

# Try XGBoost first, fall back to sklearn
try:
    import xgboost as xgb
    _USE_XGB = True
except Exception:
    _USE_XGB = False

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from funding.ml.feature_engineer import build_features_all_symbols, get_feature_columns

logger = logging.getLogger(__name__)

MODEL_DIR = Path("data/funding_models")

# Suppress Optuna info logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _make_classifier(params: dict):
    """Create a classifier with the given params."""
    if _USE_XGB:
        return xgb.XGBClassifier(
            eval_metric="auc",
            use_label_encoder=False,
            verbosity=0,
            **params,
        )
    else:
        sk_params = {
            "n_estimators": params.get("n_estimators", 300),
            "max_depth": params.get("max_depth", 6),
            "learning_rate": params.get("learning_rate", 0.05),
            "subsample": params.get("subsample", 0.8),
            "min_samples_leaf": params.get("min_child_samples", 20),
        }
        return GradientBoostingClassifier(**sk_params)


def _make_regressor(params: dict):
    """Create a regressor with the given params."""
    if _USE_XGB:
        return xgb.XGBRegressor(
            eval_metric="mae",
            verbosity=0,
            **params,
        )
    else:
        sk_params = {
            "n_estimators": params.get("n_estimators", 300),
            "max_depth": params.get("max_depth", 6),
            "learning_rate": params.get("learning_rate", 0.05),
            "subsample": params.get("subsample", 0.8),
            "min_samples_leaf": params.get("min_child_samples", 20),
        }
        return GradientBoostingRegressor(**sk_params)


class FundingPredictor:
    """Gradient boosted tree funding rate predictor."""

    def __init__(self, model_dir: Optional[Path] = None):
        self._model_dir = model_dir or MODEL_DIR
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._classifier = None
        self._regressor = None
        self._feature_cols: List[str] = []
        self._metrics: Dict[str, Any] = {}

    def train(
        self,
        df: pd.DataFrame,
        tune: bool = True,
        n_trials: int = 30,
        test_months: int = 2,
    ) -> Dict[str, Any]:
        """Train direction classifier + magnitude regressor.

        Uses walk-forward: train on all data except last `test_months`,
        validate on last `test_months`.
        """
        self._feature_cols = get_feature_columns(df)
        if not self._feature_cols:
            raise ValueError("No feature columns found")

        X = df[self._feature_cols].copy().fillna(0).replace([np.inf, -np.inf], 0)
        y_direction = df["next_rate_positive"].values
        y_magnitude = df["next_funding_rate"].values

        # Walk-forward split
        split_date = df.index.max() - pd.DateOffset(months=test_months)
        train_mask = df.index <= split_date
        test_mask = df.index > split_date

        X_train, X_test = X[train_mask], X[test_mask]
        y_dir_train, y_dir_test = y_direction[train_mask], y_direction[test_mask]
        y_mag_train, y_mag_test = y_magnitude[train_mask], y_magnitude[test_mask]

        logger.info(
            "Train: %d rows, Test: %d rows, Features: %d, Split: %s, Engine: %s",
            len(X_train), len(X_test), len(self._feature_cols), split_date,
            "XGBoost" if _USE_XGB else "sklearn",
        )

        # Get params (tune or default)
        if tune:
            clf_params = self._tune_classifier(X_train, y_dir_train, n_trials)
            reg_params = self._tune_regressor(X_train, y_mag_train, n_trials)
        else:
            clf_params = {
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 5,
            }
            reg_params = dict(clf_params)

        # Train classifier
        self._classifier = _make_classifier(clf_params)
        self._classifier.fit(X_train, y_dir_train)

        # Train regressor
        self._regressor = _make_regressor(reg_params)
        self._regressor.fit(X_train, y_mag_train)

        # Evaluate
        self._metrics = self._evaluate(X_test, y_dir_test, y_mag_test)
        logger.info("Model metrics: %s", {k: v for k, v in self._metrics.items()
                                           if not k.startswith("top_features")})
        return self._metrics

    def _tune_classifier(self, X: pd.DataFrame, y: np.ndarray, n_trials: int) -> dict:
        """Tune classifier with Optuna."""
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 3, 20),
            }
            split = int(len(X) * 0.8)
            clf = _make_classifier(params)
            clf.fit(X.iloc[:split], y[:split])
            preds = clf.predict_proba(X.iloc[split:])[:, 1]
            return roc_auc_score(y[split:], preds)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        logger.info("Best classifier AUC=%.4f", study.best_value)
        return study.best_params

    def _tune_regressor(self, X: pd.DataFrame, y: np.ndarray, n_trials: int) -> dict:
        """Tune regressor with Optuna."""
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 3, 20),
            }
            split = int(len(X) * 0.8)
            reg = _make_regressor(params)
            reg.fit(X.iloc[:split], y[:split])
            preds = reg.predict(X.iloc[split:])
            return -mean_absolute_error(y[split:], preds)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        logger.info("Best regressor MAE=%.6f", -study.best_value)
        return study.best_params

    def _evaluate(self, X_test, y_dir_test, y_mag_test) -> Dict[str, Any]:
        """Evaluate both models on test set."""
        dir_probs = self._classifier.predict_proba(X_test)[:, 1]
        dir_preds = (dir_probs > 0.5).astype(int)
        mag_preds = self._regressor.predict(X_test)

        metrics = {
            "direction_accuracy": float(accuracy_score(y_dir_test, dir_preds)),
            "direction_auc": float(roc_auc_score(y_dir_test, dir_probs)),
            "direction_f1": float(f1_score(y_dir_test, dir_preds)),
            "magnitude_mae": float(mean_absolute_error(y_mag_test, mag_preds)),
            "test_size": len(X_test),
            "positive_rate": float(y_dir_test.mean()),
            "feature_count": len(self._feature_cols),
            "engine": "xgboost" if _USE_XGB else "sklearn",
        }

        # Feature importance (top 10)
        if hasattr(self._classifier, "feature_importances_"):
            clf_imp = dict(
                sorted(
                    zip(self._feature_cols, self._classifier.feature_importances_),
                    key=lambda x: x[1], reverse=True,
                )[:10]
            )
            metrics["top_features_classifier"] = {k: float(v) for k, v in clf_imp.items()}

        if hasattr(self._regressor, "feature_importances_"):
            reg_imp = dict(
                sorted(
                    zip(self._feature_cols, self._regressor.feature_importances_),
                    key=lambda x: x[1], reverse=True,
                )[:10]
            )
            metrics["top_features_regressor"] = {k: float(v) for k, v in reg_imp.items()}

        return metrics

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Predict next funding rate for new data."""
        if self._classifier is None or self._regressor is None:
            raise RuntimeError("Model not trained. Call train() first.")

        X = features[self._feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        dir_probs = self._classifier.predict_proba(X)[:, 1]
        mag_preds = self._regressor.predict(X)

        result = pd.DataFrame(index=features.index)
        result["direction_prob"] = dir_probs
        result["predicted_rate"] = mag_preds
        result["confidence"] = np.abs(dir_probs - 0.5) * 2
        result["predicted_positive"] = (dir_probs > 0.5).astype(int)
        return result

    def save(self, name: str = "funding_predictor") -> None:
        """Save models to disk."""
        with open(self._model_dir / f"{name}_clf.pkl", "wb") as f:
            pickle.dump(self._classifier, f)
        with open(self._model_dir / f"{name}_reg.pkl", "wb") as f:
            pickle.dump(self._regressor, f)

        meta = {
            "feature_cols": self._feature_cols,
            "metrics": self._metrics,
            "name": name,
        }
        with open(self._model_dir / f"{name}_meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)
        logger.info("Saved model '%s' to %s", name, self._model_dir)

    def load(self, name: str = "funding_predictor") -> None:
        """Load models from disk."""
        with open(self._model_dir / f"{name}_clf.pkl", "rb") as f:
            self._classifier = pickle.load(f)
        with open(self._model_dir / f"{name}_reg.pkl", "rb") as f:
            self._regressor = pickle.load(f)

        meta_path = self._model_dir / f"{name}_meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        self._feature_cols = meta["feature_cols"]
        self._metrics = meta.get("metrics", {})
        logger.info("Loaded model '%s' with %d features", name, len(self._feature_cols))

    @property
    def metrics(self) -> Dict[str, Any]:
        return self._metrics

    @property
    def feature_columns(self) -> List[str]:
        return self._feature_cols


def walk_forward_evaluate(
    df: pd.DataFrame,
    n_splits: int = 6,
    train_months: int = 6,
    test_months: int = 1,
    tune: bool = False,
) -> List[Dict[str, Any]]:
    """Walk-forward cross-validation."""
    results = []
    max_date = df.index.max()

    for i in range(n_splits):
        test_end = max_date - pd.DateOffset(months=i * test_months)
        test_start = test_end - pd.DateOffset(months=test_months)

        train_df = df[df.index <= test_start]
        test_df = df[(df.index > test_start) & (df.index <= test_end)]

        if len(train_df) < 100 or len(test_df) < 10:
            continue

        fold_df = pd.concat([train_df, test_df])
        predictor = FundingPredictor()
        metrics = predictor.train(fold_df, tune=tune, test_months=test_months)
        metrics["fold"] = n_splits - i
        metrics["train_end"] = str(test_start)
        metrics["test_end"] = str(test_end)
        results.append(metrics)

        logger.info(
            "Fold %d: dir_acc=%.3f, auc=%.3f, mag_mae=%.6f",
            metrics["fold"], metrics["direction_accuracy"],
            metrics["direction_auc"], metrics["magnitude_mae"],
        )

    return results
