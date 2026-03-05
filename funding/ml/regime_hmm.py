"""
Gaussian Hidden Markov Model for volatility regime detection.

Detects 4 volatility regimes using a GaussianHMM fitted on features
produced by build_regime_features() / get_regime_feature_columns().

Regime ordering (by mean realized volatility):
    0 — low     (lowest volatility HMM state)
    1 — medium
    2 — high
    3 — crisis  (highest volatility HMM state)

Usage:
    from funding.ml.regime_hmm import RegimeHMM
    from funding.ml.regime_features import build_regime_features, get_regime_feature_columns

    df = build_regime_features(["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    hmm = RegimeHMM(n_states=4)
    metrics = hmm.fit(df)
    regime = hmm.predict_regime(df.tail(1)[hmm._feature_columns])
    hmm.save("regime_hmm")
    hmm.load("regime_hmm")
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from hmmlearn.hmm import GaussianHMM
    _HMMLEARN_AVAILABLE = True
except ImportError:
    _HMMLEARN_AVAILABLE = False
    GaussianHMM = None  # type: ignore[assignment,misc]

try:
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "scikit-learn is required for RegimeHMM. "
        "Install it with: pip install scikit-learn"
    ) from exc

try:
    from funding.ml.regime_features import get_regime_feature_columns
except ImportError:
    get_regime_feature_columns = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_DIR = Path("data/funding_models/regime")


class RegimeHMM:
    """
    Gaussian HMM wrapper for 4-state volatility regime detection.

    States are reordered after fitting so that regime 0 always corresponds
    to the HMM state with the lowest mean realized_vol, and regime 3
    corresponds to the highest.

    Attributes:
        _n_states (int): Number of hidden states (default 4).
        _model_dir (Path): Directory used for save/load.
        _model (GaussianHMM or None): Fitted hmmlearn model.
        _feature_columns (list[str]): Feature columns used during fit.
        _metrics (dict): Training metrics (log_likelihood, AIC, BIC, stability).
        _regime_map (dict): Maps HMM internal state index -> regime label (0-3).
        _scaler (StandardScaler or None): Fitted feature scaler.
    """

    def __init__(
        self,
        n_states: int = 4,
        model_dir: Optional[str] = None,
    ) -> None:
        if not _HMMLEARN_AVAILABLE:
            raise ImportError(
                "hmmlearn is required for RegimeHMM. "
                "Install it with: pip install hmmlearn"
            )
        self._n_states: int = n_states
        self._model_dir: Path = Path(model_dir) if model_dir else _DEFAULT_MODEL_DIR
        self._model: Optional[GaussianHMM] = None
        self._feature_columns: List[str] = []
        self._metrics: Dict[str, Any] = {}
        self._regime_map: Dict[int, int] = {}
        self._scaler: Optional[StandardScaler] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def metrics(self) -> Dict[str, Any]:
        """Training metrics dict (log_likelihood, AIC, BIC, stability)."""
        return dict(self._metrics)

    @property
    def regime_map(self) -> Dict[int, int]:
        """Maps HMM internal state index to regime label (0=low vol, 3=crisis)."""
        return dict(self._regime_map)

    @property
    def n_states(self) -> int:
        """Number of HMM hidden states."""
        return self._n_states

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_regime_map(self, X_scaled: np.ndarray) -> Dict[int, int]:
        """
        Order HMM states by mean of realized_vol feature (index 0 if present,
        otherwise the first feature column), ascending.

        Returns a dict mapping {hmm_state_index: regime_label} where
        regime_label 0 = lowest volatility, 3 = highest.

        Args:
            X_scaled: Scaled feature array used to fit the model (n_samples, n_features).

        Returns:
            Dict mapping internal HMM state int -> regime label int (0-3).
        """
        # Determine which column index corresponds to realized_vol
        vol_col_idx = 0
        if "realized_vol_24h" in self._feature_columns:
            vol_col_idx = self._feature_columns.index("realized_vol_24h")
        elif "realized_vol_8h" in self._feature_columns:
            vol_col_idx = self._feature_columns.index("realized_vol_8h")

        # Mean of the vol feature per HMM state (in scaled space is fine for ordering)
        state_vol_means = []
        for state in range(self._n_states):
            # Use HMM means matrix (shape: n_states, n_features)
            mean_val = self._model.means_[state, vol_col_idx]
            state_vol_means.append((state, mean_val))

        # Sort states by vol mean ascending
        sorted_states = sorted(state_vol_means, key=lambda t: t[1])
        regime_map = {hmm_state: regime_label for regime_label, (hmm_state, _) in enumerate(sorted_states)}

        logger.debug(
            "Regime map (hmm_state -> regime_label): %s | vol means: %s",
            regime_map,
            {s: f"{v:.4f}" for s, v in state_vol_means},
        )
        return regime_map

    def _compute_stability(self, state_sequence: np.ndarray) -> float:
        """
        Compute regime stability as the average length of consecutive
        same-state runs.

        Args:
            state_sequence: 1-D array of HMM state predictions.

        Returns:
            Mean consecutive same-state run length (float).
        """
        if len(state_sequence) == 0:
            return 0.0

        runs = []
        current_run = 1
        for i in range(1, len(state_sequence)):
            if state_sequence[i] == state_sequence[i - 1]:
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1
        runs.append(current_run)
        return float(np.mean(runs))

    def _validate_fitted(self, method_name: str) -> None:
        """Raise RuntimeError if the model has not been fitted yet."""
        if self._model is None or not self._feature_columns or self._scaler is None:
            raise RuntimeError(
                f"RegimeHMM.{method_name}() called before fit(). "
                "Call fit() first."
            )

    def _prepare_X(self, features: pd.DataFrame) -> np.ndarray:
        """
        Validate columns, drop NaN rows, scale features.

        Args:
            features: DataFrame with columns matching _feature_columns.

        Returns:
            Scaled numpy array (n_valid_samples, n_features).

        Raises:
            ValueError: If required feature columns are missing.
        """
        missing = [c for c in self._feature_columns if c not in features.columns]
        if missing:
            raise ValueError(
                f"Features DataFrame is missing columns required by the model: {missing}"
            )

        X = features[self._feature_columns].values.astype(np.float64)
        # Replace any remaining NaN/inf with column mean (last resort)
        col_means = np.nanmean(X, axis=0)
        for j in range(X.shape[1]):
            nan_mask = ~np.isfinite(X[:, j])
            if nan_mask.any():
                logger.debug(
                    "Feature '%s' has %d non-finite values; imputing with column mean %.4f",
                    self._feature_columns[j],
                    nan_mask.sum(),
                    col_means[j],
                )
                X[nan_mask, j] = col_means[j] if np.isfinite(col_means[j]) else 0.0
        return self._scaler.transform(X)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Fit the Gaussian HMM on the provided feature DataFrame.

        States are reordered by mean realized volatility so that regime 0
        is always the lowest-volatility state and regime 3 the highest.

        Args:
            df: Feature DataFrame from build_regime_features().  The
                DataFrame is expected to have a DatetimeIndex at 8h
                granularity.  Rows with NaN values are dropped before
                fitting.
            feature_columns: Optional explicit list of feature column names.
                If not provided, get_regime_feature_columns(df) is called.

        Returns:
            Dict with keys:
                log_likelihood (float): Final HMM log-likelihood on train data.
                aic (float): Akaike Information Criterion.
                bic (float): Bayesian Information Criterion.
                stability (float): Mean consecutive same-state run length.
                n_samples (int): Number of training samples after NaN drop.
                n_features (int): Number of features used.
                converged (bool): Whether the EM algorithm converged.

        Raises:
            ValueError: If df is empty after NaN removal or feature columns
                are missing.
            ImportError: If hmmlearn is not installed.
        """
        # Resolve feature columns
        if feature_columns is not None:
            self._feature_columns = list(feature_columns)
        elif get_regime_feature_columns is not None:
            self._feature_columns = get_regime_feature_columns(df)
        else:
            # Fallback: all numeric columns except known non-features
            exclude = {"symbol", "regime_label"}
            self._feature_columns = [
                c for c in df.columns
                if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
            ]

        if not self._feature_columns:
            raise ValueError(
                "No feature columns found in DataFrame. "
                "Pass feature_columns explicitly or ensure df has numeric columns."
            )

        # Extract and clean feature matrix
        X_raw = df[self._feature_columns].copy()
        before_drop = len(X_raw)
        X_raw = X_raw.dropna()
        after_drop = len(X_raw)

        if after_drop == 0:
            raise ValueError(
                "DataFrame is empty after dropping NaN rows. "
                "Ensure sufficient history is provided (rolling windows need warm-up)."
            )

        if before_drop != after_drop:
            logger.info(
                "Dropped %d/%d rows with NaN values before HMM fit",
                before_drop - after_drop,
                before_drop,
            )

        X_np = X_raw.values.astype(np.float64)
        n_samples, n_features = X_np.shape

        logger.info(
            "Fitting GaussianHMM: n_states=%d, n_samples=%d, n_features=%d",
            self._n_states,
            n_samples,
            n_features,
        )

        # Fit scaler
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_np)

        # Fit HMM
        self._model = GaussianHMM(
            n_components=self._n_states,
            covariance_type="full",
            n_iter=100,
            random_state=42,
        )
        self._model.fit(X_scaled)

        # Build regime map (reorder states by vol level)
        self._regime_map = self._build_regime_map(X_scaled)

        # Compute metrics
        log_likelihood = self._model.score(X_scaled)

        # Number of free parameters for full covariance HMM:
        #   transition matrix: n_states * (n_states - 1)
        #   initial probs:     n_states - 1
        #   means:             n_states * n_features
        #   full covariance:   n_states * n_features * (n_features + 1) / 2
        n_params = (
            self._n_states * (self._n_states - 1)
            + (self._n_states - 1)
            + self._n_states * n_features
            + self._n_states * n_features * (n_features + 1) // 2
        )
        aic = 2 * n_params - 2 * log_likelihood
        bic = n_params * np.log(n_samples) - 2 * log_likelihood

        state_sequence = self._model.predict(X_scaled)
        stability = self._compute_stability(state_sequence)

        self._metrics = {
            "log_likelihood": float(log_likelihood),
            "aic": float(aic),
            "bic": float(bic),
            "stability": float(stability),
            "n_samples": int(n_samples),
            "n_features": int(n_features),
            "converged": bool(self._model.monitor_.converged),
        }

        logger.info(
            "HMM fit complete: log_likelihood=%.4f, AIC=%.2f, BIC=%.2f, "
            "stability=%.2f, converged=%s",
            log_likelihood,
            aic,
            bic,
            stability,
            self._metrics["converged"],
        )
        return dict(self._metrics)

    def predict_regime(self, features: pd.DataFrame) -> int:
        """
        Predict the current volatility regime for the given feature row(s).

        If multiple rows are passed, the prediction for the **last** row is
        returned (most recent observation).

        Args:
            features: DataFrame with columns matching _feature_columns.
                      Typically a single-row slice of the output of
                      build_regime_features().

        Returns:
            Regime integer: 0 (low vol) to 3 (crisis).

        Raises:
            RuntimeError: If model has not been fitted.
            ValueError: If required feature columns are missing.
        """
        self._validate_fitted("predict_regime")
        X_scaled = self._prepare_X(features)
        hmm_states = self._model.predict(X_scaled)
        # Take the last predicted state (most recent observation)
        hmm_state = int(hmm_states[-1])
        regime = self._regime_map.get(hmm_state, hmm_state)
        logger.debug(
            "predict_regime: hmm_state=%d -> regime=%d",
            hmm_state,
            regime,
        )
        return int(regime)

    def predict_regime_proba(self, features: pd.DataFrame) -> np.ndarray:
        """
        Return posterior probabilities for each regime, mapped to regime order.

        The returned array has shape (n_rows, n_states), where column *i*
        corresponds to regime *i* (0=low vol, 3=crisis).

        Args:
            features: DataFrame with columns matching _feature_columns.

        Returns:
            np.ndarray of shape (n_rows, n_states) with posterior probabilities.
            Each row sums to 1.0.

        Raises:
            RuntimeError: If model has not been fitted.
            ValueError: If required feature columns are missing.
        """
        self._validate_fitted("predict_regime_proba")
        X_scaled = self._prepare_X(features)
        # predict_proba returns shape (n_samples, n_states) in HMM state order
        raw_proba = self._model.predict_proba(X_scaled)

        # Reorder columns: column index = regime label, value from HMM state index
        # regime_map: {hmm_state -> regime_label}
        n_rows = raw_proba.shape[0]
        regime_proba = np.zeros((n_rows, self._n_states), dtype=np.float64)

        for hmm_state, regime_label in self._regime_map.items():
            if hmm_state < raw_proba.shape[1] and regime_label < self._n_states:
                regime_proba[:, regime_label] = raw_proba[:, hmm_state]

        return regime_proba

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, name: str = "regime_hmm") -> None:
        """
        Persist the fitted model, scaler, and metadata to disk.

        Files written to self._model_dir:
            <name>_model.pkl   — pickled GaussianHMM
            <name>_scaler.pkl  — pickled StandardScaler
            <name>_meta.json   — JSON with regime_map, metrics, feature_columns

        Args:
            name: Base filename stem (without extension).

        Raises:
            RuntimeError: If model has not been fitted.
        """
        self._validate_fitted("save")
        self._model_dir.mkdir(parents=True, exist_ok=True)

        model_path = self._model_dir / f"{name}_model.pkl"
        scaler_path = self._model_dir / f"{name}_scaler.pkl"
        meta_path = self._model_dir / f"{name}_meta.json"

        with open(model_path, "wb") as fh:
            pickle.dump(self._model, fh, protocol=pickle.HIGHEST_PROTOCOL)

        with open(scaler_path, "wb") as fh:
            pickle.dump(self._scaler, fh, protocol=pickle.HIGHEST_PROTOCOL)

        meta = {
            "n_states": self._n_states,
            "feature_columns": self._feature_columns,
            # JSON keys must be strings
            "regime_map": {str(k): v for k, v in self._regime_map.items()},
            "metrics": self._metrics,
        }
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

        logger.info(
            "RegimeHMM saved: model=%s, scaler=%s, meta=%s",
            model_path,
            scaler_path,
            meta_path,
        )

    def load(self, name: str = "regime_hmm") -> "RegimeHMM":
        """
        Load a previously saved model, scaler, and metadata from disk.

        Args:
            name: Base filename stem matching the one used in save().

        Returns:
            self (for chaining).

        Raises:
            FileNotFoundError: If any of the three files are missing.
        """
        model_path = self._model_dir / f"{name}_model.pkl"
        scaler_path = self._model_dir / f"{name}_scaler.pkl"
        meta_path = self._model_dir / f"{name}_meta.json"

        for path in (model_path, scaler_path, meta_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"RegimeHMM load failed: file not found: {path}"
                )

        with open(model_path, "rb") as fh:
            self._model = pickle.load(fh)

        with open(scaler_path, "rb") as fh:
            self._scaler = pickle.load(fh)

        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)

        self._n_states = int(meta["n_states"])
        self._feature_columns = list(meta["feature_columns"])
        # JSON keys are strings; restore int keys
        self._regime_map = {int(k): int(v) for k, v in meta["regime_map"].items()}
        self._metrics = meta.get("metrics", {})

        logger.info(
            "RegimeHMM loaded from %s: n_states=%d, features=%d",
            self._model_dir / name,
            self._n_states,
            len(self._feature_columns),
        )
        return self
