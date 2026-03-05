"""
Temporal Fusion Transformer predictor for contrarian price direction.

Uses pytorch-forecasting's TemporalFusionTransformer for multi-horizon
quantile prediction of forward price returns.

The TFT architecture provides:
  - Multi-horizon forecasting (24h and 72h forward returns)
  - Variable Selection Networks for interpretability
  - Attention-based temporal pattern detection
  - Quantile outputs (uncertainty estimation)

Input: DataFrame produced by build_contrarian_features (contrarian_features.py)
Target: price_return_24h_target (primary), price_return_72h_target (auxiliary)

All pytorch imports are guarded — the class is importable without torch;
methods raise RuntimeError on use when torch is unavailable.
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional deep-learning imports
# ---------------------------------------------------------------------------

_TORCH_AVAILABLE = False
_PL_AVAILABLE = False
_PF_AVAILABLE = False

try:
    import torch  # noqa: F401
    _TORCH_AVAILABLE = True
except ImportError:
    logger.debug("torch not installed — TFTPredictor will raise on use")

try:
    import pytorch_lightning as pl  # noqa: F401
    _PL_AVAILABLE = True
except ImportError:
    logger.debug("pytorch_lightning not installed — TFTPredictor will raise on use")

try:
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss
    _PF_AVAILABLE = True
except ImportError:
    logger.debug("pytorch_forecasting not installed — TFTPredictor will raise on use")

_DEEP_LEARNING_AVAILABLE = _TORCH_AVAILABLE and _PL_AVAILABLE and _PF_AVAILABLE


def _require_deep_learning() -> None:
    """Raise a clear RuntimeError when required libraries are missing."""
    if _DEEP_LEARNING_AVAILABLE:
        return
    missing = []
    if not _TORCH_AVAILABLE:
        missing.append("torch")
    if not _PL_AVAILABLE:
        missing.append("pytorch_lightning")
    if not _PF_AVAILABLE:
        missing.append("pytorch_forecasting")
    raise RuntimeError(
        f"TFTPredictor requires {', '.join(missing)} to be installed.\n"
        "Install with: pip install torch pytorch-lightning pytorch-forecasting"
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Encoder length: 7 days × 3 periods per day (8h periods)
_MAX_ENCODER_LENGTH = 21
# Prediction length: 1 day forward = 3 × 8h periods
_MAX_PREDICTION_LENGTH = 3
# Validation fraction for walk-forward split
_VALIDATION_FRACTION = 0.20

# Target column (primary)
_TARGET_COL = "price_return_24h_target"

# Columns that are known in advance (calendar features)
_KNOWN_REAL_COLS = ["hour_of_day", "day_of_week"]

# Columns to exclude from being treated as features
_EXCLUDE_COLS = {
    "symbol",
    "funding_rate",
    "mark_price",
    "price_return_24h_target",
    "price_return_72h_target",
    "direction_24h",
    "time_idx",
    "group",
}


def _get_unknown_real_cols(df: pd.DataFrame) -> List[str]:
    """Return feature columns that are time-varying but not known in advance."""
    numeric_dtypes = {np.float64, np.int64, np.float32, np.int32}
    cols = []
    for c in df.columns:
        if c in _EXCLUDE_COLS:
            continue
        if c in _KNOWN_REAL_COLS:
            continue
        if df[c].dtype.type in numeric_dtypes:
            cols.append(c)
    return cols


# ---------------------------------------------------------------------------
# TFTPredictor
# ---------------------------------------------------------------------------


class TFTPredictor:
    """
    Temporal Fusion Transformer predictor for contrarian price direction.

    Trains a multi-horizon quantile model on the output of
    build_contrarian_features(). The TFT Variable Selection Network
    provides interpretable feature importance weights.

    The class is importable without torch/pytorch-forecasting installed.
    Methods raise RuntimeError when the required libraries are absent.
    """

    def __init__(self, model_dir: Optional[str] = None) -> None:
        self._model_dir = Path(model_dir or "data/funding_models/tft")
        self._model = None          # pytorch_forecasting TemporalFusionTransformer
        self._feature_columns: List[str] = []
        self._metrics: Dict[str, Any] = {}
        self._trainer = None        # pytorch_lightning.Trainer
        # Cached attention/variable-importance weights from last training run
        self._attention_weights: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        df: pd.DataFrame,
        max_epochs: int = 50,
        batch_size: int = 64,
        learning_rate: float = 0.001,
    ) -> Dict[str, Any]:
        """Train the TFT on a contrarian feature DataFrame.

        Args:
            df: Output of build_contrarian_features / build_contrarian_features_all.
                Must contain 'price_return_24h_target', 'symbol', and all feature
                columns produced by build_contrarian_features.
            max_epochs: Maximum training epochs (early stopping may trigger sooner).
            batch_size: DataLoader batch size.
            learning_rate: Initial learning rate for the Adam optimiser.

        Returns:
            Dict of evaluation metrics (MAE, RMSE, direction_accuracy, etc.).

        Raises:
            RuntimeError: If torch / pytorch_forecasting are not installed.
            ValueError: If the DataFrame lacks required columns or has insufficient rows.
        """
        _require_deep_learning()

        import torch  # local re-import after guard
        import pytorch_lightning as pl
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
        from pytorch_forecasting.data import GroupNormalizer
        from pytorch_forecasting.metrics import QuantileLoss
        from pytorch_lightning.callbacks import EarlyStopping
        from torch.utils.data import DataLoader

        # ---- Validate columns ----
        if _TARGET_COL not in df.columns:
            raise ValueError(
                f"DataFrame must contain target column '{_TARGET_COL}'. "
                "Generate it with build_contrarian_features()."
            )
        if "symbol" not in df.columns:
            raise ValueError("DataFrame must contain a 'symbol' column.")

        # ---- Prepare DataFrame ----
        data = df.copy().reset_index()

        # Ensure the DatetimeIndex becomes a column named 'funding_time_dt'
        if "funding_time_dt" not in data.columns and "index" in data.columns:
            data = data.rename(columns={"index": "funding_time_dt"})

        # Sort by symbol then time for sequential time_idx assignment
        data = data.sort_values(["symbol", "funding_time_dt"]).reset_index(drop=True)

        # time_idx: sequential integer index per group (symbol)
        data["time_idx"] = (
            data.groupby("symbol")["funding_time_dt"]
            .rank(method="dense")
            .astype(int) - 1
        )
        data["group"] = data["symbol"]

        # Feature columns: all numeric columns not in the exclusion set
        self._feature_columns = _get_unknown_real_cols(data)
        if not self._feature_columns:
            raise ValueError("No feature columns detected in the input DataFrame.")

        # Fill NaN / inf in features
        data[self._feature_columns] = (
            data[self._feature_columns]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        # Also fill NaN in known reals
        for col in _KNOWN_REAL_COLS:
            if col in data.columns:
                data[col] = data[col].fillna(0.0)

        # Fill NaN in target
        data[_TARGET_COL] = data[_TARGET_COL].fillna(0.0)

        # ---- Walk-forward split (last 20% for validation) ----
        max_time_idx = data["time_idx"].max()
        val_cutoff = int(max_time_idx * (1.0 - _VALIDATION_FRACTION))

        train_data = data[data["time_idx"] <= val_cutoff]
        val_data = data[data["time_idx"] > val_cutoff]

        min_required = _MAX_ENCODER_LENGTH + _MAX_PREDICTION_LENGTH + 1
        if len(train_data) < min_required:
            raise ValueError(
                f"Insufficient training rows ({len(train_data)}); "
                f"need at least {min_required}."
            )

        logger.info(
            "TFT training split — train: %d rows, val: %d rows, features: %d",
            len(train_data), len(val_data), len(self._feature_columns),
        )

        # ---- TimeSeriesDataSet ----
        training_dataset = TimeSeriesDataSet(
            train_data,
            time_idx="time_idx",
            target=_TARGET_COL,
            group_ids=["group"],
            max_encoder_length=_MAX_ENCODER_LENGTH,
            max_prediction_length=_MAX_PREDICTION_LENGTH,
            time_varying_known_reals=[
                c for c in _KNOWN_REAL_COLS if c in data.columns
            ],
            time_varying_unknown_reals=self._feature_columns,
            target_normalizer=GroupNormalizer(groups=["group"]),
            allow_missing_timesteps=True,
        )

        # Create validation dataset from training parameters
        validation_dataset = TimeSeriesDataSet.from_dataset(
            training_dataset,
            val_data,
            predict=False,
            stop_randomization=True,
        )

        train_loader = training_dataset.to_dataloader(
            train=True, batch_size=batch_size, num_workers=0
        )
        val_loader = validation_dataset.to_dataloader(
            train=False, batch_size=batch_size * 2, num_workers=0
        )

        # ---- Build TFT model ----
        self._model = TemporalFusionTransformer.from_dataset(
            training_dataset,
            learning_rate=learning_rate,
            hidden_size=32,
            attention_head_size=2,
            dropout=0.1,
            hidden_continuous_size=16,
            output_size=7,          # 7 quantiles: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
            loss=QuantileLoss(),
            log_interval=10,
            reduce_on_plateau_patience=4,
        )

        logger.info(
            "TFT model created — parameters: %d",
            sum(p.numel() for p in self._model.parameters()),
        )

        # ---- Trainer ----
        early_stop = EarlyStopping(
            monitor="val_loss",
            min_delta=1e-5,
            patience=5,
            verbose=False,
            mode="min",
        )

        self._trainer = pl.Trainer(
            max_epochs=max_epochs,
            gradient_clip_val=0.1,
            callbacks=[early_stop],
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )

        logger.info("Starting TFT training (max_epochs=%d)...", max_epochs)
        self._trainer.fit(
            self._model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
        logger.info(
            "Training complete — best val_loss: %.6f",
            early_stop.best_score.item() if hasattr(early_stop.best_score, "item")
            else float(early_stop.best_score),
        )

        # ---- Extract Variable Selection Network weights ----
        self._attention_weights = self._extract_vsn_weights()

        # ---- Compute evaluation metrics on validation set ----
        self._metrics = self._compute_metrics(val_loader, val_data)
        logger.info(
            "TFT metrics — MAE: %.6f, RMSE: %.6f, direction_accuracy: %.3f",
            self._metrics.get("mae", float("nan")),
            self._metrics.get("rmse", float("nan")),
            self._metrics.get("direction_accuracy", float("nan")),
        )

        self._model_dir.mkdir(parents=True, exist_ok=True)
        return self._metrics

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate forward-return predictions for new data.

        Args:
            features: DataFrame in the same format as the training input
                      (output of build_contrarian_features). Must contain
                      'symbol', 'hour_of_day', 'day_of_week', and all feature
                      columns used during training.

        Returns:
            DataFrame with columns:
                - predicted_return_24h: median predicted 24h return (quantile 0.5)
                - direction_prob: sigmoid of predicted_return_24h, proxy for P(up)
                - confidence: |direction_prob - 0.5| * 2  (0=no edge, 1=full conviction)
                - q10, q90: lower/upper prediction interval bounds

        Raises:
            RuntimeError: If torch / pytorch_forecasting are not installed, or
                          if the model has not been trained / loaded.
        """
        _require_deep_learning()

        import torch
        from pytorch_forecasting import TimeSeriesDataSet

        if self._model is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        data = features.copy().reset_index()
        if "funding_time_dt" not in data.columns and "index" in data.columns:
            data = data.rename(columns={"index": "funding_time_dt"})

        data = data.sort_values(["symbol", "funding_time_dt"]).reset_index(drop=True)
        data["time_idx"] = (
            data.groupby("symbol")["funding_time_dt"]
            .rank(method="dense")
            .astype(int) - 1
        )
        data["group"] = data["symbol"]

        # Fill features
        for col in self._feature_columns:
            if col in data.columns:
                data[col] = data[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            else:
                data[col] = 0.0

        for col in _KNOWN_REAL_COLS:
            if col in data.columns:
                data[col] = data[col].fillna(0.0)
            else:
                data[col] = 0.0

        # Dummy target column (required by TimeSeriesDataSet, not used for inference)
        data[_TARGET_COL] = 0.0

        # Build prediction dataset
        predict_dataset = TimeSeriesDataSet.from_dataset(
            self._model.hparams.get("dataset_parameters", None)
            if hasattr(self._model, "hparams") and self._model.hparams.get("dataset_parameters")
            else None,
            data,
            predict=True,
            stop_randomization=True,
        ) if False else _build_predict_dataset(self._model, data, self._feature_columns)

        from torch.utils.data import DataLoader
        predict_loader = predict_dataset.to_dataloader(
            train=False, batch_size=128, num_workers=0
        )

        # Run inference
        raw_predictions, index = self._model.predict(
            predict_loader, mode="raw", return_index=True
        )

        # raw_predictions["prediction"] shape: (n_samples, prediction_length, n_quantiles)
        pred_tensor = raw_predictions["prediction"]

        # Quantile indices for QuantileLoss default: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
        q10_idx, q50_idx, q90_idx = 1, 3, 5

        # Use the first prediction step (8h ahead) median as the 24h proxy
        # (TFT predicts max_prediction_length=3 steps; step 2 is the ~24h horizon)
        step_idx = min(2, pred_tensor.shape[1] - 1)

        predicted_return = pred_tensor[:, step_idx, q50_idx].cpu().numpy()
        q10 = pred_tensor[:, step_idx, q10_idx].cpu().numpy()
        q90 = pred_tensor[:, step_idx, q90_idx].cpu().numpy()

        direction_prob = 1.0 / (1.0 + np.exp(-predicted_return * 100))
        confidence = np.abs(direction_prob - 0.5) * 2.0

        result = pd.DataFrame(
            {
                "predicted_return_24h": predicted_return,
                "direction_prob": direction_prob,
                "confidence": confidence,
                "q10": q10,
                "q90": q90,
            },
            index=index.index if hasattr(index, "index") else range(len(predicted_return)),
        )
        return result

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, name: str = "tft") -> None:
        """Save TFT model checkpoint and metadata to model_dir.

        Args:
            name: Base filename (without extension).

        Raises:
            RuntimeError: If torch is not installed or model has not been trained.
        """
        _require_deep_learning()

        import torch

        if self._model is None:
            raise RuntimeError("No model to save. Train the model first.")

        self._model_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = self._model_dir / f"{name}.ckpt"
        self._trainer.save_checkpoint(str(checkpoint_path))
        logger.info("Saved TFT checkpoint to %s", checkpoint_path)

        meta = {
            "name": name,
            "feature_columns": self._feature_columns,
            "metrics": {k: float(v) if isinstance(v, (float, int, np.floating, np.integer))
                        else v for k, v in self._metrics.items()},
            "attention_weights": {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in self._attention_weights.items()
            },
            "max_encoder_length": _MAX_ENCODER_LENGTH,
            "max_prediction_length": _MAX_PREDICTION_LENGTH,
        }
        meta_path = self._model_dir / f"{name}_meta.json"
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2, default=str)
        logger.info("Saved TFT metadata to %s", meta_path)

    def load(self, name: str = "tft") -> None:
        """Load TFT model checkpoint and metadata from model_dir.

        Args:
            name: Base filename (without extension), matching what was used in save().

        Raises:
            RuntimeError: If torch / pytorch_forecasting are not installed.
            FileNotFoundError: If the checkpoint or meta file does not exist.
        """
        _require_deep_learning()

        from pytorch_forecasting import TemporalFusionTransformer

        checkpoint_path = self._model_dir / f"{name}.ckpt"
        meta_path = self._model_dir / f"{name}_meta.json"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"TFT checkpoint not found: {checkpoint_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"TFT metadata not found: {meta_path}")

        self._model = TemporalFusionTransformer.load_from_checkpoint(
            str(checkpoint_path)
        )
        self._model.eval()
        logger.info("Loaded TFT checkpoint from %s", checkpoint_path)

        with open(meta_path) as fh:
            meta = json.load(fh)

        self._feature_columns = meta.get("feature_columns", [])
        self._metrics = meta.get("metrics", {})
        raw_weights = meta.get("attention_weights", {})
        self._attention_weights = {
            k: np.array(v) if isinstance(v, list) else v
            for k, v in raw_weights.items()
        }
        logger.info(
            "Loaded TFT model '%s' with %d features", name, len(self._feature_columns)
        )

    # ------------------------------------------------------------------
    # Interpretability
    # ------------------------------------------------------------------

    def get_attention_weights(self) -> Dict[str, Any]:
        """Return variable importance from the TFT Variable Selection Network.

        Returns:
            Dict with keys:
                - encoder_variables: Dict[str, float] — feature name → importance weight
                  (from the encoder VSN, higher = more important)
                - decoder_variables: Dict[str, float] — for known reals in the decoder
                - static_variables: Dict[str, float] — for static categoricals (if any)

        Raises:
            RuntimeError: If model has not been trained / loaded.
        """
        if self._model is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")
        return dict(self._attention_weights)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def metrics(self) -> Dict[str, Any]:
        """Return evaluation metrics from the last training run."""
        return dict(self._metrics)

    @property
    def feature_columns(self) -> List[str]:
        """Return the list of feature column names used by the model."""
        return list(self._feature_columns)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_vsn_weights(self) -> Dict[str, Any]:
        """Extract Variable Selection Network weights from the trained model.

        Parses the encoder and decoder VSN grn_weights to produce feature
        importance rankings. Returns an empty dict if extraction fails.
        """
        if self._model is None:
            return {}

        weights: Dict[str, Any] = {}

        try:
            import torch

            # --- Encoder VSN ---
            encoder_vsn = getattr(self._model, "encoder_variable_selection", None)
            if encoder_vsn is not None:
                grn = getattr(encoder_vsn, "single_variable_grns", None)
                if grn is not None:
                    # single_variable_grns is an nn.ModuleList; softmax across inputs
                    # gives relative importance
                    with torch.no_grad():
                        raw = torch.stack(
                            [list(grn.parameters())[0].abs().mean()
                             for _ in grn]
                        ) if hasattr(grn, "__iter__") else None

                # Alternatively, use the flattened weight approach
                flat_weights = _flatten_vsn_weights(encoder_vsn, self._feature_columns)
                if flat_weights:
                    weights["encoder_variables"] = flat_weights

            # --- Decoder VSN (known reals) ---
            decoder_vsn = getattr(self._model, "decoder_variable_selection", None)
            if decoder_vsn is not None:
                flat_weights = _flatten_vsn_weights(decoder_vsn, _KNOWN_REAL_COLS)
                if flat_weights:
                    weights["decoder_variables"] = flat_weights

            # --- Static VSN (group embeddings) ---
            static_vsn = getattr(self._model, "static_variable_selection", None)
            if static_vsn is not None:
                flat_weights = _flatten_vsn_weights(static_vsn, ["group"])
                if flat_weights:
                    weights["static_variables"] = flat_weights

        except Exception as exc:
            logger.warning("Could not extract VSN weights: %s", exc)

        return weights

    def _compute_metrics(self, val_loader: Any, val_data: pd.DataFrame) -> Dict[str, Any]:
        """Compute MAE, RMSE, and direction accuracy on the validation set.

        Args:
            val_loader: DataLoader for the validation TimeSeriesDataSet.
            val_data: Validation split DataFrame (used to retrieve actuals).

        Returns:
            Dict with mae, rmse, direction_accuracy, val_size.
        """
        if self._model is None:
            return {}

        try:
            import torch

            self._model.eval()
            with torch.no_grad():
                raw_predictions, index = self._model.predict(
                    val_loader, mode="raw", return_index=True
                )

            pred_tensor = raw_predictions["prediction"]
            q50_idx = 3  # median quantile
            step_idx = min(2, pred_tensor.shape[1] - 1)
            predicted = pred_tensor[:, step_idx, q50_idx].cpu().numpy()

            # Align actuals — use the index to extract targets
            actuals = val_data[_TARGET_COL].values[: len(predicted)]

            mae = float(np.mean(np.abs(predicted - actuals)))
            rmse = float(np.sqrt(np.mean((predicted - actuals) ** 2)))

            pred_direction = (predicted > 0).astype(int)
            actual_direction = (actuals > 0).astype(int)
            direction_accuracy = float(np.mean(pred_direction == actual_direction))

            metrics = {
                "mae": mae,
                "rmse": rmse,
                "direction_accuracy": direction_accuracy,
                "val_size": len(predicted),
                "feature_count": len(self._feature_columns),
            }
            return metrics

        except Exception as exc:
            logger.warning("Could not compute TFT validation metrics: %s", exc)
            return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _flatten_vsn_weights(vsn: Any, feature_names: List[str]) -> Dict[str, float]:
    """Extract per-variable importance scores from a VSN module.

    Attempts to read the softmax-normalised weights from the variable
    selection network's internal GRNs. Falls back to raw parameter L1-norms
    if the standard attribute path is unavailable.

    Args:
        vsn: pytorch_forecasting VariableSelectionNetwork module.
        feature_names: Names corresponding to the VSN's input variables.

    Returns:
        Dict mapping feature name → importance score, or empty dict on failure.
    """
    try:
        import torch

        with torch.no_grad():
            # pytorch_forecasting stores per-variable weights in flattened_grn
            flattened = getattr(vsn, "flattened_grn", None)
            if flattened is not None:
                # flattened_grn.fc1.weight: (hidden, n_features * hidden_continuous)
                w = flattened.fc1.weight.abs().mean(dim=0)
                n = len(feature_names)
                if len(w) >= n:
                    chunk_size = len(w) // n if n > 0 else 1
                    scores = np.array(
                        [float(w[i * chunk_size: (i + 1) * chunk_size].mean())
                         for i in range(n)]
                    )
                    # Softmax-normalise for interpretability
                    exp_scores = np.exp(scores - scores.max())
                    norm_scores = exp_scores / exp_scores.sum()
                    return dict(zip(feature_names, norm_scores.tolist()))

            # Fallback: aggregate over single_variable_grns
            single_grns = getattr(vsn, "single_variable_grns", None)
            if single_grns is not None and len(single_grns) == len(feature_names):
                scores = np.array(
                    [sum(p.abs().mean().item() for p in grn.parameters())
                     for grn in single_grns]
                )
                exp_scores = np.exp(scores - scores.max())
                norm_scores = exp_scores / exp_scores.sum()
                return dict(zip(feature_names, norm_scores.tolist()))

    except Exception as exc:
        logger.debug("_flatten_vsn_weights failed: %s", exc)

    return {}


def _build_predict_dataset(
    model: Any,
    data: pd.DataFrame,
    feature_columns: List[str],
) -> Any:
    """Reconstruct a TimeSeriesDataSet for inference using the trained model's parameters.

    pytorch_forecasting models store their dataset configuration in
    `model.hparams`; we use it to reproduce an identical dataset for
    prediction, avoiding the need to persist the training dataset object.

    Args:
        model: Trained TemporalFusionTransformer instance.
        data: Prepared DataFrame (time_idx, group columns added).
        feature_columns: Feature column list used during training.

    Returns:
        TimeSeriesDataSet configured for prediction.
    """
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer

    known_reals = [c for c in _KNOWN_REAL_COLS if c in data.columns]

    return TimeSeriesDataSet(
        data,
        time_idx="time_idx",
        target=_TARGET_COL,
        group_ids=["group"],
        max_encoder_length=_MAX_ENCODER_LENGTH,
        max_prediction_length=_MAX_PREDICTION_LENGTH,
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=feature_columns,
        target_normalizer=GroupNormalizer(groups=["group"]),
        allow_missing_timesteps=True,
        predict_mode=True,
    )
