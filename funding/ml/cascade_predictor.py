"""
4-layer Transformer encoder for liquidation cascade prediction.

Predicts two outputs from a 24-step (24h at 1h resolution) feature window:
  - Fragility probability : P(cascade within next 4h) — binary head
  - Severity              : predicted drawdown % — regression head

Architecture (_CascadeTransformerNet):
  - Linear projection:  n_features -> d_model=128
  - Learnable positional encoding: shape (1, window_size, d_model)
  - TransformerEncoder: 4 layers, nhead=8, dim_feedforward=256, dropout=0.15
  - Global average pooling over sequence dimension
  - Binary head:   Linear(d_model, 1) + sigmoid  -> fragility probability
  - Severity head: Linear(d_model, 1)             -> drawdown % (unbounded regression)

Training:
  - Focal loss (alpha=0.25, gamma=2.0) for the binary head to handle severe
    class imbalance (cascades are rare events, typically <2% of hours).
  - MSE loss for the severity head, restricted to positive-label rows only.

Usage:
    from funding.ml.cascade_predictor import CascadePredictor
    from funding.ml.cascade_features import build_cascade_features, label_cascade_events, get_cascade_feature_columns

    df   = build_cascade_features(["BTCUSDT", "ETHUSDT"])
    labels = label_cascade_events(df)
    df["cascade_label"] = labels
    feat_cols = get_cascade_feature_columns(df)

    predictor = CascadePredictor()
    metrics   = predictor.fit(df, feature_columns=feat_cols)

    prob, severity = predictor.predict_fragility(recent_features_df)
    predictor.save("cascade")
    predictor.load("cascade")
"""
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional PyTorch imports — wrapped so the module can be imported even when
# PyTorch is not installed.  Prediction will raise a RuntimeError at call time.
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    _TORCH_AVAILABLE = False
    torch = None       # type: ignore[assignment]
    nn = None          # type: ignore[assignment]
    Dataset = None     # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]

try:
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover
    _SKLEARN_AVAILABLE = False
    StandardScaler = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_DIR = Path("data/funding_models/cascade")


# ===========================================================================
# Internal PyTorch module factory
# ===========================================================================

def _build_cascade_net(window_size: int, n_features: int):
    """Factory: builds _CascadeTransformerNet only when torch is available.

    Returns an instance of the inner PyTorch module with the given dimensions.

    Args:
        window_size: Number of 1h input steps (e.g. 24).
        n_features:  Number of input features per step.

    Returns:
        Initialised _CascadeTransformerNet instance (nn.Module).

    Raises:
        RuntimeError: If PyTorch is not installed.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is not installed. Install it with: pip install torch"
        )

    class _CascadeTransformerNet(nn.Module):
        """
        4-layer Transformer encoder for cascade detection.

        Forward pass:
            x   : (batch, window_size, n_features) float tensor
            out : dict with keys
                    "prob"     : (batch, 1) sigmoid fragility probability
                    "severity" : (batch, 1) raw drawdown regression output
        """

        def __init__(
            self,
            n_features: int,
            window_size: int,
            d_model: int = 128,
            nhead: int = 8,
            dim_feedforward: int = 256,
            dropout: float = 0.15,
            num_layers: int = 4,
        ) -> None:
            super().__init__()

            self.d_model = d_model
            self.window_size = window_size

            # --- Input projection ---
            self.input_projection = nn.Linear(n_features, d_model)

            # --- Learnable positional encoding ---
            # Shape: (1, window_size, d_model) so it broadcasts over batch
            self.positional_encoding = nn.Parameter(
                torch.zeros(1, window_size, d_model)
            )
            nn.init.normal_(self.positional_encoding, mean=0.0, std=0.02)

            # --- Transformer encoder ---
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,   # input shape: (batch, seq, d_model)
                norm_first=True,    # Pre-norm (more stable for small datasets)
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
                enable_nested_tensor=False,
            )

            # --- Output heads ---
            # Binary head: probability of cascade in next 4h
            self.binary_head = nn.Linear(d_model, 1)

            # Severity head: predicted drawdown magnitude
            self.severity_head = nn.Linear(d_model, 1)

            # --- Weight initialisation ---
            self._init_weights()

        def _init_weights(self) -> None:
            """Initialise projection and head weights for stable training."""
            nn.init.xavier_uniform_(self.input_projection.weight)
            nn.init.zeros_(self.input_projection.bias)
            nn.init.xavier_uniform_(self.binary_head.weight)
            nn.init.zeros_(self.binary_head.bias)
            nn.init.xavier_uniform_(self.severity_head.weight)
            nn.init.zeros_(self.severity_head.bias)

        def forward(self, x: "torch.Tensor") -> Dict[str, "torch.Tensor"]:
            """
            Args:
                x: (batch, window_size, n_features) float32 tensor

            Returns:
                dict:
                    "prob"     : (batch, 1) — sigmoid probability [0, 1]
                    "severity" : (batch, 1) — drawdown regression (unbounded)
            """
            # Project features to model dimension: (batch, seq, d_model)
            x = self.input_projection(x)

            # Add learnable positional encoding
            x = x + self.positional_encoding

            # Transformer encoder: (batch, seq, d_model)
            x = self.transformer_encoder(x)

            # Global average pooling over sequence: (batch, d_model)
            x = x.mean(dim=1)

            # Dual heads
            prob = torch.sigmoid(self.binary_head(x))       # (batch, 1)
            severity = self.severity_head(x)                # (batch, 1)

            return {"prob": prob, "severity": severity}

    return _CascadeTransformerNet(n_features=n_features, window_size=window_size)


# ===========================================================================
# Focal loss
# ===========================================================================

def _focal_loss_fn(
    probs: "torch.Tensor",
    targets: "torch.Tensor",
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> "torch.Tensor":
    """Sigmoid focal loss for binary classification with severe class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where:
        p_t = p      if y = 1
        p_t = 1 - p  if y = 0
        alpha_t = alpha      if y = 1
        alpha_t = 1 - alpha  if y = 0

    Args:
        probs:   (batch, 1) tensor — predicted sigmoid probabilities.
        targets: (batch, 1) tensor — binary labels 0 or 1.
        alpha:   Weighting factor for positive class (rare cascades).
                 Higher alpha → more weight on positive examples.
        gamma:   Focusing parameter — down-weights easy negatives.
                 gamma=0 recovers standard binary cross-entropy.

    Returns:
        Scalar mean focal loss.
    """
    probs   = probs.clamp(1e-7, 1.0 - 1e-7)
    targets = targets.float()

    # Cross-entropy term
    bce = -(targets * torch.log(probs) + (1.0 - targets) * torch.log(1.0 - probs))

    # Focal weight: (1 - p_t)^gamma
    p_t   = probs * targets + (1.0 - probs) * (1.0 - targets)
    focal_weight = (1.0 - p_t) ** gamma

    # Alpha balancing
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)

    loss = alpha_t * focal_weight * bce
    return loss.mean()


# ===========================================================================
# Internal Dataset factory
# ===========================================================================

def _build_dataset_class():
    """Factory: returns the Dataset subclass only when torch is available."""
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed.")

    class _CascadeWindowDataset(Dataset):
        """
        Sliding-window dataset for the cascade transformer.

        Each sample:
            X        : (window_size, n_features) float32 tensor
            y_bin    : scalar float32 tensor — binary cascade label
            y_sev    : scalar float32 tensor — severity target (drawdown %)
                       NaN-filled as -1.0 for negative labels (masked in loss)
        """

        def __init__(
            self,
            X: np.ndarray,
            y_bin: np.ndarray,
            y_sev: np.ndarray,
            window_size: int,
        ) -> None:
            super().__init__()
            self.X     = torch.from_numpy(X.astype(np.float32))
            self.y_bin = torch.from_numpy(y_bin.astype(np.float32))
            self.y_sev = torch.from_numpy(y_sev.astype(np.float32))
            self.window_size = window_size
            self.n_samples = len(X) - window_size + 1

        def __len__(self) -> int:
            return self.n_samples

        def __getitem__(self, idx: int):
            end = idx + self.window_size
            window = self.X[idx:end]               # (window_size, n_features)
            label  = self.y_bin[end - 1]           # label at last step in window
            sev    = self.y_sev[end - 1]
            return window, label.unsqueeze(0), sev.unsqueeze(0)

    return _CascadeWindowDataset


# ===========================================================================
# Public CascadePredictor class
# ===========================================================================

class CascadePredictor:
    """
    4-layer Transformer encoder for liquidation cascade detection.

    Attributes:
        _window_size    : Number of 1h input steps (default 24).
        _n_features     : Number of input features (set at fit time).
        _model_dir      : Directory for model persistence artefacts.
        _model          : _CascadeTransformerNet (nn.Module) or None.
        _scaler         : Fitted StandardScaler or None.
        _feature_columns: Feature column names used during training.
        _metrics        : Training/validation metrics from the last fit().

    Training notes:
        - Binary head uses focal loss (alpha=0.25, gamma=2.0) to handle the
          extreme class imbalance typical of cascade data (<<5% positive rate).
        - Severity head uses MSE restricted to positive-label rows only.
        - StandardScaler is applied to all input features before windowing.
        - Early stopping on validation AUROC (or binary accuracy when AUROC
          cannot be computed due to single-class validation folds).
        - No GPU assumed — CPU-only for portability.

    Example:
        predictor = CascadePredictor(window_size=24)
        metrics   = predictor.fit(feature_df, feature_columns=feat_cols)
        prob, severity = predictor.predict_fragility(recent_df)
        predictor.save("cascade")
    """

    def __init__(
        self,
        n_features: Optional[int] = None,
        window_size: int = 24,
        model_dir: Optional[str] = None,
    ) -> None:
        """Initialise the predictor.

        Args:
            n_features:  Optional hint for the number of input features.
                         Set automatically at fit() time.
            window_size: Number of 1h historical steps in each input window.
                         Default: 24 (= 24h of lookback at 1h resolution).
            model_dir:   Directory for saving/loading artefacts.
                         Default: data/funding_models/cascade
        """
        self._window_size = window_size
        self._n_features  = n_features
        self._model_dir   = Path(model_dir or _DEFAULT_MODEL_DIR)
        self._model       = None          # _CascadeTransformerNet (nn.Module) or None
        self._scaler      = None          # StandardScaler or None
        self._feature_columns: List[str] = []
        self._metrics: Dict[str, Any]    = {}

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        epochs: int = 200,
        batch_size: int = 64,
        lr: float = 0.0005,
        patience: int = 20,
    ) -> Dict[str, Any]:
        """Train the cascade predictor on a labeled feature DataFrame.

        Args:
            df:              DataFrame containing a "cascade_label" column
                             (binary int 0/1) and numeric feature columns.
                             Optionally may contain a "cascade_severity"
                             column (forward drawdown %) for the severity head;
                             if absent, the severity head trains on zeros for
                             positive-label rows (still useful for thresholding).
            feature_columns: Explicit list of feature column names.  If None,
                             all numeric columns except known metadata/target
                             columns are used.
            epochs:          Maximum training epochs (early stopping may reduce).
            batch_size:      Mini-batch size for DataLoader.
            lr:              Adam learning rate.
            patience:        Early-stopping patience in epochs (monitors val loss).

        Returns:
            Dict with keys:
                "train_loss"        : final epoch training loss
                "val_loss"          : best validation loss
                "val_binary_accuracy": accuracy on validation set
                "val_positive_rate" : fraction of positive labels in val set
                "epochs_trained"    : actual epochs run
                "n_train"           : number of training rows
                "n_val"             : number of validation rows
                "n_features"        : number of input features used

        Raises:
            RuntimeError: If PyTorch or scikit-learn is not installed.
            ValueError:   If required columns are missing or data is insufficient.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is not installed. Install it with: pip install torch"
            )
        if not _SKLEARN_AVAILABLE:
            raise RuntimeError(
                "scikit-learn is not installed. Install it with: pip install scikit-learn"
            )

        if "cascade_label" not in df.columns:
            raise ValueError(
                "DataFrame must contain a 'cascade_label' column. "
                "Use label_cascade_events() to create it."
            )

        # --- Resolve feature columns ---
        if feature_columns is not None:
            self._feature_columns = list(feature_columns)
        else:
            _meta_and_target = {
                "cascade_label", "cascade_severity", "symbol",
            }
            self._feature_columns = [
                c for c in df.columns
                if c not in _meta_and_target and pd.api.types.is_numeric_dtype(df[c])
            ]

        if not self._feature_columns:
            raise ValueError("No numeric feature columns found in DataFrame.")

        # --- Build clean arrays ---
        cols_needed = self._feature_columns + ["cascade_label"]
        clean = df[cols_needed].copy()
        clean = clean.replace([np.inf, -np.inf], np.nan)
        clean = clean.dropna(subset=self._feature_columns)
        # Fill remaining NaN in label with 0
        clean["cascade_label"] = clean["cascade_label"].fillna(0).astype(int)

        if len(clean) < self._window_size + 2:
            raise ValueError(
                f"Insufficient data after cleaning: {len(clean)} rows, "
                f"need at least {self._window_size + 2}."
            )

        X_raw = clean[self._feature_columns].values

        # Binary labels
        y_bin = clean["cascade_label"].values.astype(np.float32)

        # Severity labels (optional)
        if "cascade_severity" in df.columns:
            y_sev_raw = df["cascade_severity"].reindex(clean.index).fillna(0.0).values.astype(np.float32)
        else:
            # Default: for positive rows, severity = 0 (unknown); masked by loss
            y_sev_raw = np.zeros(len(clean), dtype=np.float32)

        # --- Fit and apply StandardScaler ---
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_raw)
        self._n_features = X_scaled.shape[1]

        # --- Temporal train/val split (last 20% for validation) ---
        n_total = len(X_scaled)
        n_val   = max(self._window_size, int(n_total * 0.2))
        n_train = n_total - n_val

        if n_train < self._window_size:
            raise ValueError(
                f"Not enough training rows: {n_train} (need >= {self._window_size})."
            )

        X_train, X_val = X_scaled[:n_train], X_scaled[n_train:]
        y_bin_train, y_bin_val = y_bin[:n_train], y_bin[n_train:]
        y_sev_train, y_sev_val = y_sev_raw[:n_train], y_sev_raw[n_train:]

        logger.info(
            "CascadePredictor.fit: features=%d, window=%d, "
            "train_rows=%d, val_rows=%d, epochs=%d, "
            "pos_rate_train=%.3f",
            self._n_features, self._window_size, n_train, len(X_val), epochs,
            float(y_bin_train.mean()),
        )

        # --- Build model ---
        self._model = _build_cascade_net(self._window_size, self._n_features)

        # --- Build datasets / loaders ---
        _CascadeWindowDataset = _build_dataset_class()

        train_ds = _CascadeWindowDataset(X_train, y_bin_train, y_sev_train, self._window_size)
        val_ds   = _CascadeWindowDataset(X_val,   y_bin_val,   y_sev_val,   self._window_size)

        if len(train_ds) == 0:
            raise ValueError(
                f"Training dataset is empty after windowing "
                f"(train_rows={n_train}, window_size={self._window_size})."
            )

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, drop_last=False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, drop_last=False,
        )

        # --- Optimizer and scheduler ---
        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=patience // 2,
        )

        device = torch.device("cpu")
        self._model.to(device)

        # --- Training loop ---
        best_val_loss = float("inf")
        best_state_dict = None
        epochs_no_improve = 0
        epochs_trained = 0
        last_train_loss = float("inf")

        for epoch in range(1, epochs + 1):
            self._model.train()
            total_train_loss = 0.0
            n_batches = 0

            for X_batch, y_bin_batch, y_sev_batch in train_loader:
                X_batch     = X_batch.to(device)
                y_bin_batch = y_bin_batch.to(device)
                y_sev_batch = y_sev_batch.to(device)

                optimizer.zero_grad()
                out = self._model(X_batch)

                # Focal loss on binary head
                loss_bin = _focal_loss_fn(out["prob"], y_bin_batch)

                # MSE on severity head, restricted to positive-label rows
                pos_mask = (y_bin_batch > 0.5).squeeze(1)
                if pos_mask.any():
                    pred_sev = out["severity"][pos_mask]
                    true_sev = y_sev_batch[pos_mask]
                    loss_sev = torch.nn.functional.mse_loss(pred_sev, true_sev)
                else:
                    loss_sev = torch.tensor(0.0, device=device)

                loss = loss_bin + 0.1 * loss_sev   # severity is auxiliary
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                optimizer.step()

                total_train_loss += loss.item()
                n_batches += 1

            avg_train_loss = total_train_loss / max(n_batches, 1)
            last_train_loss = avg_train_loss

            # Validation
            val_loss, val_bin_acc = self._evaluate_loader(val_loader, device)
            scheduler.step(val_loss)

            epochs_trained = epoch

            if epoch % 20 == 0 or epoch == 1:
                logger.debug(
                    "Epoch %d/%d — train_loss=%.5f, val_loss=%.5f, val_acc=%.4f",
                    epoch, epochs, avg_train_loss, val_loss, val_bin_acc,
                )

            # Early stopping on val loss
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state_dict = {k: v.clone() for k, v in self._model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info(
                        "Early stopping at epoch %d (patience=%d, best_val_loss=%.5f)",
                        epoch, patience, best_val_loss,
                    )
                    break

        # Restore best weights
        if best_state_dict is not None:
            self._model.load_state_dict(best_state_dict)
        self._model.eval()

        # Final validation metrics
        _, final_val_acc = self._evaluate_loader(val_loader, device)
        val_positive_rate = float(y_bin_val.mean()) if len(y_bin_val) > 0 else 0.0

        self._metrics = {
            "train_loss":          float(last_train_loss),
            "val_loss":            float(best_val_loss),
            "val_binary_accuracy": float(final_val_acc),
            "val_positive_rate":   float(val_positive_rate),
            "epochs_trained":      epochs_trained,
            "n_train":             n_train,
            "n_val":               len(X_val),
            "n_features":          self._n_features,
            "window_size":         self._window_size,
        }

        logger.info(
            "CascadePredictor fit complete: val_acc=%.4f, best_val_loss=%.5f, "
            "epochs=%d/%d",
            final_val_acc, best_val_loss, epochs_trained, epochs,
        )
        return self._metrics

    # -----------------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------------

    def predict_fragility(
        self,
        features: pd.DataFrame,
    ) -> Tuple[float, float]:
        """Predict cascade fragility from recent feature history.

        Uses the most recent `window_size` rows of *features* as the input
        window.  Scales features using the fitted StandardScaler.

        Args:
            features: DataFrame with at least `window_size` rows containing
                      all feature columns used during training (in any order).
                      Extra columns are ignored.

        Returns:
            Tuple (probability, severity):
                probability : float in [0, 1] — P(cascade within next 4h).
                              Higher values indicate more fragile market state.
                severity    : float — predicted drawdown magnitude (%).
                              Positive value → predicted downward move.
                              Only reliable when probability is high.

        Raises:
            RuntimeError: If PyTorch is not installed or model not trained/loaded.
            ValueError:   If *features* has fewer than `window_size` rows or
                          missing required columns.
        """
        out = self._forward(features)
        probability = float(out["prob"].item())
        severity    = float(out["severity"].item())
        return probability, severity

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def save(self, name: str = "cascade") -> None:
        """Save model state_dict, scaler, and metadata to `_model_dir`.

        Files written:
            {name}_state.pt    — PyTorch state_dict
            {name}_scaler.pkl  — Fitted StandardScaler (pickle)
            {name}_meta.json   — Hyperparameters and metrics

        Args:
            name: Base filename (no extension).  Defaults to "cascade".

        Raises:
            RuntimeError: If model has not been trained yet (call fit() first).
        """
        if self._model is None:
            raise RuntimeError(
                "Model has not been trained yet. Call fit() first."
            )
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is not installed. Install it with: pip install torch"
            )

        self._model_dir.mkdir(parents=True, exist_ok=True)

        # State dict
        torch.save(
            self._model.state_dict(),
            self._model_dir / f"{name}_state.pt",
        )

        # Scaler
        with open(self._model_dir / f"{name}_scaler.pkl", "wb") as fh:
            pickle.dump(self._scaler, fh)

        # Metadata
        meta = {
            "name":            name,
            "window_size":     self._window_size,
            "n_features":      self._n_features,
            "feature_columns": self._feature_columns,
            "metrics":         self._metrics,
        }
        with open(self._model_dir / f"{name}_meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, default=str)

        logger.info(
            "CascadePredictor saved: name='%s', dir=%s", name, self._model_dir
        )

    def load(self, name: str = "cascade") -> None:
        """Load model state_dict, scaler, and metadata from `_model_dir`.

        Args:
            name: Base filename (no extension) used when saving.

        Raises:
            RuntimeError:      If PyTorch or scikit-learn is not installed.
            FileNotFoundError: If any required artefact file is missing.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is not installed. Install it with: pip install torch"
            )
        if not _SKLEARN_AVAILABLE:
            raise RuntimeError(
                "scikit-learn is not installed. Install it with: pip install scikit-learn"
            )

        # Load metadata
        meta_path = self._model_dir / f"{name}_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)

        self._window_size     = meta["window_size"]
        self._n_features      = meta["n_features"]
        self._feature_columns = meta["feature_columns"]
        self._metrics         = meta.get("metrics", {})

        # Load scaler
        scaler_path = self._model_dir / f"{name}_scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

        with open(scaler_path, "rb") as fh:
            self._scaler = pickle.load(fh)

        # Rebuild and load model weights
        self._model = _build_cascade_net(self._window_size, self._n_features)
        state_path = self._model_dir / f"{name}_state.pt"
        if not state_path.exists():
            raise FileNotFoundError(f"State dict file not found: {state_path}")

        state_dict = torch.load(state_path, map_location="cpu")
        self._model.load_state_dict(state_dict)
        self._model.eval()

        logger.info(
            "CascadePredictor loaded: name='%s', features=%d, window=%d",
            name, self._n_features, self._window_size,
        )

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def metrics(self) -> Dict[str, Any]:
        """Training/validation metrics from the last call to fit().

        Returns a copy so the caller cannot mutate internal state.
        """
        return dict(self._metrics)

    @property
    def window_size(self) -> int:
        """Number of 1h steps in each input window."""
        return self._window_size

    @property
    def n_features(self) -> Optional[int]:
        """Number of input features (set at fit/load time, None before)."""
        return self._n_features

    @property
    def feature_columns(self) -> List[str]:
        """Feature column names used during training (in training order)."""
        return list(self._feature_columns)

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _forward(self, features: pd.DataFrame) -> Dict[str, "torch.Tensor"]:
        """Scale *features*, take last `window_size` rows, run a forward pass.

        Args:
            features: DataFrame with at least window_size rows and all
                      feature columns used during training.

        Returns:
            Dict with keys "prob" and "severity", each a (1, 1) tensor.

        Raises:
            RuntimeError: If PyTorch is not installed or model not trained/loaded.
            ValueError:   If features has too few rows or missing columns.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is not installed. Install it with: pip install torch"
            )
        if self._model is None or self._scaler is None:
            raise RuntimeError(
                "Model has not been trained or loaded. "
                "Call fit() or load() first."
            )
        if len(features) < self._window_size:
            raise ValueError(
                f"Need at least {self._window_size} rows for prediction, "
                f"got {len(features)}."
            )

        missing = [c for c in self._feature_columns if c not in features.columns]
        if missing:
            raise ValueError(
                f"Missing feature columns in prediction DataFrame: {missing}"
            )

        # Take last window_size rows, in training column order
        X_raw = (
            features[self._feature_columns]
            .tail(self._window_size)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .values
        )
        X_scaled = self._scaler.transform(X_raw)               # (window_size, n_features)
        X_tensor = torch.from_numpy(X_scaled.astype(np.float32)).unsqueeze(0)  # (1, window, feat)

        self._model.eval()
        with torch.no_grad():
            out = self._model(X_tensor)  # dict of (1, 1) tensors

        return out

    def _evaluate_loader(
        self,
        loader: "DataLoader",
        device: "torch.device",
    ) -> Tuple[float, float]:
        """Evaluate model on a DataLoader.

        Returns:
            (avg_loss: float, binary_accuracy: float)
        """
        self._model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        n_batches = 0

        with torch.no_grad():
            for X_batch, y_bin_batch, y_sev_batch in loader:
                X_batch     = X_batch.to(device)
                y_bin_batch = y_bin_batch.to(device)
                y_sev_batch = y_sev_batch.to(device)

                out = self._model(X_batch)

                loss_bin = _focal_loss_fn(out["prob"], y_bin_batch)

                pos_mask = (y_bin_batch > 0.5).squeeze(1)
                if pos_mask.any():
                    loss_sev = torch.nn.functional.mse_loss(
                        out["severity"][pos_mask], y_sev_batch[pos_mask]
                    )
                else:
                    loss_sev = torch.tensor(0.0, device=device)

                loss = loss_bin + 0.1 * loss_sev
                total_loss += loss.item()
                n_batches += 1

                # Binary accuracy: threshold at 0.5
                preds = (out["prob"] >= 0.5).float()
                correct += (preds == y_bin_batch).sum().item()
                total += y_bin_batch.numel()

        avg_loss = total_loss / max(n_batches, 1)
        accuracy = correct / total if total > 0 else 0.0
        return avg_loss, accuracy
