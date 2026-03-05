"""
Volatility regime classifier using a small 2-layer Transformer encoder.

Classifies 8h funding periods into one of 4 volatility regimes:
  0 — low     (bottom 25% realized vol)
  1 — medium  (25–50%)
  2 — high    (50–75%)
  3 — crisis  (top 25%)

Architecture:
  - Linear projection: n_features -> d_model=64
  - Learnable positional encoding
  - TransformerEncoder: 2 layers, nhead=4, dim_feedforward=128, dropout=0.1
  - Global average pooling over sequence dimension
  - Classification head: Linear(d_model, n_states)

Input shape per sample: (window_size, n_features)
Default window: 9 × 8h periods = 3 days of lookback.

Usage:
    from funding.ml.regime_transformer import RegimeTransformer

    clf = RegimeTransformer(window_size=9, n_states=4)
    metrics = clf.fit(df, feature_columns=feature_cols, epochs=100)
    regime = clf.predict_regime(recent_df)
    proba  = clf.predict_regime_proba(recent_df)
    clf.save("regime_transformer")
    clf.load("regime_transformer")
"""
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional PyTorch imports — wrapped so the module can be imported even if
# PyTorch is not installed (prediction will raise at call time).
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    Dataset = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]

from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_DIR = Path("data/funding_models/regime")


# ===========================================================================
# Internal PyTorch module
# ===========================================================================

def _build_net(window_size: int, n_features: int, n_states: int):
    """Factory: builds _RegimeTransformerNet only when torch is available."""
    if not _TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is not installed. Install it with: pip install torch"
        )

    class _RegimeTransformerNet(nn.Module):
        """
        2-layer Transformer encoder for volatility regime classification.

        Forward pass:
            x  : (batch, window_size, n_features)
            out: (batch, n_states)  — raw logits (no softmax)
        """

        def __init__(
            self,
            n_features: int,
            window_size: int,
            n_states: int,
            d_model: int = 64,
            nhead: int = 4,
            dim_feedforward: int = 128,
            dropout: float = 0.1,
            num_layers: int = 2,
        ) -> None:
            super().__init__()

            self.d_model = d_model
            self.window_size = window_size

            # Project raw features into model dimension
            self.input_projection = nn.Linear(n_features, d_model)

            # Learnable positional encoding: shape (1, window_size, d_model)
            self.positional_encoding = nn.Parameter(
                torch.zeros(1, window_size, d_model)
            )
            nn.init.normal_(self.positional_encoding, mean=0.0, std=0.02)

            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,  # (batch, seq, d_model)
                norm_first=False,
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
                enable_nested_tensor=False,
            )

            # Classification head (applied after global average pooling)
            self.classifier = nn.Linear(d_model, n_states)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """
            Args:
                x: (batch, window_size, n_features) float tensor

            Returns:
                logits: (batch, n_states)
            """
            # Project to model dimension: (batch, seq, d_model)
            x = self.input_projection(x)

            # Add positional encoding
            x = x + self.positional_encoding

            # Transformer encoder: (batch, seq, d_model)
            x = self.transformer_encoder(x)

            # Global average pooling over sequence dimension: (batch, d_model)
            x = x.mean(dim=1)

            # Classification logits: (batch, n_states)
            logits = self.classifier(x)
            return logits

    return _RegimeTransformerNet(
        n_features=n_features,
        window_size=window_size,
        n_states=n_states,
    )


# ===========================================================================
# Internal Dataset
# ===========================================================================

def _build_dataset_class():
    """Factory: returns the Dataset subclass only when torch is available."""
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed.")

    class _WindowDataset(Dataset):
        """
        Sliding-window dataset for the transformer.

        Each sample is:
            X: (window_size, n_features) float32 tensor
            y: scalar int64 tensor — regime label at the last timestep
        """

        def __init__(
            self,
            X: np.ndarray,
            y: np.ndarray,
            window_size: int,
        ) -> None:
            super().__init__()
            self.X = torch.from_numpy(X.astype(np.float32))
            self.y = torch.from_numpy(y.astype(np.int64))
            self.window_size = window_size
            # Number of valid starting positions
            self.n_samples = len(X) - window_size + 1

        def __len__(self) -> int:
            return self.n_samples

        def __getitem__(self, idx: int):
            window = self.X[idx : idx + self.window_size]  # (window_size, n_features)
            label = self.y[idx + self.window_size - 1]     # label at last timestep
            return window, label

    return _WindowDataset


# ===========================================================================
# Public class
# ===========================================================================

class RegimeTransformer:
    """
    Small 2-layer Transformer encoder for volatility regime classification.

    Attributes:
        _window_size     : Number of 8h periods in each input window (default 9).
        _n_states        : Number of regime classes (default 4).
        _n_features      : Number of input features (set during fit).
        _model_dir       : Directory for saving/loading model artifacts.
        _model           : Underlying _RegimeTransformerNet (nn.Module) or None.
        _scaler          : Fitted StandardScaler or None.
        _feature_columns : Feature column names used during training.
        _metrics         : Dict of training/validation metrics from last fit().
    """

    def __init__(
        self,
        n_features: Optional[int] = None,
        window_size: int = 9,
        n_states: int = 4,
        model_dir: Optional[str] = None,
    ) -> None:
        self._window_size = window_size
        self._n_states = n_states
        self._n_features = n_features
        self._model_dir = Path(model_dir or _DEFAULT_MODEL_DIR)
        self._model = None          # PyTorch nn.Module
        self._scaler = None         # StandardScaler
        self._feature_columns: List[str] = []
        self._metrics: Dict[str, Any] = {}

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 0.001,
        patience: int = 10,
    ) -> Dict[str, Any]:
        """
        Train the transformer on a labeled DataFrame.

        Args:
            df             : DataFrame containing a "regime_label" column and
                             feature columns (numeric, no NaN after dropna).
            feature_columns: Explicit list of feature column names.  If None,
                             all numeric columns except "regime_label" and
                             "symbol" are used.
            epochs         : Maximum training epochs.
            batch_size     : Mini-batch size for DataLoader.
            lr             : Adam learning rate.
            patience       : Early-stopping patience (epochs on val accuracy).

        Returns:
            Dict with keys:
                "train_loss", "val_loss", "val_accuracy",
                "per_class_accuracy", "confusion_summary",
                "epochs_trained", "n_train", "n_val", "n_features"

        Raises:
            RuntimeError : If PyTorch is not installed.
            ValueError   : If "regime_label" column is missing or data is
                           insufficient.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is not installed. Install it with: pip install torch"
            )

        if "regime_label" not in df.columns:
            raise ValueError(
                "DataFrame must contain a 'regime_label' column. "
                "Use regime_features.assign_regime_labels() to create it."
            )

        # Resolve feature columns
        if feature_columns is not None:
            self._feature_columns = list(feature_columns)
        else:
            exclude = {"regime_label", "symbol"}
            self._feature_columns = [
                c for c in df.columns
                if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
            ]

        if not self._feature_columns:
            raise ValueError("No numeric feature columns found in DataFrame.")

        # Drop rows with NaN in features or label
        cols_needed = self._feature_columns + ["regime_label"]
        clean = df[cols_needed].dropna()

        if len(clean) < self._window_size + 2:
            raise ValueError(
                f"Insufficient data after dropna: {len(clean)} rows, "
                f"need at least {self._window_size + 2}."
            )

        labels = clean["regime_label"].values.astype(int)

        # Validate label range
        unique_labels = np.unique(labels)
        if unique_labels.max() >= self._n_states or unique_labels.min() < 0:
            raise ValueError(
                f"regime_label values must be in [0, {self._n_states - 1}]. "
                f"Found: {unique_labels}"
            )

        # Fit and apply StandardScaler
        X_raw = clean[self._feature_columns].values
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_raw)

        self._n_features = X_scaled.shape[1]

        # Train / validation split — last 20% for validation (temporal order)
        n_total = len(X_scaled)
        n_val = max(self._window_size, int(n_total * 0.2))
        n_train = n_total - n_val

        if n_train < self._window_size:
            raise ValueError(
                f"Not enough training rows: {n_train} (need >= {self._window_size})."
            )

        X_train, y_train = X_scaled[:n_train], labels[:n_train]
        X_val, y_val = X_scaled[n_train:], labels[n_train:]

        logger.info(
            "RegimeTransformer.fit: features=%d, window=%d, "
            "train_rows=%d, val_rows=%d, epochs=%d",
            self._n_features, self._window_size, n_train, len(X_val), epochs,
        )

        # Build model
        self._model = _build_net(self._window_size, self._n_features, self._n_states)

        # Build datasets / loaders
        _WindowDataset = _build_dataset_class()
        train_ds = _WindowDataset(X_train, y_train, self._window_size)
        val_ds = _WindowDataset(X_val, y_val, self._window_size)

        if len(train_ds) == 0:
            raise ValueError(
                f"Training dataset is empty after windowing "
                f"(train_rows={n_train}, window_size={self._window_size})."
            )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

        # Optimizer, scheduler, loss
        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",       # maximize val_accuracy
            factor=0.5,
            patience=patience // 2,
        )
        criterion = nn.CrossEntropyLoss()

        device = torch.device("cpu")  # CPU-only for portability
        self._model.to(device)

        best_val_acc = -1.0
        best_state_dict = None
        epochs_no_improve = 0
        epochs_trained = 0
        last_train_loss = float("inf")
        last_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            # --- Training pass ---
            self._model.train()
            total_train_loss = 0.0
            n_train_batches = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                logits = self._model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                optimizer.step()

                total_train_loss += loss.item()
                n_train_batches += 1

            avg_train_loss = total_train_loss / max(n_train_batches, 1)

            # --- Validation pass ---
            val_acc, avg_val_loss = self._evaluate_loader(val_loader, criterion, device)
            scheduler.step(val_acc)

            epochs_trained = epoch
            last_train_loss = avg_train_loss
            last_val_loss = avg_val_loss

            if epoch % 10 == 0 or epoch == 1:
                logger.debug(
                    "Epoch %d/%d — train_loss=%.4f, val_loss=%.4f, val_acc=%.4f",
                    epoch, epochs, avg_train_loss, avg_val_loss, val_acc,
                )

            # Early stopping
            if val_acc > best_val_acc + 1e-4:
                best_val_acc = val_acc
                best_state_dict = {k: v.clone() for k, v in self._model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info(
                        "Early stopping at epoch %d (patience=%d, best_val_acc=%.4f)",
                        epoch, patience, best_val_acc,
                    )
                    break

        # Restore best weights
        if best_state_dict is not None:
            self._model.load_state_dict(best_state_dict)

        # Compute detailed val metrics with best model
        per_class_acc, confusion = self._compute_detailed_metrics(
            val_loader, device
        )

        self._metrics = {
            "train_loss": float(last_train_loss),
            "val_loss": float(last_val_loss),
            "val_accuracy": float(best_val_acc),
            "per_class_accuracy": per_class_acc,
            "confusion_summary": confusion,
            "epochs_trained": epochs_trained,
            "n_train": n_train,
            "n_val": len(X_val),
            "n_features": self._n_features,
        }

        logger.info(
            "RegimeTransformer fit complete: val_acc=%.4f, epochs=%d/%d",
            best_val_acc, epochs_trained, epochs,
        )
        return self._metrics

    # -----------------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------------

    def predict_regime(self, features: pd.DataFrame) -> int:
        """
        Predict the current volatility regime from recent feature history.

        Args:
            features: DataFrame with at least `window_size` rows, containing
                      all feature columns used during training.  The most
                      recent `window_size` rows are used.

        Returns:
            Predicted regime as an integer in [0, n_states - 1].

        Raises:
            RuntimeError : If PyTorch is not installed or model not trained.
            ValueError   : If insufficient rows are provided.
        """
        logits = self._forward(features)
        return int(torch.argmax(logits, dim=-1).item())

    def predict_regime_proba(self, features: pd.DataFrame) -> np.ndarray:
        """
        Return softmax class probabilities for each volatility regime.

        Args:
            features: DataFrame with at least `window_size` rows.

        Returns:
            1-D numpy array of shape (n_states,) with probabilities summing to 1.

        Raises:
            RuntimeError : If PyTorch is not installed or model not trained.
            ValueError   : If insufficient rows are provided.
        """
        logits = self._forward(features)
        proba = torch.softmax(logits, dim=-1).squeeze(0).detach().numpy()
        return proba

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def save(self, name: str = "regime_transformer") -> None:
        """
        Save model state_dict, scaler, and metadata to `_model_dir`.

        Files written:
            {name}_state.pt    — PyTorch state_dict
            {name}_scaler.pkl  — Fitted StandardScaler (pickle)
            {name}_meta.json   — Hyperparameters and metrics

        Args:
            name: Base filename (no extension).

        Raises:
            RuntimeError: If model has not been trained yet.
        """
        if self._model is None:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")

        self._model_dir.mkdir(parents=True, exist_ok=True)

        # Save state dict
        torch.save(
            self._model.state_dict(),
            self._model_dir / f"{name}_state.pt",
        )

        # Save scaler
        with open(self._model_dir / f"{name}_scaler.pkl", "wb") as fh:
            pickle.dump(self._scaler, fh)

        # Save metadata
        meta = {
            "name": name,
            "window_size": self._window_size,
            "n_states": self._n_states,
            "n_features": self._n_features,
            "feature_columns": self._feature_columns,
            "metrics": self._metrics,
        }
        with open(self._model_dir / f"{name}_meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, default=str)

        logger.info(
            "RegimeTransformer saved: name='%s', dir=%s", name, self._model_dir
        )

    def load(self, name: str = "regime_transformer") -> None:
        """
        Load model state_dict, scaler, and metadata from `_model_dir`.

        Args:
            name: Base filename (no extension) used when saving.

        Raises:
            RuntimeError : If PyTorch is not installed.
            FileNotFoundError: If any required artifact file is missing.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is not installed. Install it with: pip install torch"
            )

        # Load metadata first
        meta_path = self._model_dir / f"{name}_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)

        self._window_size = meta["window_size"]
        self._n_states = meta["n_states"]
        self._n_features = meta["n_features"]
        self._feature_columns = meta["feature_columns"]
        self._metrics = meta.get("metrics", {})

        # Load scaler
        scaler_path = self._model_dir / f"{name}_scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

        with open(scaler_path, "rb") as fh:
            self._scaler = pickle.load(fh)

        # Rebuild and load model weights
        self._model = _build_net(
            self._window_size, self._n_features, self._n_states
        )
        state_path = self._model_dir / f"{name}_state.pt"
        if not state_path.exists():
            raise FileNotFoundError(f"State dict file not found: {state_path}")

        state_dict = torch.load(state_path, map_location="cpu")
        self._model.load_state_dict(state_dict)
        self._model.eval()

        logger.info(
            "RegimeTransformer loaded: name='%s', features=%d, window=%d",
            name, self._n_features, self._window_size,
        )

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def metrics(self) -> Dict[str, Any]:
        """Training/validation metrics from the last call to fit()."""
        return dict(self._metrics)

    @property
    def n_states(self) -> int:
        """Number of regime classes."""
        return self._n_states

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _forward(self, features: pd.DataFrame) -> "torch.Tensor":
        """
        Scale features, take last `window_size` rows, run a forward pass.

        Returns:
            Logits tensor of shape (1, n_states).
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

        # Select feature columns (training order preserved)
        missing = [c for c in self._feature_columns if c not in features.columns]
        if missing:
            raise ValueError(
                f"Missing feature columns in prediction DataFrame: {missing}"
            )

        X_raw = features[self._feature_columns].tail(self._window_size).values
        X_scaled = self._scaler.transform(X_raw)               # (window_size, n_features)
        X_tensor = torch.from_numpy(X_scaled.astype(np.float32)).unsqueeze(0)  # (1, window, feat)

        self._model.eval()
        with torch.no_grad():
            logits = self._model(X_tensor)  # (1, n_states)
        return logits

    def _evaluate_loader(
        self,
        loader: "DataLoader",
        criterion: "nn.Module",
        device: "torch.device",
    ):
        """
        Evaluate model on a DataLoader.

        Returns:
            (accuracy: float, avg_loss: float)
        """
        self._model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        n_batches = 0

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                logits = self._model(X_batch)
                loss = criterion(logits, y_batch)
                total_loss += loss.item()
                n_batches += 1
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == y_batch).sum().item()
                total += len(y_batch)

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / max(n_batches, 1)
        return accuracy, avg_loss

    def _compute_detailed_metrics(
        self,
        loader: "DataLoader",
        device: "torch.device",
    ) -> tuple:
        """
        Compute per-class accuracy and a simplified confusion matrix summary.

        Returns:
            per_class_acc (Dict[int, float]) : accuracy per regime class
            confusion (Dict[str, int])       : flat confusion matrix
                keys like "true_0_pred_1" -> count
        """
        self._model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(device)
                logits = self._model(X_batch)
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(y_batch.numpy().tolist())

        if not all_labels:
            return {}, {}

        all_preds = np.array(all_preds, dtype=int)
        all_labels = np.array(all_labels, dtype=int)

        # Per-class accuracy
        per_class_acc: Dict[int, float] = {}
        for cls in range(self._n_states):
            mask = all_labels == cls
            if mask.sum() > 0:
                per_class_acc[cls] = float((all_preds[mask] == cls).mean())
            else:
                per_class_acc[cls] = float("nan")

        # Confusion matrix as flat dict
        confusion: Dict[str, int] = {}
        for true_cls in range(self._n_states):
            for pred_cls in range(self._n_states):
                key = f"true_{true_cls}_pred_{pred_cls}"
                mask = all_labels == true_cls
                count = int((all_preds[mask] == pred_cls).sum()) if mask.sum() > 0 else 0
                confusion[key] = count

        return per_class_acc, confusion
