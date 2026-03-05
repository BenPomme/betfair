"""
Lightweight fallback contrarian model when XGBoost/TFT are unavailable.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


class ContrarianBaselineModel:
    """
    Rule-based probabilistic model with the same predict() schema as ML models.
    """

    def __init__(self, model_dir: str = "data/funding_models/contrarian_baseline"):
        self._model_dir = Path(model_dir)
        self._params: Dict[str, Any] = {
            "zscore_weight": 0.85,
            "rate_change_weight": 120.0,
            "base_conf": 0.55,
        }

    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        # Baseline keeps fixed parameters; returns minimal metrics-style payload.
        rows = int(len(df)) if df is not None else 0
        return {"rows": rows, "baseline": 1.0}

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        if features is None or features.empty:
            return pd.DataFrame(
                columns=[
                    "direction_prob",
                    "predicted_return_24h",
                    "predicted_return_72h",
                    "confidence",
                    "predicted_direction",
                ]
            )

        z = features.get("funding_zscore_30", pd.Series(np.zeros(len(features)))).fillna(0.0).astype(float)
        rc = features.get("rate_change_8h", pd.Series(np.zeros(len(features)))).fillna(0.0).astype(float)

        score = -(z * float(self._params["zscore_weight"])) - (rc * float(self._params["rate_change_weight"]))
        direction_prob = 1.0 / (1.0 + np.exp(-score))
        direction_prob = np.clip(direction_prob, 0.01, 0.99)
        confidence = np.clip(
            float(self._params["base_conf"]) + (np.abs(direction_prob - 0.5) * 0.9),
            0.5,
            0.99,
        )
        pred_ret_24h = score * 0.005
        pred_ret_72h = pred_ret_24h * 1.5
        predicted_direction = direction_prob >= 0.5

        return pd.DataFrame(
            {
                "direction_prob": direction_prob.astype(float),
                "predicted_return_24h": pred_ret_24h.astype(float),
                "predicted_return_72h": pred_ret_72h.astype(float),
                "confidence": confidence.astype(float),
                "predicted_direction": predicted_direction.astype(bool),
            },
            index=features.index,
        )

    def save(self, name: str = "contrarian_baseline") -> None:
        self._model_dir.mkdir(parents=True, exist_ok=True)
        payload = {"params": self._params}
        (self._model_dir / f"{name}_meta.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    def load(self, name: str = "contrarian_baseline") -> None:
        meta = self._model_dir / f"{name}_meta.json"
        if not meta.exists():
            return
        try:
            raw = json.loads(meta.read_text(encoding="utf-8"))
            params = raw.get("params", {})
            if isinstance(params, dict):
                self._params.update(params)
        except Exception:
            return
