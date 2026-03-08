from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from polymarket.utils import parse_ts, to_float, utc_now, utc_now_iso


class QuantumFoldLabelStore:
    def __init__(self, runtime_dir: str | Path, *, horizons: Iterable[int]) -> None:
        self.runtime_dir = Path(runtime_dir)
        self.pending_path = self.runtime_dir / "quantum_fold_pending_examples.json"
        self.examples_path = self.runtime_dir / "quantum_fold_examples.jsonl"
        self.labels_path = self.runtime_dir / "quantum_fold_labels.jsonl"
        self.horizons = sorted({int(value) for value in horizons if int(value) > 0})
        self.runtime_dir.mkdir(parents=True, exist_ok=True)

    def load_pending(self) -> Dict[str, Dict[str, Any]]:
        if not self.pending_path.exists():
            return {}
        try:
            return json.loads(self.pending_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def save_pending(self, payload: Dict[str, Dict[str, Any]]) -> None:
        self.pending_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def load_labels(self, limit: int | None = None) -> List[Dict[str, Any]]:
        if not self.labels_path.exists():
            return []
        rows: List[Dict[str, Any]] = []
        for line in self.labels_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
        return rows[-limit:] if limit is not None else rows

    def _append_jsonl(self, path: Path, row: Dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=str) + "\n")

    def track_examples(self, examples: Iterable[Dict[str, Any]]) -> Dict[str, int]:
        pending = self.load_pending()
        added = 0
        for example in examples:
            if not isinstance(example, dict):
                continue
            example_id = str(example.get("example_id") or "")
            if not example_id or example_id in pending:
                continue
            payload = {
                **example,
                "tracked_at": example.get("tracked_at") or utc_now_iso(),
                "settled_horizons": [],
                "final_settled": False,
            }
            pending[example_id] = payload
            self._append_jsonl(self.examples_path, payload)
            added += 1
        self.save_pending(pending)
        return {"tracked": added, "pending_total": len(pending)}

    def update_labels(self, quote_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        pending = self.load_pending()
        now = utc_now()
        completed: List[Dict[str, Any]] = []
        retained: Dict[str, Dict[str, Any]] = {}
        for example_id, example in pending.items():
            token_id = str(example.get("token_id") or "")
            current = dict(quote_map.get(token_id) or {})
            tracked_at = parse_ts(example.get("tracked_at"))
            if tracked_at is None:
                continue
            entry_midpoint = to_float(example.get("entry_midpoint"), 0.0)
            cost_buffer = to_float(example.get("cost_buffer"), 0.0)
            final_done = bool(example.get("final_settled"))
            settled_horizons = {int(value) for value in example.get("settled_horizons") or []}
            for horizon in self.horizons:
                if horizon in settled_horizons:
                    continue
                if (now - tracked_at).total_seconds() < horizon:
                    continue
                exit_midpoint = to_float(current.get("midpoint") or current.get("last_trade_price"), entry_midpoint)
                net_return = exit_midpoint - entry_midpoint - cost_buffer
                row = {
                    "label_id": f"{example_id}:{horizon}",
                    "example_id": example_id,
                    "ts": utc_now_iso(),
                    "token_id": token_id,
                    "market_slug": example.get("market_slug"),
                    "event_slug": example.get("event_slug"),
                    "horizon_seconds": horizon,
                    "horizon_label": f"{horizon}s",
                    "entry_midpoint": round(entry_midpoint, 6),
                    "exit_midpoint": round(exit_midpoint, 6),
                    "cost_buffer": round(cost_buffer, 6),
                    "net_return": round(net_return, 6),
                    "target": 1 if net_return > 0 else 0,
                    "features": dict(example.get("features") or {}),
                    "model_predictions": dict(example.get("model_predictions") or {}),
                }
                completed.append(row)
                self._append_jsonl(self.labels_path, row)
                settled_horizons.add(horizon)
            resolved = bool(current.get("resolved") or current.get("closed")) and current.get("resolution") is not None
            if resolved and not final_done:
                resolution = to_float(current.get("resolution"), 0.0)
                if resolution not in {0.0, 1.0}:
                    resolution = 1.0 if str(current.get("resolution")).strip().lower() in {"yes", "true", "winner"} else 0.0
                final_return = resolution - entry_midpoint - cost_buffer
                final_row = {
                    "label_id": f"{example_id}:final",
                    "example_id": example_id,
                    "ts": utc_now_iso(),
                    "token_id": token_id,
                    "market_slug": example.get("market_slug"),
                    "event_slug": example.get("event_slug"),
                    "horizon_seconds": None,
                    "horizon_label": "final",
                    "entry_midpoint": round(entry_midpoint, 6),
                    "exit_midpoint": round(resolution, 6),
                    "cost_buffer": round(cost_buffer, 6),
                    "net_return": round(final_return, 6),
                    "target": int(resolution >= 1.0),
                    "baseline_probability": round(entry_midpoint, 6),
                    "features": dict(example.get("features") or {}),
                    "model_predictions": dict(example.get("model_predictions") or {}),
                }
                completed.append(final_row)
                self._append_jsonl(self.labels_path, final_row)
                final_done = True
            if len(settled_horizons) < len(self.horizons) or not final_done:
                retained[example_id] = {
                    **example,
                    "settled_horizons": sorted(settled_horizons),
                    "final_settled": final_done,
                }
        self.save_pending(retained)
        return {
            "completed": len(completed),
            "pending_total": len(retained),
            "settled_labels": completed,
        }
