from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import json

import config
from portfolio.types import ModelShadowAccount, StrategyAccount


class PortfolioStateStore:
    def __init__(self, portfolio_id: str, root: Optional[str] = None):
        self.portfolio_id = portfolio_id
        base_root = Path(root or getattr(config, "PORTFOLIO_STATE_ROOT", "data/portfolios"))
        self.base_dir = base_root / portfolio_id
        self.models_dir = self.base_dir / "models"
        self.runtime_dir = self.base_dir / "runtime"
        self.account_path = self.base_dir / "account.json"
        self.balance_history_path = self.base_dir / "balance_history.jsonl"
        self.trades_path = self.base_dir / "trades.jsonl"
        self.events_path = self.base_dir / "events.jsonl"
        self.readiness_path = self.base_dir / "readiness.json"
        self.config_snapshot_path = self.base_dir / "config_snapshot.json"
        self.heartbeat_path = self.base_dir / "heartbeat.json"
        self.state_path = self.base_dir / "state.json"
        self.pid_path = self.base_dir / "runner.pid"
        self.stop_signal_path = self.base_dir / "stop.signal"
        self.ensure()

    def ensure(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)

    def write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def read_json(self, path: Path, default: Any = None) -> Any:
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default

    def write_jsonl(self, path: Path, rows: Iterable[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, default=str) + "\n")

    def read_jsonl(self, path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        items: List[Dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
        if limit is not None:
            return items[-limit:]
        return items

    def write_account(self, account: StrategyAccount) -> None:
        self.write_json(self.account_path, account.to_dict())

    def read_account(self) -> Optional[StrategyAccount]:
        return StrategyAccount.from_dict(self.read_json(self.account_path))

    def write_balance_history(self, rows: Iterable[Dict[str, Any]]) -> None:
        self.write_jsonl(self.balance_history_path, rows)

    def read_balance_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        return self.read_jsonl(self.balance_history_path, limit=limit)

    def write_trades(self, rows: Iterable[Dict[str, Any]]) -> None:
        self.write_jsonl(self.trades_path, rows)

    def read_trades(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        return self.read_jsonl(self.trades_path, limit=limit)

    def write_events(self, rows: Iterable[Dict[str, Any]]) -> None:
        self.write_jsonl(self.events_path, rows)

    def read_events(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        return self.read_jsonl(self.events_path, limit=limit)

    def write_readiness(self, payload: Dict[str, Any]) -> None:
        self.write_json(self.readiness_path, payload)

    def read_readiness(self) -> Dict[str, Any]:
        return self.read_json(self.readiness_path, default={}) or {}

    def write_config_snapshot(self, payload: Dict[str, Any]) -> None:
        self.write_json(self.config_snapshot_path, payload)

    def read_config_snapshot(self) -> Dict[str, Any]:
        return self.read_json(self.config_snapshot_path, default={}) or {}

    def write_heartbeat(self, payload: Dict[str, Any]) -> None:
        self.write_json(self.heartbeat_path, payload)

    def read_heartbeat(self) -> Dict[str, Any]:
        return self.read_json(self.heartbeat_path, default={}) or {}

    def write_state(self, payload: Dict[str, Any]) -> None:
        self.write_json(self.state_path, payload)

    def read_state(self) -> Dict[str, Any]:
        return self.read_json(self.state_path, default={}) or {}

    def write_models(self, models: Iterable[ModelShadowAccount]) -> None:
        for model in models:
            model_dir = self.models_dir / model.model_id
            model_dir.mkdir(parents=True, exist_ok=True)
            self.write_json(model_dir / "state.json", model.to_dict())

    def read_models(self) -> List[Dict[str, Any]]:
        models: List[Dict[str, Any]] = []
        if not self.models_dir.exists():
            return models
        for state_path in sorted(self.models_dir.glob("*/state.json")):
            payload = self.read_json(state_path)
            if payload:
                models.append(payload)
        return models

    def write_pid(self, pid: int) -> None:
        self.pid_path.write_text(str(pid), encoding="utf-8")

    def read_pid(self) -> Optional[int]:
        if not self.pid_path.exists():
            return None
        try:
            return int(self.pid_path.read_text(encoding="utf-8").strip())
        except Exception:
            return None

    def clear_pid(self) -> None:
        if self.pid_path.exists():
            self.pid_path.unlink()

    def set_stop_requested(self) -> None:
        self.stop_signal_path.write_text("stop", encoding="utf-8")

    def clear_stop_requested(self) -> None:
        if self.stop_signal_path.exists():
            self.stop_signal_path.unlink()

    def stop_requested(self) -> bool:
        return self.stop_signal_path.exists()
