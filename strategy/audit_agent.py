"""
Autonomous Audit Agent:
- Runs every 2-4 hours
- Collects prediction model metrics, CLV data, trade results, feature weights
- Logs findings and optionally auto-applies safe suggestions (no local LLM/Ollama)
"""
from __future__ import annotations

import json
import logging
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import config

logger = logging.getLogger(__name__)

AUDIT_INTERVAL_SECONDS = int(os.getenv("AUDIT_INTERVAL_SECONDS", "7200"))  # 2 hours
AUDIT_LOG_DIR = os.getenv("AUDIT_LOG_DIR", "data/audit")

# Safe env keys the audit agent can suggest changing
AUDIT_ALLOWED_ENV_KEYS = {
    "PREDICTION_MIN_EDGE",
    "PREDICTION_STAKE_FRACTION",
    "VALUE_BET_MIN_EDGE",
    "VALUE_BET_KELLY_FRACTION",
    "SCAN_MAX_MARKETS",
    "SCAN_INTERVAL_SECONDS",
}


@dataclass(frozen=True)
class AuditReport:
    ts: str
    mode: str  # "llm" or "metrics_only"
    metrics: Dict[str, Any]
    analysis: str
    recommendations: List[dict]
    applied: bool


class AuditAgent:
    def __init__(self):
        self.interval_seconds = max(600, AUDIT_INTERVAL_SECONDS)
        self._log_dir = Path(AUDIT_LOG_DIR)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = self._log_dir / "reports.jsonl"
        self.last_report: Optional[AuditReport] = None

    def _log(self, report: AuditReport) -> None:
        payload = {
            "ts": report.ts,
            "mode": report.mode,
            "metrics": report.metrics,
            "analysis": report.analysis,
            "recommendations": report.recommendations,
            "applied": report.applied,
        }
        try:
            with self._log_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, separators=(",", ":")) + "\n")
        except Exception:
            logger.exception("Failed to write audit report")

    def _read_jsonl_tail(self, path: Path, max_lines: int = 1000) -> List[dict]:
        if not path.exists():
            return []
        lines = []
        with path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            data = b""
            block = 1024 * 1024
            while size > 0 and data.count(b"\n") < max_lines + 10:
                rd = min(block, size)
                size -= rd
                f.seek(size)
                data = f.read(rd) + data
        for ln in data.decode("utf-8", "ignore").splitlines():
            if ln.strip():
                try:
                    lines.append(json.loads(ln))
                except json.JSONDecodeError:
                    continue
        return lines[-max_lines:]

    def _collect_prediction_metrics(self) -> Dict[str, Any]:
        state_dir = Path("data/prediction/state")
        if not state_dir.exists():
            return {}
        models = {}
        for f in state_dir.glob("*.json"):
            try:
                raw = json.loads(f.read_text(encoding="utf-8"))
                model_id = raw.get("model_id", f.stem)
                settled = int(raw.get("settled_bets", 0))
                brier = raw.get("brier_sum", 0) / max(1, settled) if settled > 0 else None
                models[model_id] = {
                    "settled_bets": settled,
                    "avg_brier": round(brier, 6) if brier else None,
                    "wins": raw.get("wins", 0),
                    "losses": raw.get("losses", 0),
                    "total_pnl": raw.get("total_pnl", 0),
                    "balance": raw.get("balance", 0),
                    "stake_fraction": raw.get("stake_fraction", 0),
                    "min_edge": raw.get("min_edge", 0),
                    "model_updates": raw.get("update_count", 0),
                }
            except Exception:
                continue
        return models

    def _collect_feature_weights(self) -> Dict[str, Any]:
        model_dir = Path("data/prediction/models")
        if not model_dir.exists():
            return {}
        weights = {}
        for f in model_dir.glob("*.json"):
            try:
                raw = json.loads(f.read_text(encoding="utf-8"))
                w = raw.get("weights", raw.get("w", {}))
                if isinstance(w, dict):
                    weights[f.stem] = w
            except Exception:
                continue
        return weights

    def _collect_clv_summary(self) -> Dict[str, Any]:
        clv_dir = Path("data/clv")
        if not clv_dir.exists():
            return {"entries": 0}
        entries = []
        for f in sorted(clv_dir.glob("*.jsonl"))[-3:]:
            entries.extend(self._read_jsonl_tail(f, max_lines=500))
        if not entries:
            return {"entries": 0}
        clvs = [e.get("clv", 0) for e in entries if "clv" in e]
        return {
            "entries": len(entries),
            "avg_clv": round(sum(clvs) / max(1, len(clvs)), 6) if clvs else None,
            "positive_clv_rate": round(sum(1 for c in clvs if c > 0) / max(1, len(clvs)), 4) if clvs else None,
        }

    def _collect_trade_summary(self) -> Dict[str, Any]:
        trades_path = Path(config.PAPER_TRADES_LOG_PATH)
        trades = self._read_jsonl_tail(trades_path, max_lines=500)
        if not trades:
            return {"total_trades": 0}
        by_type = Counter()
        pnl_by_type: Dict[str, float] = {}
        for t in trades:
            atype = t.get("arb_type", "unknown")
            by_type[atype] += 1
            pnl_by_type[atype] = pnl_by_type.get(atype, 0) + float(t.get("net_profit_eur", 0))
        return {
            "total_trades": len(trades),
            "by_type": dict(by_type),
            "pnl_by_type": {k: round(v, 4) for k, v in pnl_by_type.items()},
        }

    def _collect_architect_summary(self) -> Dict[str, Any]:
        decisions = self._read_jsonl_tail(
            Path("data/architect/decisions.jsonl"), max_lines=50
        )
        if not decisions:
            return {"total_decisions": 0}
        llm_count = sum(1 for d in decisions if d.get("mode") == "llm")
        applied_count = sum(1 for d in decisions if d.get("applied"))
        return {
            "total_decisions": len(decisions),
            "llm_decisions": llm_count,
            "applied": applied_count,
        }

    def collect_all_metrics(self) -> Dict[str, Any]:
        return {
            "prediction_models": self._collect_prediction_metrics(),
            "feature_weights": self._collect_feature_weights(),
            "clv": self._collect_clv_summary(),
            "trades": self._collect_trade_summary(),
            "architect": self._collect_architect_summary(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _metrics_only_analysis(self, metrics: Dict[str, Any]) -> str:
        trades = metrics.get("trades") or {}
        clv = metrics.get("clv") or {}
        total_trades = int(trades.get("total_trades", 0) or 0)
        avg_clv = clv.get("avg_clv")
        parts = [f"Trades analyzed: {total_trades}."]
        if avg_clv is not None:
            parts.append(f"Avg CLV: {avg_clv:+.4f}.")
        return " ".join(parts)

    def _sanitize_recommendations(self, recs: list) -> List[dict]:
        out = []
        for r in recs:
            if not isinstance(r, dict):
                continue
            rtype = str(r.get("type", "")).strip().lower()
            if rtype == "set_env":
                key = str(r.get("key", "")).strip()
                if key not in AUDIT_ALLOWED_ENV_KEYS:
                    continue
                out.append({
                    "type": "set_env",
                    "key": key,
                    "value": str(r.get("value", "")),
                    "reason": str(r.get("reason", "")),
                })
            elif rtype == "observation":
                out.append({
                    "type": "observation",
                    "reason": str(r.get("reason", "")),
                })
        return out[:3]

    def _apply_recommendations(self, recs: List[dict]) -> bool:
        applied = False
        env_path = Path(".env")
        for r in recs:
            if r.get("type") != "set_env":
                continue
            key = r["key"]
            value = r["value"]
            lines: List[str] = []
            if env_path.exists():
                lines = env_path.read_text(encoding="utf-8").splitlines()
            found = False
            out_lines: List[str] = []
            for line in lines:
                if line.startswith(f"{key}="):
                    out_lines.append(f"{key}={value}")
                    found = True
                else:
                    out_lines.append(line)
            if not found:
                out_lines.append(f"{key}={value}")
            env_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
            applied = True
        return applied

    async def run(
        self,
        is_running: Callable[[], bool],
        auto_apply: bool = False,
        on_report: Optional[Callable[[dict], None]] = None,
    ) -> None:
        import asyncio

        async def _sleep_interruptible(seconds: float) -> None:
            remaining = max(0.0, float(seconds))
            while remaining > 0 and is_running():
                step = min(5.0, remaining)
                await asyncio.sleep(step)
                remaining -= step

        while is_running():
            now = datetime.now(timezone.utc)
            try:
                metrics = await asyncio.to_thread(self.collect_all_metrics)
                analysis = self._metrics_only_analysis(metrics)
                recs = []
                mode = "metrics_only"

                applied = False
                if auto_apply and recs:
                    applied = await asyncio.to_thread(self._apply_recommendations, recs)

                report = AuditReport(
                    ts=now.isoformat(),
                    mode=mode,
                    metrics=metrics,
                    analysis=analysis,
                    recommendations=recs,
                    applied=applied,
                )
            except Exception as e:
                logger.exception("Audit agent cycle failed: %s", e)
                report = AuditReport(
                    ts=now.isoformat(),
                    mode="error",
                    metrics={},
                    analysis=f"Error: {type(e).__name__}: {e}",
                    recommendations=[],
                    applied=False,
                )

            self.last_report = report
            self._log(report)
            if on_report is not None:
                try:
                    on_report({
                        "ts": report.ts,
                        "mode": report.mode,
                        "analysis": report.analysis,
                        "recommendations": report.recommendations,
                        "applied": report.applied,
                    })
                except Exception:
                    pass
            await _sleep_interruptible(self.interval_seconds)
