"""
Live QA Agent:
- Monitors runtime quality (freshness, model artifacts, learning activity)
- Applies bounded safe tweaks (.env keys + safe maintenance commands)
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import config

logger = logging.getLogger(__name__)


ALLOWED_ENV_KEYS = {
    "PAPER_TRADING",
    "SCAN_MAX_MARKETS",
    "SCAN_MAX_HOURS_AHEAD",
    "SCAN_INTERVAL_SECONDS",
    "PRICE_POLL_INTERVAL_SECONDS",
    "POLLER_BATCH_SIZE",
    "POLLER_CONCURRENT_BATCHES",
    "POLLER_REQUEST_TIMEOUT_SECONDS",
    "STALE_PRICE_SECONDS_PREMATCH",
    "STALE_PRICE_SECONDS_INPLAY",
    "PREDICTION_ENABLED",
    "ARCHITECT_ENABLED",
}


@dataclass(frozen=True)
class QADecision:
    ts: str
    mode: str  # "rules"
    applied: bool
    reason: str
    metrics: Dict[str, Any]
    actions: List[dict]
    results: List[dict]


class LiveQAAgent:
    def __init__(self):
        self.interval_seconds = max(60, int(config.QA_AGENT_INTERVAL_SECONDS))
        self.last_run_ts: Optional[datetime] = None
        self.last_decision: Optional[QADecision] = None
        self._log_dir = Path(config.QA_LOG_DIR)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = self._log_dir / "decisions.jsonl"
        self._degraded_since_ts: Optional[datetime] = None

    def _log(self, decision: QADecision) -> None:
        payload = {
            "ts": decision.ts,
            "mode": decision.mode,
            "applied": decision.applied,
            "reason": decision.reason,
            "metrics": decision.metrics,
            "actions": decision.actions,
            "results": decision.results,
        }
        try:
            with self._log_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, separators=(",", ":")) + "\n")
        except Exception:
            logger.exception("Failed to write QA decision log")

    def _read_tail_lines(self, path: Path, max_lines: int = 50000) -> List[str]:
        if not path.exists():
            return []
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
        lines = [ln for ln in data.decode("utf-8", "ignore").splitlines() if ln.strip()]
        return lines[-max_lines:]

    def _collect_candidate_metrics(self) -> Dict[str, Any]:
        today = datetime.now(timezone.utc).date().isoformat()
        path = Path(config.CANDIDATE_LOG_DIR) / f"{today}.jsonl"
        lines = self._read_tail_lines(path, max_lines=50000)
        if not lines:
            return {
                "records": 0,
                "fresh_snapshot_rate": 0.0,
                "stale_rate": 0.0,
                "scored_rate": 0.0,
                "executed_rate": 0.0,
                "top_reasons": {},
            }
        reasons = Counter()
        with_snapshot = 0
        scored = 0
        executed = 0
        for ln in lines:
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            reasons[str(r.get("reason", ""))] += 1
            if r.get("has_snapshot") is True:
                with_snapshot += 1
            if r.get("decision"):
                scored += 1
            if r.get("executed"):
                executed += 1
        total = max(1, len(lines))
        stale = reasons.get("stale_or_missing_snapshot", 0)
        return {
            "records": len(lines),
            "fresh_snapshot_rate": with_snapshot / total,
            "stale_rate": stale / total,
            "scored_rate": scored / total,
            "executed_rate": executed / total,
            "top_reasons": dict(reasons.most_common(6)),
        }

    def _collect_prediction_metrics(self, prediction_manager: Optional[Any]) -> Dict[str, Any]:
        if prediction_manager is None:
            return {"enabled": False, "models": 0, "settled_total": 0}
        states = prediction_manager.initial_state()
        settled = sum(int(s.get("settled_bets", 0)) for s in states.values())
        updates = sum(int(s.get("model_updates", 0)) for s in states.values())
        model_details = {}
        for model_id, st in states.items():
            model_details[model_id] = {
                "settled_bets": int(st.get("settled_bets", 0)),
                "avg_brier": float(st.get("avg_brier", 0)),
                "roi_pct": float(st.get("roi_pct", 0)),
                "win_rate": float(st.get("win_rate", 0)),
                "total_pnl_eur": float(st.get("total_pnl_eur", 0)),
                "stake_fraction": float(st.get("stake_fraction", 0.05)),
                "min_edge": float(st.get("min_edge", 0.03)),
            }
        return {
            "enabled": True,
            "models": len(states),
            "settled_total": settled,
            "model_updates_total": updates,
            "model_details": model_details,
        }

    def _collect_clv_metrics(self) -> Dict[str, Any]:
        clv_dir = Path(config.CLV_LOG_DIR)
        if not clv_dir.exists():
            return {"entries": 0}
        entries = []
        for f in sorted(clv_dir.glob("*.jsonl"))[-2:]:
            lines = self._read_tail_lines(f, max_lines=200)
            for ln in lines:
                try:
                    entries.append(json.loads(ln))
                except json.JSONDecodeError:
                    continue
        if not entries:
            return {"entries": 0}
        clvs = [e.get("clv", 0) for e in entries if "clv" in e]
        return {
            "entries": len(entries),
            "avg_clv": round(sum(clvs) / max(1, len(clvs)), 6) if clvs else None,
            "positive_clv_rate": round(sum(1 for c in clvs if c > 0) / max(1, len(clvs)), 4) if clvs else None,
        }

    def _collect_artifact_metrics(self) -> Dict[str, Any]:
        def _exists(path: str) -> bool:
            return bool(path) and Path(path).exists()

        return {
            "scoring_model_exists": _exists(config.ML_LINEAR_MODEL_PATH),
            "fill_model_exists": _exists(config.FILL_MODEL_PATH),
        }

    def _collect_runtime_metrics(
        self,
        runtime_state_provider: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        snapshot: Dict[str, Any] = {}
        if runtime_state_provider is not None:
            try:
                payload = runtime_state_provider()
                if isinstance(payload, dict):
                    snapshot = payload
            except Exception:
                logger.exception("Runtime state provider failed")
        running = bool(snapshot.get("running", False))
        health = snapshot.get("health", {})
        if not isinstance(health, dict):
            health = {}
        system_ok = bool(health.get("system_ok", True))
        risk_ok = bool(health.get("risk_ok", True))
        feed_status = str(health.get("feed_status", "unknown") or "unknown")
        recovery_action_state = str(health.get("recovery_action_state", "") or "")
        degraded = bool(running and not system_ok)
        now = datetime.now(timezone.utc)
        if degraded:
            if self._degraded_since_ts is None:
                self._degraded_since_ts = now
            degraded_seconds = max(0.0, (now - self._degraded_since_ts).total_seconds())
        else:
            self._degraded_since_ts = None
            degraded_seconds = 0.0
        return {
            "running": running,
            "system_ok": system_ok,
            "risk_ok": risk_ok,
            "degraded": degraded,
            "degraded_seconds": round(degraded_seconds, 1),
            "last_scan_age_sec": health.get("last_scan_age_sec"),
            "last_prediction_age_sec": health.get("last_prediction_age_sec"),
            "last_architect_age_sec": health.get("last_architect_age_sec"),
            "feed_status": feed_status,
            "recovery_action_state": recovery_action_state,
        }

    def collect_metrics(
        self,
        prediction_manager: Optional[Any],
        runtime_state_provider: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        return {
            "candidate": self._collect_candidate_metrics(),
            "prediction": self._collect_prediction_metrics(prediction_manager),
            "artifacts": self._collect_artifact_metrics(),
            "runtime": self._collect_runtime_metrics(runtime_state_provider),
            "clv": self._collect_clv_metrics(),
        }

    def _rules_actions(self, metrics: Dict[str, Any]) -> List[dict]:
        c = metrics.get("candidate", {})
        a = metrics.get("artifacts", {})
        r = metrics.get("runtime", {})
        actions: List[dict] = []

        stale_rate = float(c.get("stale_rate", 0.0))
        fresh_rate = float(c.get("fresh_snapshot_rate", 0.0))

        if stale_rate > 0.90:
            actions.append(
                {
                    "type": "set_env",
                    "key": "SCAN_MAX_MARKETS",
                    "value": "80",
                    "reason": "high_stale_rate_reduce_universe",
                }
            )
            actions.append(
                {
                    "type": "set_env",
                    "key": "PRICE_POLL_INTERVAL_SECONDS",
                    "value": "0.8",
                    "reason": "high_stale_rate_faster_poll",
                }
            )
        elif fresh_rate > 0.50 and stale_rate < 0.50:
            actions.append(
                {
                    "type": "set_env",
                    "key": "SCAN_MAX_MARKETS",
                    "value": "120",
                    "reason": "freshness_recovered_expand_universe",
                }
            )

        if not a.get("scoring_model_exists") or not a.get("fill_model_exists"):
            actions.append({"type": "run_cmd", "cmd": "retrain_models", "reason": "missing_artifacts"})

        degraded_seconds = float(r.get("degraded_seconds", 0.0))
        should_restart = bool(
            config.QA_RESTART_ON_DEGRADED_ENABLED
            and r.get("running") is True
            and r.get("system_ok") is False
            and r.get("risk_ok", True) is True
            and degraded_seconds >= float(config.QA_DEGRADED_MIN_AGE_SECONDS)
            and r.get("recovery_action_state") not in {"auth_blocked"}
        )
        if should_restart:
            actions.append(
                {
                    "type": "restart_runtime",
                    "reason": "health_degraded_while_running",
                }
            )
        elif r.get("feed_status") == "degraded" and r.get("recovery_action_state") == "refresh_market_universe":
            actions.append(
                {
                    "type": "set_env",
                    "key": "SCAN_MAX_MARKETS",
                    "value": "80",
                    "reason": "feed_degraded_reduce_universe",
                }
            )

        return actions

    def answer_question_sync(self, question: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Operator-facing Q&A with deterministic rules only (no LLM).

        This intentionally avoids any local-model/Ollama dependencies to reduce RAM usage.
        """
        q = str(question or "").strip()
        if not q:
            raise ValueError("question_required")
        ctx = context or {}
        runtime = (ctx.get("runtime_state") or {}).get("health") or {}
        last = ctx.get("last_qa_decision") or {}
        running = bool((ctx.get("runtime_state") or {}).get("running", False))
        system_ok = runtime.get("system_ok")
        risk_ok = runtime.get("risk_ok")
        mode = str(last.get("mode", "rules"))
        applied = bool(last.get("applied", False))
        reason = str(last.get("reason", ""))
        actions = last.get("actions") or []
        line = f"Runtime running={running}, system_ok={system_ok}, risk_ok={risk_ok}. Last QA: mode={mode}, applied={applied}, reason={reason}, actions={len(actions)}."
        return line

    def _sanitize_actions(self, actions: List[dict]) -> List[dict]:
        out: List[dict] = []
        for a in actions:
            if not isinstance(a, dict):
                continue
            t = str(a.get("type", "")).strip().lower()
            if t == "set_env":
                key = str(a.get("key", "")).strip()
                if key not in ALLOWED_ENV_KEYS:
                    continue
                out.append(
                    {
                        "type": "set_env",
                        "key": key,
                        "value": str(a.get("value", "")),
                        "reason": str(a.get("reason", "")),
                    }
                )
            elif t == "run_cmd":
                cmd = str(a.get("cmd", "")).strip()
                if cmd not in {"retrain_models", "validate_gates", "qa_scan"}:
                    continue
                out.append({"type": "run_cmd", "cmd": cmd, "reason": str(a.get("reason", ""))})
            elif t == "restart_runtime":
                out.append({"type": "restart_runtime", "reason": str(a.get("reason", ""))})
        return out[:3]

    def _set_env_key(self, key: str, value: str) -> bool:
        env_path = Path(".env")
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
        return True

    async def _run_safe_command(self, cmd_name: str) -> Dict[str, Any]:
        if cmd_name == "retrain_models":
            cmd = [".venv/bin/bash", "scripts/retrain_models.sh"]
        elif cmd_name == "validate_gates":
            cmd = [".venv/bin/python", "scripts/validate_performance_gates.py"]
        elif cmd_name == "qa_scan":
            cmd = [".venv/bin/pytest", "-q"]
        else:
            return {"ok": False, "reason": "unsupported_cmd"}
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=180.0)
            return {
                "ok": proc.returncode == 0,
                "returncode": proc.returncode,
                "stdout_tail": stdout.decode("utf-8", "ignore")[-500:],
                "stderr_tail": stderr.decode("utf-8", "ignore")[-500:],
            }
        except Exception as e:
            return {"ok": False, "reason": f"exec_error:{e}"}

    async def _apply_actions(
        self,
        actions: List[dict],
        request_shutdown: Optional[Callable[[], None]] = None,
        request_restart: Optional[Callable[[str], bool]] = None,
    ) -> List[dict]:
        results: List[dict] = []
        for a in actions:
            if a.get("type") == "set_env":
                key = a["key"]
                value = a["value"]
                ok = self._set_env_key(key, value)
                out = {"action": a, "ok": ok}
                if key == "PAPER_TRADING" and request_shutdown is not None:
                    request_shutdown()
                    out["requires_restart"] = True
                results.append(out)
            elif a.get("type") == "run_cmd":
                cmd_result = await self._run_safe_command(a["cmd"])
                results.append({"action": a, **cmd_result})
            elif a.get("type") == "restart_runtime":
                reason = str(a.get("reason", "")) or "qa_requested_restart"
                ok = bool(request_restart(reason)) if request_restart is not None else False
                results.append(
                    {
                        "action": a,
                        "ok": ok,
                        "requires_restart": True,
                    }
                )
        return results

    async def run(
        self,
        is_running: Callable[[], bool],
        prediction_manager: Optional[Any] = None,
        on_decision: Optional[Callable[[dict], None]] = None,
        runtime_state_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        request_shutdown: Optional[Callable[[], None]] = None,
        request_restart: Optional[Callable[[str], bool]] = None,
    ) -> None:
        async def _sleep_interruptible(seconds: float) -> None:
            remaining = max(0.0, float(seconds))
            while remaining > 0 and is_running():
                step = min(5.0, remaining)
                await asyncio.sleep(step)
                remaining -= step

        while is_running():
            now = datetime.now(timezone.utc)
            try:
                metrics = self.collect_metrics(
                    prediction_manager,
                    runtime_state_provider=runtime_state_provider,
                )
                mode = "rules"
                proposed = self._rules_actions(metrics)
                actions = self._sanitize_actions(proposed)
                results: List[dict] = []
                applied = False
                reason = "ok" if actions else "no_action"
                if actions and config.QA_AGENT_AUTO_APPLY:
                    results = await self._apply_actions(
                        actions,
                        request_shutdown=request_shutdown,
                        request_restart=request_restart,
                    )
                    applied = any(bool(r.get("ok")) for r in results)
                decision = QADecision(
                    ts=now.isoformat(),
                    mode=mode,
                    applied=applied,
                    reason=reason,
                    metrics=metrics,
                    actions=actions,
                    results=results,
                )
            except Exception as e:
                logger.exception("QA agent cycle failed")
                decision = QADecision(
                    ts=now.isoformat(),
                    mode="rules",
                    applied=False,
                    reason=f"cycle_error:{type(e).__name__}",
                    metrics={},
                    actions=[],
                    results=[],
                )
            self.last_run_ts = now
            self.last_decision = decision
            self._log(decision)
            if on_decision is not None:
                try:
                    on_decision(
                        {
                            "ts": decision.ts,
                            "mode": decision.mode,
                            "applied": decision.applied,
                            "reason": decision.reason,
                            "actions": decision.actions,
                            "results": decision.results,
                            "metrics": decision.metrics,
                        }
                    )
                except Exception:
                    pass
            await _sleep_interruptible(self.interval_seconds)
