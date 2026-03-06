#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List
from urllib import error, request

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

try:
    from dotenv import load_dotenv

    load_dotenv(project_root / ".env")
except ImportError:
    pass

import config
from monitoring.transports.discord_webhook import send_discord


STATE_PATH = project_root / "data" / "runtime" / "deploy_watcher_state.json"
DASHBOARD_URL = f"http://127.0.0.1:{getattr(config, 'COMMAND_CENTER_PORT', 8000)}"


def _run_git(args: list[str]) -> str:
    out = subprocess.check_output(["git", *args], cwd=str(project_root), stderr=subprocess.STDOUT)
    return out.decode("utf-8", errors="replace").strip()


def _write_state(payload: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _dashboard_pid() -> int | None:
    try:
        command = (
            f"(Get-NetTCPConnection -LocalPort {int(getattr(config, 'COMMAND_CENTER_PORT', 8000))} "
            f"-State Listen | Select-Object -First 1 -ExpandProperty OwningProcess)"
        )
        out = subprocess.check_output(["powershell", "-Command", command], cwd=str(project_root), stderr=subprocess.DEVNULL)
        value = out.decode("utf-8", errors="replace").strip()
        return int(value) if value else None
    except Exception:
        return None


def _wait_for_dashboard(timeout: float = 45.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with request.urlopen(f"{DASHBOARD_URL}/api/portfolios", timeout=5):
                return True
        except Exception:
            time.sleep(1.0)
    return False


def _api_call(path: str, method: str = "GET") -> dict | None:
    req = request.Request(f"{DASHBOARD_URL}{path}", method=method)
    try:
        with request.urlopen(req, timeout=15) as response:
            payload = response.read().decode("utf-8")
        return json.loads(payload) if payload else {}
    except error.HTTPError as exc:
        return {"ok": False, "error": f"http_{exc.code}"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _restart_portfolios(portfolios: List[str]) -> List[str]:
    restarted: List[str] = []
    for portfolio_id in portfolios:
        result = _api_call(f"/api/portfolios/{portfolio_id}/start", method="POST")
        if result and result.get("ok"):
            restarted.append(portfolio_id)
    return restarted


def _perform_restart_cycle(state: dict) -> None:
    portfolios = [item.strip() for item in str(getattr(config, "DEPLOY_WATCHER_AUTO_RESTART_PORTFOLIOS", "")).split(",") if item.strip()]
    for portfolio_id in portfolios:
        _api_call(f"/api/portfolios/{portfolio_id}/stop", method="POST")
    if bool(getattr(config, "DEPLOY_WATCHER_AUTO_RESTART_DASHBOARD", False)):
        dashboard_pid = _dashboard_pid()
        if dashboard_pid:
            try:
                os.kill(dashboard_pid, 15)
            except Exception:
                pass
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        subprocess.Popen(
            [sys.executable, str(project_root / "scripts" / "run_dashboard.py")],
            cwd=str(project_root),
            creationflags=creationflags,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if not _wait_for_dashboard():
            state["status"] = "restart_failed"
            state["error"] = "dashboard_restart_timeout"
            return
    restarted = _restart_portfolios(portfolios)
    state["restarted_portfolios"] = restarted


def _send_update_notification(state: dict) -> None:
    if not config.NOTIFICATIONS_ENABLED:
        return
    if not state.get("last_deploy_ts"):
        return
    lines = [
        f"Auto-deploy applied on {state.get('branch')}",
        f"Commit {state.get('local_head')}",
    ]
    if state.get("restarted_portfolios"):
        lines.append("Restarted: " + ", ".join(state.get("restarted_portfolios") or []))
    send_discord("\n".join(lines), username="Strategy Deploy")


def main() -> int:
    interval = max(30, int(getattr(config, "DEPLOY_WATCHER_INTERVAL_SECONDS", 120)))
    remote = str(getattr(config, "DEPLOY_WATCHER_REMOTE", "origin") or "origin")
    branch = str(getattr(config, "DEPLOY_WATCHER_BRANCH", "main") or "main")
    auto_restart = bool(getattr(config, "DEPLOY_WATCHER_AUTO_RESTART_DASHBOARD", False))
    enabled = bool(getattr(config, "DEPLOY_WATCHER_ENABLED", False))
    while True:
        state = {
            "enabled": enabled,
            "branch": branch,
            "remote": remote,
            "interval_seconds": interval,
            "auto_restart_dashboard": auto_restart,
            "status": "idle",
            "last_check_ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "watcher_pid": os.getpid(),
        }
        try:
            _run_git(["fetch", remote, branch])
            local_head = _run_git(["rev-parse", "HEAD"])
            remote_head = _run_git(["rev-parse", f"{remote}/{branch}"])
            state["local_head"] = local_head[:12]
            state["remote_head"] = remote_head[:12]
            state["updates_available"] = local_head != remote_head
            if enabled and state["updates_available"]:
                _run_git(["pull", "--ff-only", remote, branch])
                state["status"] = "updated"
                state["last_deploy_ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                state["local_head"] = _run_git(["rev-parse", "HEAD"])[:12]
                _write_state(state)
                _perform_restart_cycle(state)
                _send_update_notification(state)
            else:
                state["status"] = "updates_available" if state["updates_available"] else "up_to_date"
        except Exception as exc:
            state["status"] = "error"
            state["error"] = str(exc)
        _write_state(state)
        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(main())
