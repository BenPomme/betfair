#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

try:
    from dotenv import load_dotenv

    load_dotenv(project_root / ".env")
except ImportError:
    pass

import config


STATE_PATH = project_root / "data" / "runtime" / "deploy_watcher_state.json"


def _run_git(args: list[str]) -> str:
    out = subprocess.check_output(["git", *args], cwd=str(project_root), stderr=subprocess.STDOUT)
    return out.decode("utf-8", errors="replace").strip()


def _write_state(payload: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
            else:
                state["status"] = "updates_available" if state["updates_available"] else "up_to_date"
        except Exception as exc:
            state["status"] = "error"
            state["error"] = str(exc)
        _write_state(state)
        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(main())
