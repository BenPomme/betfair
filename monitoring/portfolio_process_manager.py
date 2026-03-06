from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

from monitoring.portfolio_registry import get_portfolio_spec
from portfolio.state_store import PortfolioStateStore


class PortfolioProcessManager:
    def __init__(self) -> None:
        self._processes: Dict[str, subprocess.Popen] = {}
        self._project_root = Path(__file__).resolve().parent.parent
        self._script_path = self._project_root / "scripts" / "run_portfolio.py"

    @staticmethod
    def _pid_running(pid: Optional[int]) -> bool:
        if not pid or pid <= 0:
            return False
        if os.name == "nt":
            try:
                process = ctypes.windll.kernel32.OpenProcess(0x1000, 0, int(pid))
                if not process:
                    return False
                try:
                    exit_code = ctypes.c_ulong()
                    if ctypes.windll.kernel32.GetExitCodeProcess(process, ctypes.byref(exit_code)) == 0:
                        return False
                    return int(exit_code.value) == 259
                finally:
                    ctypes.windll.kernel32.CloseHandle(process)
            except Exception:
                return False
        try:
            os.kill(pid, 0)
            return True
        except Exception:
            return False

    def status(self, portfolio_id: str) -> Dict[str, object]:
        store = PortfolioStateStore(portfolio_id)
        pid = store.read_pid()
        heartbeat = store.read_heartbeat()
        return {
            "running": self._pid_running(pid),
            "pid": pid,
            "heartbeat": heartbeat,
        }

    def start(self, portfolio_id: str) -> Dict[str, object]:
        spec = get_portfolio_spec(portfolio_id)
        if spec.control_mode == "disabled":
            return {"ok": False, "error": "portfolio_disabled"}
        store = PortfolioStateStore(portfolio_id)
        pid = store.read_pid()
        if self._pid_running(pid):
            return {"ok": False, "error": "already_running", "pid": pid}
        store.clear_stop_requested()
        log_path = store.base_dir / "runner.log"
        log_handle = log_path.open("a", encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, str(self._script_path), "--portfolio", portfolio_id],
            cwd=str(self._project_root),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        self._processes[portfolio_id] = proc
        store.write_pid(proc.pid)
        return {"ok": True, "pid": proc.pid}

    def stop(self, portfolio_id: str, timeout: float = 20.0) -> Dict[str, object]:
        store = PortfolioStateStore(portfolio_id)
        pid = store.read_pid()
        proc = self._processes.get(portfolio_id)
        if not self._pid_running(pid) and (proc is None or proc.poll() is not None):
            store.clear_stop_requested()
            store.clear_pid()
            return {"ok": False, "error": "not_running"}
        store.set_stop_requested()
        deadline = time.time() + timeout
        while time.time() < deadline:
            live_pid = store.read_pid()
            if not self._pid_running(live_pid):
                break
            time.sleep(0.5)
        live_pid = store.read_pid()
        if self._pid_running(live_pid):
            try:
                if proc is not None and proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=5.0)
                else:
                    os.kill(live_pid, signal.SIGTERM)
            except Exception:
                if proc is not None and proc.poll() is None:
                    proc.kill()
        self._processes.pop(portfolio_id, None)
        store.clear_pid()
        store.clear_stop_requested()
        return {"ok": True}

    def restart(self, portfolio_id: str) -> Dict[str, object]:
        self.stop(portfolio_id)
        return self.start(portfolio_id)
