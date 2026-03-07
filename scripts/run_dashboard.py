#!/usr/bin/env python3
"""
Start the dashboard UI. Open http://127.0.0.1:8000 in your browser,
then click "Start trading" to run paper trading.
Sets cwd to project root and loads .env so credentials work from any launch dir.
Set `DASHBOARD_RELOAD=true` for auto-reload during development.
"""
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

try:
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env", override=True)
except ImportError:
    pass

if __name__ == "__main__":
    import uvicorn
    reload_enabled = os.getenv("DASHBOARD_RELOAD", "false").lower() == "true"
    from config import COMMAND_CENTER_HOST, COMMAND_CENTER_PORT

    print(f"[dashboard] starting command center on :{COMMAND_CENTER_PORT} reload={reload_enabled}")
    uvicorn.run(
        "monitoring.command_center:app",
        host=COMMAND_CENTER_HOST,
        port=COMMAND_CENTER_PORT,
        reload=reload_enabled,
    )
