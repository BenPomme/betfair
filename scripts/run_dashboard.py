#!/usr/bin/env python3
"""
Start the dashboard UI. Open http://127.0.0.1:8000 in your browser,
then click "Start trading" to run paper trading.
Sets cwd to project root and loads .env so credentials work from any launch dir.
"""
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

try:
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")
except ImportError:
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "monitoring.dashboard:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
