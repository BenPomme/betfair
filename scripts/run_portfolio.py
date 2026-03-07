#!/usr/bin/env python3
import argparse
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

from monitoring.portfolio_registry import create_runner


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--portfolio", required=True)
    args = parser.parse_args()
    runner = create_runner(args.portfolio)
    runner.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
