#!/usr/bin/env python3
from __future__ import annotations

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

from factory.binance_auth_diagnostic import (  # noqa: E402
    diagnostics_to_json,
    format_binance_auth_diagnostics,
    run_binance_auth_diagnostics,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a read-only Binance auth diagnostic against prod and testnet spot/futures endpoints.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a human-readable summary.")
    args = parser.parse_args()

    results = run_binance_auth_diagnostics()
    if args.json:
        print(diagnostics_to_json(results))
    else:
        print(format_binance_auth_diagnostics(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
