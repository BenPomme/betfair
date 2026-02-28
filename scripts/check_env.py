#!/usr/bin/env python3
"""
Check that .env and config are ready for running the engine.
Run this when your API key arrives to confirm before starting the stream.
"""
import os
import sys

# Load .env from project root
try:
    from dotenv import load_dotenv
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    load_dotenv(root / ".env")
except ImportError:
    pass

def main():
    errors = []
    warnings = []

    username = os.getenv("BF_USERNAME", "").strip()
    password = os.getenv("BF_PASSWORD", "").strip()
    app_key = os.getenv("BF_APP_KEY", "").strip()

    if not username:
        errors.append("BF_USERNAME is not set in .env")
    if not password:
        errors.append("BF_PASSWORD is not set in .env")
    if not app_key or app_key == "your_app_key":
        warnings.append("BF_APP_KEY is missing or still placeholder — add your key from developer.betfair.com")

    paper = os.getenv("PAPER_TRADING", "true").lower() == "true"
    if not paper:
        warnings.append("PAPER_TRADING is false — only set after paper gate is passed")

    if errors:
        print("Errors (fix these):")
        for e in errors:
            print("  -", e)
    if warnings:
        print("Warnings:")
        for w in warnings:
            print("  -", w)

    if errors:
        sys.exit(1)
    if warnings:
        print("\nYou can run the app; fix warnings when you want streaming/live.")
    else:
        print("Config looks good. Run with PAPER_TRADING=true first.")
    sys.exit(0)

if __name__ == "__main__":
    main()
