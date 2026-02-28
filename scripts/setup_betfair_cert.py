#!/usr/bin/env python3
"""
Ensure a Betfair client certificate exists, then show the one-time upload step.
Run from project root: python scripts/setup_betfair_cert.py

Cert-based login is the recommended method for this project (no browser, works headless).
"""
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CERTS_DIR = PROJECT_ROOT / "certs"
CRT_FILE = CERTS_DIR / "client-2048.crt"
KEY_FILE = CERTS_DIR / "client-2048.key"
UPLOAD_URL = "https://myaccount.betfair.es/accountdetails/mysecurity?showAPI=1"


def main() -> None:
    if CRT_FILE.exists() and KEY_FILE.exists():
        print("Certificate already exists:")
        print(f"  {CRT_FILE}")
        print(f"  {KEY_FILE}")
    else:
        print("Creating certificate (OpenSSL)...")
        script = PROJECT_ROOT / "scripts" / "create_betfair_cert.sh"
        if not script.exists():
            print(f"Missing {script}", file=sys.stderr)
            sys.exit(1)
        try:
            subprocess.run(
                [os.environ.get("SHELL", "bash"), str(script)],
                cwd=str(PROJECT_ROOT),
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Certificate creation failed: {e}", file=sys.stderr)
            sys.exit(1)
        if not CRT_FILE.exists():
            print("Certificate was not created.", file=sys.stderr)
            sys.exit(1)

    print()
    print("Next step (one-time): link the certificate to your Betfair account")
    print("  1. Log in at https://www.betfair.es")
    print("  2. Open this page:")
    print(f"     {UPLOAD_URL}")
    print("  3. Under 'Automated Betting Program Access' click Edit")
    print("  4. Browse and select: " + str(CRT_FILE))
    print("  5. Click Upload Certificate")
    print()
    print("Then set in .env: BF_CERTS_PATH=./certs")
    print("After that, main.py and test_real_data.py will use cert login automatically.")


if __name__ == "__main__":
    main()
