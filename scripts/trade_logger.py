"""
Lightweight trade logger: polls dashboard, verifies math locally, saves full
trade history + issues to JSON. No LLM needed — just runs all night.

Usage: python scripts/trade_logger.py [--interval 30]

Output: monitoring/reports/trade_log_YYYYMMDD.json (appended each check)
"""
import argparse
import json
import logging
import time
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [LOGGER] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPORT_DIR = Path(__file__).parent.parent / "monitoring" / "reports"
DASHBOARD_URL = "http://127.0.0.1:8000"


def fetch_state(url: str) -> Optional[dict]:
    try:
        r = requests.get(f"{url}/api/state", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error("Dashboard unreachable: %s", e)
        return None


def verify_trade(t: dict) -> List[str]:
    """Local math verification. Returns list of issues."""
    issues = []
    sels = t.get("selections", [])
    if not sels:
        return ["No selections"]

    sel = sels[0]
    arb_type = t.get("arb_type", "")
    net_profit = t.get("net_profit_eur")
    roi = t.get("net_roi_pct", 0)

    if arb_type == "cross_market":
        bp = sel.get("back_price")
        lp = sel.get("lay_price")
        bs = sel.get("stake_eur")
        ls = sel.get("lay_stake_eur")

        if bp and lp and bp <= lp:
            issues.append(f"NO ARB: back {bp} <= lay {lp}")

        if bs and ls and bp and lp:
            expected_ls = round(bs * bp / lp, 2)
            if abs(expected_ls - ls) > 0.05:
                issues.append(f"LAY STAKE WRONG: expected {expected_ls}, got {ls}")

            # Win scenario
            back_win = bs * (bp - 1)
            lay_loss = ls * (lp - 1)
            gross_win = back_win - lay_loss
            comm_win = round(back_win * 0.05, 2)
            net_win = gross_win - comm_win

            # Lose scenario
            gross_lose = ls - bs
            comm_lose = round(ls * 0.05, 2)
            net_lose = gross_lose - comm_lose

            if net_win < 0 or net_lose < 0:
                issues.append(f"NEGATIVE SCENARIO: win={net_win:.2f} lose={net_lose:.2f}")

            if net_profit is not None:
                expected = min(net_win, net_lose)
                if abs(expected - net_profit) > 0.15:
                    issues.append(f"NET MISMATCH: expected {expected:.2f}, got {net_profit:.4f}")

        realistic = t.get("fill_simulated_realistic_net")
        if realistic and net_profit and realistic > net_profit * 1.5:
            issues.append(f"REALISTIC > OPTIMISTIC: {realistic:.2f} > {net_profit:.4f}")

        name = sel.get("name", "")
        if name.isdigit():
            issues.append(f"NUMERIC NAME: '{name}'")

    if roi > 0.25:
        issues.append(f"SUSPICIOUS ROI: {roi*100:.1f}%")

    return issues


def run_logger(dashboard_url: str, interval: int):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.utcnow().strftime("%Y%m%d")
    log_path = REPORT_DIR / f"trade_log_{date_str}.json"

    # Load existing log if resuming
    log_data = {"checks": [], "all_trades": [], "issues_summary": []}
    if log_path.exists():
        try:
            log_data = json.loads(log_path.read_text())
        except Exception:
            pass

    seen_trade_keys = set()
    for t in log_data.get("all_trades", []):
        seen_trade_keys.add(f"{t.get('ts')}_{t.get('market_id')}")

    logger.info("Trade logger started — polling every %ds", interval)
    logger.info("Log file: %s", log_path)

    while True:
        state = fetch_state(dashboard_url)
        if state is None:
            time.sleep(interval)
            continue

        if not state.get("running", False):
            logger.info("Engine stopped. Waiting...")
            time.sleep(interval)
            continue

        trades = state.get("trades", [])
        ss = state.get("scan_stats", {})
        now = datetime.utcnow().isoformat() + "Z"

        # Find new trades
        new_trades = []
        all_issues = {}
        for i, t in enumerate(trades):
            key = f"{t.get('ts')}_{t.get('market_id')}"
            if key not in seen_trade_keys:
                seen_trade_keys.add(key)
                new_trades.append(t)
                log_data["all_trades"].append(t)

            issues = verify_trade(t)
            if issues:
                all_issues[i] = issues

        # Log check
        check = {
            "timestamp": now,
            "balance_eur": state.get("balance_eur"),
            "daily_pnl_eur": state.get("daily_pnl_eur"),
            "trade_count": len(trades),
            "new_trades": len(new_trades),
            "scan_stats": ss,
            "issues_count": len(all_issues),
        }
        log_data["checks"].append(check)

        # Log issues
        if all_issues:
            for idx, issues in all_issues.items():
                entry = {
                    "timestamp": now,
                    "trade_index": idx,
                    "event": trades[idx].get("event", "?"),
                    "market_id": trades[idx].get("market_id", "?"),
                    "issues": issues,
                }
                log_data["issues_summary"].append(entry)
                for issue in issues:
                    logger.warning("Trade %d (%s): %s", idx + 1, trades[idx].get("event", "?"), issue)

        # Save
        log_path.write_text(json.dumps(log_data, indent=2))

        # Console summary
        if new_trades:
            for t in new_trades:
                sel = t.get("selections", [{}])[0]
                logger.info(
                    "NEW TRADE: %s | %s B@%s L@%s | net=%.4f roi=%.2f%%",
                    t.get("event", "?"),
                    sel.get("name", "?"),
                    sel.get("back_price", "?"),
                    sel.get("lay_price", "?"),
                    t.get("net_profit_eur", 0),
                    t.get("net_roi_pct", 0) * 100,
                )
        else:
            logger.info(
                "Balance: %.2f | Trades: %d | Scans: %d | Cross: %d | Issues: %d",
                state.get("balance_eur", 0),
                len(trades),
                ss.get("total_scans", 0),
                ss.get("cross_market_found", 0),
                len(all_issues),
            )

        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--dashboard-url", default=DASHBOARD_URL)
    args = parser.parse_args()
    run_logger(args.dashboard_url, args.interval)
