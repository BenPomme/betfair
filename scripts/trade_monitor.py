"""
Autonomous trade monitor: polls the dashboard API, verifies trade arithmetic,
logs findings and saves periodic reports. No local LLM/Ollama dependency.

Usage: python3 scripts/trade_monitor.py [--interval 60] [--dashboard-url http://127.0.0.1:8000]
"""
import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MONITOR] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPORT_DIR = Path(__file__).parent.parent / "monitoring" / "reports"


def fetch_state(dashboard_url: str) -> Optional[dict]:
    try:
        r = requests.get(f"{dashboard_url}/api/state", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error("Failed to fetch dashboard state: %s", e)
        return None


def verify_trade_math(trade: dict) -> List[str]:
    """Verify trade arithmetic locally (no LLM needed). Returns list of issues."""
    issues = []
    sels = trade.get("selections", [])
    if not sels:
        issues.append("No selections in trade")
        return issues

    sel = sels[0]
    arb_type = trade.get("arb_type", "")

    if arb_type == "cross_market":
        back_price = sel.get("back_price")
        lay_price = sel.get("lay_price")
        back_stake = sel.get("stake_eur")
        lay_stake = sel.get("lay_stake_eur")
        net_profit = trade.get("net_profit_eur")
        roi = trade.get("net_roi_pct")

        if back_price is None or lay_price is None:
            issues.append(f"Missing prices: back={back_price} lay={lay_price}")
            return issues

        if back_price <= lay_price:
            issues.append(f"BUG: back_price ({back_price}) <= lay_price ({lay_price}) — no arb exists")

        if back_stake and lay_stake and back_price and lay_price:
            expected_lay = round(back_stake * back_price / lay_price, 2)
            if abs(expected_lay - lay_stake) > 0.05:
                issues.append(f"Lay stake mismatch: expected {expected_lay}, got {lay_stake}")

        if back_stake and lay_stake:
            # Check scenario: selection wins
            back_win = back_stake * (back_price - 1)
            lay_loss = lay_stake * (lay_price - 1)
            gross_win = back_win - lay_loss
            comm_win = round(back_win * 0.05, 2)
            net_win = gross_win - comm_win

            # Check scenario: selection loses
            gross_lose = lay_stake - back_stake
            comm_lose = round(lay_stake * 0.05, 2)
            net_lose = gross_lose - comm_lose

            expected_net = min(net_win, net_lose)
            if net_profit is not None and abs(expected_net - net_profit) > 0.10:
                issues.append(
                    f"Net profit mismatch: expected ~{expected_net:.2f}, got {net_profit:.4f} "
                    f"(win scenario: {net_win:.2f}, lose scenario: {net_lose:.2f})"
                )

            if net_win < 0 or net_lose < 0:
                issues.append(f"BUG: Negative scenario — win: {net_win:.2f}, lose: {net_lose:.2f}")

        if roi is not None and roi > 0.25:
            issues.append(f"Suspicious ROI: {roi*100:.1f}% — likely stale/thin liquidity")

        # Check realistic net sanity
        realistic = trade.get("fill_simulated_realistic_net")
        if realistic is not None and net_profit is not None:
            if realistic > net_profit * 1.5:
                issues.append(
                    f"Realistic net ({realistic:.2f}) > optimistic ({net_profit:.4f}) — "
                    f"realistic calc may be buggy"
                )

        # Selection name check
        name = sel.get("name", "")
        if name.isdigit():
            issues.append(f"Selection name is numeric ID: '{name}' — runner name mapping failed")

    elif arb_type in ("back_back", "lay_lay"):
        # Basic checks for single-market arbs
        if net_profit is not None and net_profit < 0:
            issues.append(f"BUG: Negative net profit on {arb_type}: {net_profit}")

    return issues


def save_report(analysis: str, trade_issues: Dict[int, List[str]], state: dict) -> Path:
    """Save report to file."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = REPORT_DIR / f"monitor_{ts}.json"

    data = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "trade_count": len(state.get("trades", [])),
        "balance_eur": state.get("balance_eur"),
        "daily_pnl_eur": state.get("daily_pnl_eur"),
        "math_issues": {str(k): v for k, v in trade_issues.items()},
        "analysis": analysis,
        "scan_stats": state.get("scan_stats", {}),
    }

    path.write_text(json.dumps(data, indent=2))
    return path


def run_monitor(dashboard_url: str, interval: int) -> None:
    logger.info("Trade monitor started — polling every %ds", interval)
    logger.info("Dashboard: %s", dashboard_url)

    prev_trade_count = 0
    check_count = 0

    while True:
        check_count += 1
        logger.info("--- Check #%d ---", check_count)

        state = fetch_state(dashboard_url)
        if state is None:
            logger.warning("Dashboard unreachable, retrying in %ds", interval)
            time.sleep(interval)
            continue

        trades = state.get("trades", [])
        if not state.get("running", False):
            logger.info("Engine not running. Waiting...")
            time.sleep(interval)
            continue

        # Math verification (always, no LLM cost)
        trade_issues: Dict[int, List[str]] = {}
        for i, t in enumerate(trades):
            issues = verify_trade_math(t)
            if issues:
                trade_issues[i] = issues

        if trade_issues:
            logger.warning("Math issues found in %d trades:", len(trade_issues))
            for idx, issues in trade_issues.items():
                event = trades[idx].get("event", "?")
                for issue in issues:
                    logger.warning("  Trade %d (%s): %s", idx + 1, event, issue)

        # Write a report when there are new trades/issues or periodically
        new_trades = len(trades) > prev_trade_count
        has_issues = bool(trade_issues)

        if new_trades or has_issues or check_count % 5 == 0:
            ss = state.get("scan_stats", {}) or {}
            analysis = (
                f"running={state.get('running', False)} "
                f"balance_eur={state.get('balance_eur', 0)} "
                f"daily_pnl_eur={state.get('daily_pnl_eur', 0)} "
                f"trades={len(trades)} "
                f"scans={ss.get('total_scans', 0)} "
                f"cross_market_found={ss.get('cross_market_found', 0)} "
                f"issues={len(trade_issues)}"
            )
            path = save_report(analysis, trade_issues, state)
            logger.info("Report saved: %s", path)
        else:
            ss = state.get("scan_stats", {})
            logger.info(
                "No new trades. Balance: %.2f EUR | Scans: %d | Cross-market: %d",
                state.get("balance_eur", 0),
                ss.get("total_scans", 0),
                ss.get("cross_market_found", 0),
            )

        prev_trade_count = len(trades)
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Trade monitor with Qwen 3.5 analysis")
    parser.add_argument("--interval", type=int, default=60, help="Check interval in seconds")
    parser.add_argument("--dashboard-url", default="http://127.0.0.1:8000", help="Dashboard URL")
    args = parser.parse_args()

    run_monitor(args.dashboard_url, args.interval)


if __name__ == "__main__":
    main()
